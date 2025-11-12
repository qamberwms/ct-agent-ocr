from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
import fitz  # PyMuPDF
import re, tempfile
from jinja2 import Template
import numpy as np
import cv2
from paddleocr import PaddleOCR

# ----------------- 英文模板 -----------------
TEMPLATE = Template(
    "In {{ protocol_version_date or 'NA' }}, a {{ allocation or 'NA' }}, "
    "{{ masking or 'NA' }}, {{ design or 'NA' }}, {{ phase or 'NA' }} clinical trial "
    "({{ test_protocol_number or 'NA' }}; {{ trial_number or 'NA' }}) was planned to start "
    "in {{ country or 'China' }} to evaluate the use of {{ intervention or 'NA' }} "
    "versus {{ placebo or 'NA' }} in {{ 'healthy subjects' if healthy_subjects else 'patients' }} "
    "(expected n = {{ sample_size or 'NA' }}) for the treatment of {{ indication or 'NA' }}. "
    "The primary endpoints were {{ endpoints or 'NA' }}. "
    "As of {{ first_public_date or (protocol_version_date or 'NA') }}, it was not yet "
    "opened for patient enrollment."
)

# ----------------- 文本读取：直读优先，失败回退 OCR -----------------
def read_text(pdf_bytes: bytes) -> str:
    """
    先尝试直接读取 PDF 文字层；若文本过少则自动回退到 OCR（PaddleOCR）。
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    # 1) 直接读文字层
    parts = [(p.get_text("text") or "").strip() for p in doc]
    text = "\n".join([x for x in parts if x])
    if len(text) >= 200:
        return text

    # 2) 回退 OCR
    ocr = PaddleOCR(lang="ch", show_log=False)
    pages = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=300)
        # 转成 OpenCV 图像
        arr = np.frombuffer(pix.tobytes("png"), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            cv2.imwrite(tmp.name, img)
            res = ocr.ocr(tmp.name, cls=True)
            lines = [ln[1][0] for ln in (res[0] or [])]
            pages.append("\n".join(lines))
    return "\n".join(pages)

# ----------------- 抽取 & 归一化 -----------------
def to_half(s):
    return s.replace("\u3000"," ").replace("\xa0"," ").strip() if s else s

def kv(text, labels):
    if isinstance(labels, str): labels=[labels]
    pat = r"(?:%s)\s*[：:]\s*([^\n\r]+)" % "|".join(map(re.escape, labels))
    m = re.search(pat, text, flags=re.I)
    return to_half(m.group(1)) if m else None

def std_date(s):
    if not s: return None
    s = s.strip()
    m = re.search(r"(\d{4})[./-]?(\d{1,2})", s)
    if m: return f"{m.group(1)}-{m.group(2).zfill(2)}"
    m = re.search(r"(\d{4})", s)
    return f"{m.group(1)}-01" if m else s

def normalize_phase(s):
    if not s: return None
    low = s.lower()
    mapping = {
        "i期":"Phase I","ii期":"Phase II","iii期":"Phase III","iv期":"Phase IV",
        "一期":"Phase I","二期":"Phase II","三期":"Phase III","四期":"Phase IV",
        "phase i":"Phase I","phase ii":"Phase II","phase iii":"Phase III","phase iv":"Phase IV"
    }
    for k,v in mapping.items():
        if k in low: return v
    return s

def norm_alloc(s):
    if not s: return None
    low = s.lower()
    if "非随机" in low: return "non-randomized"
    if "随机" in low:    return "randomized"
    return s

def norm_mask(s):
    if not s: return None
    low = s.lower()
    if "双盲" in low: return "double-blind"
    if "单盲" in low: return "single-blind"
    if "开放" in low or "非盲" in low: return "open-label"
    return s

def block(text, titles):
    # 先尝试“标题: 内容”块抓取到下一个标题（以“xxx：”样式）
    for t in titles:
        m = re.search(rf"{re.escape(t)}[：:]\s*(.+?)(?:\n\S+：|\Z)", text, flags=re.S)
        if m:
            c = re.sub(r"[\r\n]+","；", m.group(1))
            return re.sub(r"[；;]{2,}","；", c).strip("； ")
    # 否则退化为简单键值对
    return kv(text, titles)

def extract(text: str) -> dict:
    t = to_half(text)
    data = {
        "trial_number": kv(t, ["注册号","登记号","备案号"]),
        "test_protocol_number": kv(t, ["试验方案编号","方案编号"]),
        "phase": normalize_phase(kv(t, ["试验分期","分期"])),
        "design": kv(t, ["设计类型","研究设计","试验设计"]),
        "allocation": norm_alloc(kv(t, ["随机化","随机"])),
        "masking": norm_mask(kv(t, ["盲法"])),
        "sample_size": None,
        "indication": kv(t, ["适应症","疾病","病种"]),
        "healthy_subjects": None,
        "intervention": kv(t, ["试验药","干预措施"]),
        "placebo": kv(t, ["对照药","对照"]),
        "endpoints": None,
        "sites": kv(t, ["研究中心","研究单位"]),
        "ethics": kv(t, ["伦理审查结论","伦理"]),
        "protocol_version_date": std_date(kv(t, ["版本日期","方案版本日期"])),
        "country": kv(t, ["国家"]) or "China",
        "first_public_date": std_date(kv(t, ["首次公示日期","首次公开日期","公示日期"])),
    }
    # 样本量数字
    ss = kv(t, ["目标入组人数","目标样本量","预计入组"])
    if ss:
        m = re.search(r"(\d{1,5})", ss.replace(",", ""))
        data["sample_size"] = m.group(1) if m else ss
    # 健康受试者
    hs = kv(t, ["是否包括健康受试者"])
    if hs:
        data["healthy_subjects"] = ("是" in hs) or ("包含" in hs) or ("健康" in hs)
    # 终点：主+次拼接
    prim = block(t, ["主要终点","主要终点指标"])
    sec  = block(t, ["次要终点","次要终点指标"])
    endpoints = "；".join([x for x in [prim, sec] if x])
    data["endpoints"] = endpoints or None
    return data

def render(data): 
    return TEMPLATE.render(**data)

# ----------------- FastAPI -----------------
app = FastAPI(title="CT Agent (OCR Version)")

@app.post("/summarize")
async def summarize(file: UploadFile = File(...)):
    pdf = await file.read()
    text = read_text(pdf)
    if not text or len(text) < 50:
        return JSONResponse({"error":"Failed to read PDF text (OCR may have failed)."}, status_code=400)
    data = extract(text)
    return JSONResponse({"summary": render(data), "fields": data})

@app.get("/")
def root():
    return PlainTextResponse("CT Agent OCR online. POST /summarize with form-data file=@trial.pdf")

