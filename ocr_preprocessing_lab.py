"""
OCR Preprocessing Lab
---------------------
App Streamlit para experimentar con tecnicas y librerias de Python que
mejoran la calidad de una imagen antes de mandarla a OCR (Claude, Tesseract,
etc.). Permite subir DPI, denoise, contraste, sharpening, binarizacion,
morfologia y deskew, combinandolos en un pipeline configurable.

Lanzar con:
    streamlit run ocr_preprocessing_lab.py
"""

import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import io
import base64

st.set_page_config(page_title="OCR Preprocessing Lab", layout="wide")

# --- Imports opcionales (cada tecnica se desactiva si su lib falta) ---
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from streamlit_image_comparison import image_comparison
    HAS_JUXTAPOSE = True
except ImportError:
    HAS_JUXTAPOSE = False


# ============================================================
# Conversiones PIL <-> OpenCV
# ============================================================
def pil_to_cv2(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    if not HAS_CV2:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


def cv2_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    if HAS_CV2 and arr.shape[2] == 3:
        return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    if HAS_CV2 and arr.shape[2] == 4:
        return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA))
    return Image.fromarray(arr)


# ============================================================
# PDF -> imagen via PyMuPDF
# ============================================================
def rasterize_pdf_page(file_bytes: bytes, page_num: int, dpi: int) -> Image.Image | None:
    if not HAS_FITZ:
        st.error("PyMuPDF no instalado: `pip install pymupdf`")
        return None
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    page = doc[page_num]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).copy()
    doc.close()
    return img


def pdf_page_count(file_bytes: bytes) -> int:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    n = len(doc)
    doc.close()
    return n


# ============================================================
# Operaciones de preprocesado
# ============================================================
PIL_INTERP = {
    "LANCZOS": Image.Resampling.LANCZOS,
    "BICUBIC": Image.Resampling.BICUBIC,
    "BILINEAR": Image.Resampling.BILINEAR,
    "NEAREST": Image.Resampling.NEAREST,
    "HAMMING": Image.Resampling.HAMMING,
}


def upscale_pil(img: Image.Image, scale: float, method: str) -> Image.Image:
    w, h = img.size
    return img.resize((int(w * scale), int(h * scale)), PIL_INTERP[method])


def upscale_cv2(img: Image.Image, scale: float, method: str) -> Image.Image:
    methods = {
        "INTER_CUBIC": cv2.INTER_CUBIC,
        "INTER_LANCZOS4": cv2.INTER_LANCZOS4,
        "INTER_LINEAR": cv2.INTER_LINEAR,
        "INTER_AREA": cv2.INTER_AREA,
        "INTER_NEAREST": cv2.INTER_NEAREST,
    }
    arr = pil_to_cv2(img)
    h, w = arr.shape[:2]
    arr = cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=methods[method])
    return cv2_to_pil(arr)


def to_grayscale(img: Image.Image) -> Image.Image:
    return ImageOps.grayscale(img)


def autocontrast(img: Image.Image, cutoff: int) -> Image.Image:
    src = img.convert("RGB") if img.mode == "RGBA" else img
    return ImageOps.autocontrast(src, cutoff=cutoff)


def equalize(img: Image.Image) -> Image.Image:
    src = img.convert("RGB") if img.mode == "RGBA" else img
    return ImageOps.equalize(src)


def adjust_contrast(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(factor)


def adjust_brightness(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)


def adjust_sharpness(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Sharpness(img).enhance(factor)


def unsharp_mask(img: Image.Image, radius: float, percent: int, threshold: int) -> Image.Image:
    return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))


def gaussian_blur_pil(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def median_filter_pil(img: Image.Image, size: int) -> Image.Image:
    if size % 2 == 0:
        size += 1
    return img.filter(ImageFilter.MedianFilter(size=size))


def clahe_cv2(img: Image.Image, clip_limit: float, tile_grid: int) -> Image.Image:
    arr = pil_to_cv2(img)
    if arr.ndim == 3:
        lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
        l = clahe.apply(l)
        arr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
        arr = clahe.apply(arr)
    return cv2_to_pil(arr)


def bilateral_cv2(img: Image.Image, d: int, sigma_color: int, sigma_space: int) -> Image.Image:
    arr = pil_to_cv2(img)
    arr = cv2.bilateralFilter(arr, d, sigma_color, sigma_space)
    return cv2_to_pil(arr)


def nlmeans_cv2(img: Image.Image, strength: int) -> Image.Image:
    arr = pil_to_cv2(img)
    if arr.ndim == 3:
        arr = cv2.fastNlMeansDenoisingColored(arr, None, strength, strength, 7, 21)
    else:
        arr = cv2.fastNlMeansDenoising(arr, None, strength, 7, 21)
    return cv2_to_pil(arr)


def otsu_threshold(img: Image.Image) -> Image.Image:
    arr = pil_to_cv2(img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2_to_pil(arr)


def adaptive_threshold(img: Image.Image, block_size: int, c: int) -> Image.Image:
    arr = pil_to_cv2(img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if block_size % 2 == 0:
        block_size += 1
    if block_size < 3:
        block_size = 3
    arr = cv2.adaptiveThreshold(
        arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
    )
    return cv2_to_pil(arr)


def morphology(img: Image.Image, op: str, ksize: int, iterations: int) -> Image.Image:
    arr = pil_to_cv2(img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((ksize, ksize), np.uint8)
    ops = {
        "Erode": cv2.MORPH_ERODE,
        "Dilate": cv2.MORPH_DILATE,
        "Open": cv2.MORPH_OPEN,
        "Close": cv2.MORPH_CLOSE,
        "TopHat": cv2.MORPH_TOPHAT,
        "BlackHat": cv2.MORPH_BLACKHAT,
    }
    arr = cv2.morphologyEx(arr, ops[op], kernel, iterations=iterations)
    return cv2_to_pil(arr)


def deskew(img: Image.Image) -> tuple[Image.Image, float]:
    arr = pil_to_cv2(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.ndim == 3 else arr
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        return img, 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.1:
        return img, angle
    h, w = arr.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        arr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return cv2_to_pil(rotated), angle


def invert(img: Image.Image) -> Image.Image:
    src = img.convert("RGB") if img.mode == "RGBA" else img
    return ImageOps.invert(src)


# ============================================================
# Claude OCR (opcional)
# ============================================================
DEFAULT_OCR_PROMPT = (
    "Extract every piece of text visible in this image, preserving layout where "
    "possible (line breaks, columns). Return only the extracted text, no commentary."
)


def call_claude_ocr(img: Image.Image, prompt: str, model: str = "claude-sonnet-4-6") -> str:
    if not HAS_ANTHROPIC:
        return "(anthropic no instalado)"
    key = st.secrets.get("ANTHROPIC_API_KEY", "").strip() if "ANTHROPIC_API_KEY" in st.secrets else ""
    if not key:
        return "(ANTHROPIC_API_KEY no esta en .streamlit/secrets.toml)"

    save_img = img
    if save_img.mode not in ("RGB", "L"):
        save_img = save_img.convert("RGB")
    buf = io.BytesIO()
    save_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return msg.content[0].text


# ============================================================
# UI
# ============================================================
st.title("OCR Preprocessing Lab")
st.caption(
    "Prueba tecnicas de upscaling y preprocesado de imagen (PIL, OpenCV, PyMuPDF) "
    "y compara su impacto sobre OCR con Claude."
)

c1, c2, c3 = st.columns(3)
c1.markdown(f"**OpenCV:** {'OK' if HAS_CV2 else 'no instalado'}")
c2.markdown(f"**PyMuPDF:** {'OK' if HAS_FITZ else 'no instalado'}")
c3.markdown(f"**Anthropic:** {'OK' if HAS_ANTHROPIC else 'no instalado'}")
if not HAS_CV2:
    st.warning("Sin OpenCV las tecnicas cv2 (CLAHE, NLMeans, thresholding, morfologia, deskew) estaran deshabilitadas. Instala: `pip install opencv-python-headless`")

uploaded = st.file_uploader(
    "Sube una imagen (PNG/JPG/TIFF/BMP) o un PDF",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "pdf"],
)

if not uploaded:
    st.info("Sube un archivo para empezar.")
    st.stop()

file_bytes = uploaded.getvalue()
is_pdf = uploaded.name.lower().endswith(".pdf")

# --- Cargar imagen base ---
if is_pdf:
    if not HAS_FITZ:
        st.error("Necesitas PyMuPDF para PDFs: `pip install pymupdf`")
        st.stop()
    n_pages = pdf_page_count(file_bytes)
    col_p, col_d = st.columns(2)
    page_num = col_p.number_input("Pagina", min_value=1, max_value=n_pages, value=1) - 1
    pdf_dpi = col_d.slider(
        "DPI de rasterizacion (PyMuPDF)", min_value=72, max_value=600, value=300, step=25,
        help="Mas alto = imagen mas grande y nitida, pero mas tokens al enviarla a Claude. Sweet spot 200-300.",
    )
    base_img = rasterize_pdf_page(file_bytes, page_num, pdf_dpi)
    st.caption(
        f"PDF rasterizado a {pdf_dpi} DPI -> {base_img.size[0]}x{base_img.size[1]} px"
    )
else:
    base_img = Image.open(io.BytesIO(file_bytes))
    base_img = ImageOps.exif_transpose(base_img)  # respetar rotacion EXIF
    if base_img.mode == "P":
        base_img = base_img.convert("RGB")
    st.caption(f"Imagen cargada: {base_img.size[0]}x{base_img.size[1]} px, modo {base_img.mode}")

# ============================================================
# Sidebar: pipeline configurable
# ============================================================
st.sidebar.header("Pipeline de preprocesado")
st.sidebar.caption("Cada bloque se aplica en orden. Activa los que quieras combinar.")

# --- 1. Upscale ---
with st.sidebar.expander("1. Upscale (subir resolucion)"):
    en_up = st.toggle("Activar", key="en_up")
    up_lib = st.radio("Libreria", ["PIL", "OpenCV"], key="up_lib", horizontal=True, disabled=not en_up)
    if up_lib == "PIL":
        up_method = st.selectbox("Interpolacion", list(PIL_INTERP.keys()), key="up_m_pil", disabled=not en_up)
    else:
        up_method = st.selectbox(
            "Interpolacion",
            ["INTER_CUBIC", "INTER_LANCZOS4", "INTER_LINEAR", "INTER_AREA", "INTER_NEAREST"],
            key="up_m_cv2", disabled=not en_up or not HAS_CV2,
        )
    up_factor = st.slider("Factor de escala", 1.0, 4.0, 2.0, 0.25, key="up_f", disabled=not en_up)

# --- 2. Grayscale ---
with st.sidebar.expander("2. Escala de grises"):
    en_gray = st.toggle("Activar", key="en_gray")

# --- 3. Denoise ---
with st.sidebar.expander("3. Reduccion de ruido"):
    en_den = st.toggle("Activar", key="en_den")
    den_method = st.selectbox(
        "Metodo",
        ["Median (PIL)", "Gaussian (PIL)", "Bilateral (cv2)", "NLMeans (cv2)"],
        key="den_m", disabled=not en_den,
    )
    if den_method == "Median (PIL)":
        den_size = st.slider("Tamano kernel (impar)", 3, 9, 3, 2, key="den_size", disabled=not en_den)
    elif den_method == "Gaussian (PIL)":
        den_radius = st.slider("Radio", 0.5, 5.0, 1.0, 0.5, key="den_rad", disabled=not en_den)
    elif den_method == "Bilateral (cv2)":
        den_d = st.slider("d (diametro)", 5, 15, 9, 2, key="den_d", disabled=not en_den or not HAS_CV2)
        den_sigma_c = st.slider("sigmaColor", 25, 200, 75, 25, key="den_sc", disabled=not en_den or not HAS_CV2)
        den_sigma_s = st.slider("sigmaSpace", 25, 200, 75, 25, key="den_ss", disabled=not en_den or not HAS_CV2)
    else:
        den_h = st.slider("Strength (h)", 3, 30, 10, 1, key="den_h", disabled=not en_den or not HAS_CV2)

# --- 4. Contraste / brillo ---
with st.sidebar.expander("4. Contraste / brillo"):
    en_con = st.toggle("Activar", key="en_con")
    con_method = st.selectbox(
        "Metodo",
        ["AutoContrast (PIL)", "Equalize (PIL)", "CLAHE (cv2)",
         "Contrast factor (PIL)", "Brightness factor (PIL)"],
        key="con_m", disabled=not en_con,
    )
    if con_method == "AutoContrast (PIL)":
        con_cut = st.slider("Cutoff %", 0, 20, 2, 1, key="con_cut", disabled=not en_con)
    elif con_method == "CLAHE (cv2)":
        clahe_clip = st.slider("clipLimit", 0.5, 8.0, 2.0, 0.5, key="con_clip", disabled=not en_con or not HAS_CV2)
        clahe_tile = st.slider("tileGridSize", 2, 16, 8, 2, key="con_tile", disabled=not en_con or not HAS_CV2)
    elif con_method == "Contrast factor (PIL)":
        con_fact = st.slider("Factor", 0.5, 3.0, 1.5, 0.1, key="con_f", disabled=not en_con)
    elif con_method == "Brightness factor (PIL)":
        bri_fact = st.slider("Factor", 0.3, 2.5, 1.0, 0.1, key="bri_f", disabled=not en_con)

# --- 5. Sharpen ---
with st.sidebar.expander("5. Sharpening (enfocar)"):
    en_sh = st.toggle("Activar", key="en_sh")
    sh_method = st.selectbox(
        "Metodo", ["UnsharpMask (PIL)", "Sharpness factor (PIL)"], key="sh_m", disabled=not en_sh,
    )
    if sh_method == "UnsharpMask (PIL)":
        usm_r = st.slider("Radio", 0.5, 5.0, 2.0, 0.5, key="usm_r", disabled=not en_sh)
        usm_p = st.slider("Percent", 50, 300, 150, 10, key="usm_p", disabled=not en_sh)
        usm_t = st.slider("Threshold", 0, 20, 3, 1, key="usm_t", disabled=not en_sh)
    else:
        sh_fact = st.slider("Factor", 0.5, 4.0, 2.0, 0.25, key="sh_f", disabled=not en_sh)

# --- 6. Binarizacion ---
with st.sidebar.expander("6. Binarizacion / threshold"):
    en_th = st.toggle("Activar", key="en_th")
    th_method = st.selectbox(
        "Metodo", ["Otsu (cv2)", "Adaptive Gaussian (cv2)"], key="th_m",
        disabled=not en_th or not HAS_CV2,
    )
    if th_method == "Adaptive Gaussian (cv2)":
        ad_block = st.slider("blockSize (impar)", 3, 51, 15, 2, key="ad_b", disabled=not en_th or not HAS_CV2)
        ad_c = st.slider("Constante C", -10, 30, 10, 1, key="ad_c", disabled=not en_th or not HAS_CV2)

# --- 7. Morfologia ---
with st.sidebar.expander("7. Morfologia (erode/dilate/open/close)"):
    en_m = st.toggle("Activar", key="en_m")
    m_op = st.selectbox(
        "Operacion", ["Erode", "Dilate", "Open", "Close", "TopHat", "BlackHat"],
        key="m_op", disabled=not en_m or not HAS_CV2,
    )
    m_ks = st.slider("Kernel", 1, 9, 3, 2, key="m_ks", disabled=not en_m or not HAS_CV2)
    m_it = st.slider("Iteraciones", 1, 5, 1, 1, key="m_it", disabled=not en_m or not HAS_CV2)

# --- 8. Deskew ---
with st.sidebar.expander("8. Deskew (corregir rotacion)"):
    en_ds = st.toggle("Activar", key="en_ds")

# --- 9. Invertir colores ---
with st.sidebar.expander("9. Invertir colores"):
    en_inv = st.toggle("Activar", key="en_inv")


# ============================================================
# Pipeline
# ============================================================
def apply_pipeline(img: Image.Image):
    steps: list[tuple[str, Image.Image]] = [("Original", img.copy())]
    cur = img.copy()

    if en_up:
        if up_lib == "PIL":
            cur = upscale_pil(cur, up_factor, up_method)
        elif HAS_CV2:
            cur = upscale_cv2(cur, up_factor, up_method)
        steps.append((f"Upscale x{up_factor} ({up_lib} / {up_method})", cur))

    if en_gray:
        cur = to_grayscale(cur)
        steps.append(("Grayscale", cur))

    if en_den:
        if den_method == "Median (PIL)":
            cur = median_filter_pil(cur, den_size)
        elif den_method == "Gaussian (PIL)":
            cur = gaussian_blur_pil(cur, den_radius)
        elif den_method == "Bilateral (cv2)" and HAS_CV2:
            cur = bilateral_cv2(cur, den_d, den_sigma_c, den_sigma_s)
        elif den_method == "NLMeans (cv2)" and HAS_CV2:
            cur = nlmeans_cv2(cur, den_h)
        steps.append((f"Denoise: {den_method}", cur))

    if en_con:
        if con_method == "AutoContrast (PIL)":
            cur = autocontrast(cur, con_cut)
        elif con_method == "Equalize (PIL)":
            cur = equalize(cur)
        elif con_method == "CLAHE (cv2)" and HAS_CV2:
            cur = clahe_cv2(cur, clahe_clip, clahe_tile)
        elif con_method == "Contrast factor (PIL)":
            cur = adjust_contrast(cur, con_fact)
        elif con_method == "Brightness factor (PIL)":
            cur = adjust_brightness(cur, bri_fact)
        steps.append((f"Contraste: {con_method}", cur))

    if en_sh:
        if sh_method == "UnsharpMask (PIL)":
            cur = unsharp_mask(cur, usm_r, usm_p, usm_t)
        else:
            cur = adjust_sharpness(cur, sh_fact)
        steps.append((f"Sharpen: {sh_method}", cur))

    if en_th and HAS_CV2:
        if th_method == "Otsu (cv2)":
            cur = otsu_threshold(cur)
        else:
            cur = adaptive_threshold(cur, ad_block, ad_c)
        steps.append((f"Threshold: {th_method}", cur))

    if en_m and HAS_CV2:
        cur = morphology(cur, m_op, m_ks, m_it)
        steps.append((f"Morfologia: {m_op} (k={m_ks}, it={m_it})", cur))

    if en_ds and HAS_CV2:
        cur, angle = deskew(cur)
        steps.append((f"Deskew (angulo={angle:.2f} grados)", cur))

    if en_inv:
        cur = invert(cur)
        steps.append(("Invertir colores", cur))

    return cur, steps


processed, steps = apply_pipeline(base_img)

# ============================================================
# Visualizacion
# ============================================================
st.subheader("Comparacion")

# Normalizar ambas imagenes a RGB y mismo tamano para el visor juxtapose
def _to_rgb(im: Image.Image) -> Image.Image:
    if im.mode == "1":
        return im.convert("L").convert("RGB")
    if im.mode in ("L", "LA", "P", "RGBA"):
        return im.convert("RGB")
    return im

base_rgb = _to_rgb(base_img)
proc_rgb = _to_rgb(processed)
# Si las dimensiones difieren (por upscale), reescalar el procesado a las del original
# para que el slider se alinee pixel a pixel.
if proc_rgb.size != base_rgb.size:
    proc_rgb_aligned = proc_rgb.resize(base_rgb.size, Image.Resampling.LANCZOS)
else:
    proc_rgb_aligned = proc_rgb

# Selector de modo de visor
viewer_options = ["Slider (deslizar)", "Lado a lado", "Pestanas"]
if not HAS_JUXTAPOSE:
    viewer_options = ["Lado a lado", "Pestanas"]
    st.info("Instala `streamlit-image-comparison` para activar el visor con slider deslizable.")

viewer_mode = st.radio("Modo de visor", viewer_options, horizontal=True, key="viewer_mode")

buf_proc = io.BytesIO()
save_proc = processed if processed.mode in ("RGB", "L", "RGBA") else processed.convert("RGB")
save_proc.save(buf_proc, format="PNG")

caption_left = (
    f"Original: {base_img.size[0]}x{base_img.size[1]} px - modo {base_img.mode} - "
    f"{len(file_bytes)/1024:.1f} KB"
)
caption_right = (
    f"Procesado: {processed.size[0]}x{processed.size[1]} px - modo {processed.mode} - "
    f"{buf_proc.tell()/1024:.1f} KB PNG"
)

if viewer_mode == "Slider (deslizar)":
    # Visor estilo juxtapose: una sola imagen con linea vertical arrastrable.
    # Recomendado en Chrome (renderiza muy fluido).
    slider_width = st.slider("Ancho del visor (px)", 400, 1400, 900, 50, key="slider_width")
    image_comparison(
        img1=base_rgb,
        img2=proc_rgb_aligned,
        label1="Original",
        label2="Procesado",
        width=slider_width,
        starting_position=50,
        show_labels=True,
        make_responsive=True,
        in_memory=True,
    )
    st.caption(caption_left)
    st.caption(caption_right)

elif viewer_mode == "Lado a lado":
    left, right = st.columns(2)
    with left:
        st.markdown("**Original**")
        st.image(base_img, use_container_width=True)
        st.caption(caption_left)
    with right:
        st.markdown("**Procesado**")
        st.image(processed, use_container_width=True)
        st.caption(caption_right)

else:  # Pestanas
    tab_o, tab_p = st.tabs(["Original", "Procesado"])
    with tab_o:
        st.image(base_img, use_container_width=True)
        st.caption(caption_left)
    with tab_p:
        st.image(processed, use_container_width=True)
        st.caption(caption_right)

show_steps = st.checkbox("Mostrar pasos intermedios del pipeline", value=False)
if show_steps and len(steps) > 2:
    for label, step_img in steps[1:]:
        st.markdown(f"**{label}**")
        st.image(step_img, use_container_width=True)

# --- Descarga ---
st.divider()
dl_col1, dl_col2 = st.columns(2)
with dl_col1:
    dl_fmt = st.selectbox("Formato de descarga", ["PNG", "JPEG"])
with dl_col2:
    dl_buf = io.BytesIO()
    img_for_save = processed
    if dl_fmt == "JPEG" and img_for_save.mode not in ("RGB", "L"):
        img_for_save = img_for_save.convert("RGB")
    img_for_save.save(dl_buf, format=dl_fmt, quality=95)
    st.download_button(
        f"Descargar imagen procesada ({dl_fmt})",
        data=dl_buf.getvalue(),
        file_name=f"processed.{dl_fmt.lower()}",
        mime=f"image/{dl_fmt.lower()}",
    )

# ============================================================
# Test OCR con Claude
# ============================================================
st.divider()
st.subheader("Test OCR con Claude")
st.caption(
    "Manda la imagen original y la procesada al mismo prompt y compara resultados. "
    "Requiere `ANTHROPIC_API_KEY` en `.streamlit/secrets.toml`."
)

if not HAS_ANTHROPIC:
    st.warning("Instala `anthropic` para usar esta seccion.")
else:
    model = st.selectbox(
        "Modelo",
        ["claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5-20251001"],
        index=0,
    )
    prompt = st.text_area("Prompt OCR", value=DEFAULT_OCR_PROMPT, height=100)
    run_col1, run_col2 = st.columns(2)

    if run_col1.button("OCR sobre Original", use_container_width=True):
        with st.spinner("Llamando a Claude (original)..."):
            try:
                result = call_claude_ocr(base_img, prompt, model)
                st.session_state["ocr_original"] = result
            except Exception as e:
                st.session_state["ocr_original"] = f"Error: {e}"

    if run_col2.button("OCR sobre Procesado", use_container_width=True):
        with st.spinner("Llamando a Claude (procesado)..."):
            try:
                result = call_claude_ocr(processed, prompt, model)
                st.session_state["ocr_processed"] = result
            except Exception as e:
                st.session_state["ocr_processed"] = f"Error: {e}"

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.markdown("**Resultado original**")
        st.code(st.session_state.get("ocr_original", "(sin resultado)"), language="text")
    with res_col2:
        st.markdown("**Resultado procesado**")
        st.code(st.session_state.get("ocr_processed", "(sin resultado)"), language="text")
