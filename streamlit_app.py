import streamlit as st
import pandas as pd
from supabase import create_client
# pyrefly: ignore [missing-import]
import anthropic
import json
import io
import re
import hashlib
import base64
import traceback
from datetime import datetime
from rapidfuzz import process, fuzz
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image, ImageFilter

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Procesador de Bank Statements - Claude", layout="wide")

if "SUPABASE_URL" not in st.secrets or "SUPABASE_KEY" not in st.secrets:
    st.error("❌ Faltan las credenciales en .streamlit/secrets.toml")
    st.stop()

try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"].strip()
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"].strip()
except Exception as e:
    st.error(f"❌ Error leyendo secrets.toml. Detalle: {e}")
    st.stop()

if not SUPABASE_URL.startswith("https://"):
    st.error("❌ La URL de Supabase es inválida. Debe comenzar con 'https://'.")
    st.stop()

ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"].strip() if "ANTHROPIC_API_KEY" in st.secrets else ""

# --- CONEXIÓN BASE DE DATOS ---
@st.cache_resource
def init_connection():
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.error(f"❌ Error al inicializar cliente Supabase: {e}")
        return None

supabase = init_connection()

# --- CLIENTES Y VENDORS (SUPABASE) ---
@st.cache_data(ttl=300, show_spinner=False)
def fetch_clients():
    if not supabase:
        return []
    try:
        res = supabase.table("clients").select("id, name").order("name").execute()
        return res.data or []
    except Exception as e:
        st.warning(f"⚠️ No se pudo cargar la lista de clientes: {e}")
        return []


@st.cache_data(ttl=300, show_spinner=False)
def fetch_vendors(client_id):
    if not supabase or not client_id:
        return []
    try:
        res = (
            supabase.table("vendors")
            .select("vendor_name, cost_account")
            .eq("client_id", client_id)
            .execute()
        )
        return res.data or []
    except Exception as e:
        st.warning(f"⚠️ No se pudo cargar la lista de vendors: {e}")
        return []


# --- MATCHING DE VENDORS ---
# Tokens genéricos bancarios que NO identifican un vendor real.
# Si tras quitarlos no queda nada con significado, no debemos hacer match.
GENERIC_TOKENS = {
    "CHECK", "CHECKS", "CHK",
    "BANKCARD", "BANKCD",
    "MERCHANT",
    "DEPOSIT", "DEPOSITS", "DEP",
    "WITHDRAWAL", "WITHDRAW", "WD",
    "PAYMENT", "PMT", "PAY", "PAID",
    "ACH",
    "ATM",
    "POS",
    "DEBIT", "DB",
    "CREDIT", "CR",
    "TRANSFER", "TRF", "XFER",
    "WIRE",
    "ZELLE",
    "BANK",
    "CARD",
    "ONLINE",
    "FROM", "TO",
    "INC", "LLC", "LTD", "CORP", "CORPORATION", "COMPANY", "CO",
    "PURCHASE", "PURCHASES",
    "REF", "TRACE", "ID",
    "USD",
}


def normalize_vendor_name(s):
    """Normaliza un nombre para matching:
    - uppercase, sin puntuación, whitespace colapsado
    - elimina tokens puramente numéricos (números de cheque, refs)
    - elimina tokens genéricos bancarios (CHECK, BANKCARD, ACH, ...)
    - requiere al menos un token de 3+ caracteres para considerarse válido
    """
    if s is None:
        return ""
    s = str(s).upper().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = [
        t for t in s.split()
        if t not in GENERIC_TOKENS and not t.isdigit()
    ]
    if not any(len(t) >= 3 for t in tokens):
        return ""
    return " ".join(tokens)


def build_vendor_index(vendors):
    index = {}
    for v in vendors:
        name = v.get("vendor_name")
        account = v.get("cost_account")
        if not name:
            continue
        norm = normalize_vendor_name(name)
        if norm and norm not in index:
            index[norm] = (name, account)
    return index


def match_vendor(query, vendor_index, vendor_names_normalized, threshold=85):
    """Returns (matched_vendor_name, cost_account, score).
    score=100 for exact match, fuzzy score for fuzzy match, 0 for no match.
    """
    norm_query = normalize_vendor_name(query)
    if not norm_query:
        return (None, None, 0)

    if norm_query in vendor_index:
        original_name, account = vendor_index[norm_query]
        return (original_name, account, 100)

    if not vendor_names_normalized:
        return (None, None, 0)

    best = process.extractOne(
        norm_query,
        vendor_names_normalized,
        scorer=fuzz.WRatio,
    )
    if best is not None:
        matched_norm, score, _ = best
        if score >= threshold:
            original_name, account = vendor_index[matched_norm]
            return (original_name, account, int(score))

    return (None, None, 0)


def build_assignment_df(final_df, vendors):
    debit_mask = pd.to_numeric(final_df.get("Amount Debit"), errors="coerce").fillna(0) > 0
    debits = final_df[debit_mask].copy()
    if debits.empty:
        debits["Matched Vendor"] = []
        debits["Cost Account"] = []
        debits["Score"] = []
        return debits

    vendor_index = build_vendor_index(vendors)
    vendor_names_normalized = list(vendor_index.keys())

    matched_names = []
    accounts = []
    scores = []
    for _, row in debits.iterrows():
        source = row.get("Source") or ""
        if not str(source).strip():
            source = row.get("Description") or ""
        matched_name, account, score = match_vendor(source, vendor_index, vendor_names_normalized)
        matched_names.append(matched_name if matched_name else "")
        accounts.append(account if account else "")
        scores.append(score)

    debits["Matched Vendor"] = matched_names
    debits["Cost Account"] = accounts
    debits["Score"] = scores
    return debits


# --- AUTENTICACIÓN ---
def login_user(email, password):
    if not supabase:
        return None
    try:
        return supabase.auth.sign_in_with_password({"email": email, "password": password})
    except Exception as e:
        error_msg = str(e)
        if "Invalid API key" in error_msg:
            st.error("🚨 ERROR CRÍTICO: La 'SUPABASE_KEY' es incorrecta. Usa la clave 'anon'/'public'.")
        elif "[Errno 8]" in error_msg or "nodename nor servname" in error_msg:
            st.error("❌ Error de Conexión: No se encuentra el servidor de Supabase.")
        else:
            st.error(f"Error de autenticación: {e}")
        return None

# --- HISTORIAL ---
def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

def save_to_history(filename, file_hash, raw_data):
    try:
        supabase.table("processed_files").insert({
            "filename": filename,
            "file_hash": file_hash,
            "raw_data": raw_data
        }).execute()
        st.toast("✅ Archivo guardado en historial")
    except Exception as e:
        st.warning(f"⚠️ No se pudo guardar en el historial (Error DB): {e}")

# --- PIPELINE A: preprocesado OCR para PDFs escaneados ---
# Receta minima validada en el lab: rasterizar a 300 DPI -> grayscale -> CLAHE -> UnsharpMask.
# No binariza (Claude lee mejor grayscale con contraste reforzado que blanco/negro puro).
PIPELINE_DPI = 300


def pdf_is_native(file_bytes: bytes) -> bool:
    """True si el PDF tiene capa de texto en alguna pagina (no necesita preprocesado)."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        try:
            for page in doc:
                if page.get_text().strip():
                    return True
            return False
        finally:
            doc.close()
    except Exception:
        return False  # ante la duda, tratar como escaneado


def apply_pipeline_a(img: Image.Image) -> Image.Image:
    """Grayscale + CLAHE(clip=2.0, tile=8) + UnsharpMask(r=1.5, p=150, t=3)."""
    arr = np.array(img)
    if arr.ndim == 3:
        # PIL trae RGB; OpenCV espera el mismo orden para RGB2GRAY
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    arr = clahe.apply(arr)
    out = Image.fromarray(arr, mode="L")
    out = out.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=3))
    return out


def rasterize_and_preprocess(file_bytes: bytes, dpi: int = PIPELINE_DPI) -> list[bytes]:
    """Rasteriza cada pagina del PDF al DPI indicado y aplica Pipeline A.
    Devuelve lista de bytes PNG, una por pagina."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pages: list[bytes] = []
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).copy()
            processed = apply_pipeline_a(img)
            buf = io.BytesIO()
            processed.save(buf, format="PNG", optimize=True)
            pages.append(buf.getvalue())
        return pages
    finally:
        doc.close()


# --- EXTRACCIÓN CON CLAUDE ---
EXTRACTION_INSTRUCTIONS = """You are an expert OCR and financial data extraction AI.
Your task is to extract every single transaction from this bank statement and return a JSON array.

CRITICAL INSTRUCTIONS TO FIX COMMON ERRORS:

1. MISSING TRANSACTIONS: 
- Process the document strictly page by page, row by row. 
- DO NOT skip any rows or summarize. ONE ROW PER TRANSACTION.
- Use your thinking process to count the rows as you go. 
- If a page begins with a transaction row without a section header, it is a continuation of the previous section. Extract it!

2. WRONG COLUMNS (DEBIT VS CREDIT):
- Always map amounts to Debit or Credit based on the column headers at the top of the table on the current page, or the section title.
- Withdrawals, Payments, Purchases, Checks → Amount Debit (money leaving account).
- Deposits, Additions, Credits → Amount Credit (money entering account).
- "FROM" typically means money is entering (Credit). "TO" means money is leaving (Debit).
- If an entire section is titled "Deposits", everything in it is a Credit. Pay attention to the layout!

3. NUMBER CONFUSION (OCR ERRORS 1 vs 7):
- Pay meticulous attention to digits. A '1' is a vertical line (sometimes with a small serif). A '7' has a distinct horizontal top bar.
- '0' vs 'O', '8' vs 'B'. Look at the resolution and surrounding context.
- If amounts don't make sense (e.g. math doesn't add up), look closer at the image.
- Transcribe challenging rows verbatim in your thinking block before outputting them.

OUTPUT FORMAT:
Return a JSON array of objects. Each object must have these EXACT keys:

- "Date": use MM/DD/YY format
- "Description": the full transaction description as it appears
- "Source": extract a clean, readable name of who sent/received the money (e.g. strip "POS DB", "WITHDRAWAL". "Zelle Payment To Juan" → "Zelle - Juan")
- "Ref Number": for checks, the check number (e.g. "3245"). Otherwise leave blank.
- "Account Number": the account number for this transaction as it appears. Repeat if only one account exists.
- "Amount Debit": the withdrawal/debit amount as a positive number (leave blank if not applicable)
- "Amount Credit": the deposit/credit amount as a positive number (leave blank if not applicable)
- "Class": MUST be exactly one of: 
  * "Credit Card" (Stripe, Square, Toast, Clover, Doordash, Uber Eats, autopay)
  * "Transfer" (ACH, wire, Zelle, Online Transfer, ODP)
  * "Check" (paid by check, has check number)
  * "Cash" (everything else: vendor ACH, tax, payroll, ATM)

IMPORTANT:
- Do not include opening balance, ending balance, or summary rows.
- No markdown formatting (no ```json). No preamble. ONLY output the JSON array."""


AUDITOR_INSTRUCTIONS = """You are an expert Financial Auditor AI.
Your task is to REVIEW and CORRECT a JSON array of extracted bank statement transactions against the original PDF document.

CRITICAL INSTRUCTIONS:
1. MISSING TRANSACTIONS: Read the document page by page and compare it to the provided JSON. If the OCR missed any transactions, ADD them to the array.
2. WRONG COLUMNS: Verify the 'Amount Debit' and 'Amount Credit' for every transaction. If a withdrawal is accidentally in the credit column, MOVE it to Debit. Use the mathematical balance and section headers to verify.
3. OCR ERRORS: Check for numbers that were misread (e.g. 7 vs 1, 0 vs O, 8 vs B). If the provided JSON has a suspicious amount, verify it against the image.
4. REMOVE DUPLICATES: If there are duplicate rows, keep only one.

The user will provide you with the PDF document and the CURRENT JSON extraction as a text block.
Your output must be the FINAL, CORRECTED JSON array. No markdown fences, no preamble, only the raw JSON array."""

def repair_incomplete_json(s):
    s = s.strip()
    
    # Clean up markdown fences if present
    s = re.sub(r"^```(?:json)?\n?", "", s)
    s = re.sub(r"\n?```$", "", s)
    s = s.strip()

    # Track state of container structures and string boundaries
    stack = []
    in_string = False
    escape = False
    
    clean_chars = []
    
    for i, char in enumerate(s):
        if escape:
            clean_chars.append(char)
            escape = False
            continue
            
        if char == '\\':
            clean_chars.append(char)
            escape = True
            continue
            
        if char == '"':
            in_string = not in_string
            clean_chars.append(char)
            continue
            
        if in_string:
            clean_chars.append(char)
            continue
            
        if char == '{':
            stack.append('}')
            clean_chars.append(char)
        elif char == '[':
            stack.append(']')
            clean_chars.append(char)
        elif char == '}':
            if stack and stack[-1] == '}':
                stack.pop()
                clean_chars.append(char)
        elif char == ']':
            if stack and stack[-1] == ']':
                stack.pop()
                clean_chars.append(char)
        else:
            clean_chars.append(char)

    reconstructed = "".join(clean_chars)
    
    try:
        return json.loads(reconstructed)
    except Exception:
        pass
        
    # Attempt to truncate at the last complete JSON object closing brace
    last_complete_idx = reconstructed.rfind('}')
    if last_complete_idx != -1:
        part = reconstructed[:last_complete_idx + 1].strip()
        if part.endswith(','):
            part = part[:-1].strip()
        if part.startswith('['):
            part += ']'
        elif not part.startswith('{'):
            if s.startswith('['):
                part = '[' + part + ']'
            elif s.startswith('{'):
                part = '{' + part + '}'
                
        try:
            return json.loads(part)
        except Exception:
            pass
            
    # Fallback to closing all open container elements
    reconstructed_closed = reconstructed
    if in_string:
        reconstructed_closed += '"'
    reconstructed_closed = re.sub(r',\s*$', '', reconstructed_closed)
    for container in reversed(stack):
        reconstructed_closed += container
        
    try:
        return json.loads(reconstructed_closed)
    except Exception:
        pass
        
    raise ValueError("Could not repair JSON")


def robust_json_loads(s):
    try:
        return json.loads(s)
    except Exception:
        pass
    
    # Try removing trailing commas inside arrays or objects
    s_cleaned = re.sub(r',\s*([\]}])', r'\1', s)
    try:
        return json.loads(s_cleaned)
    except Exception:
        pass
        
    return repair_incomplete_json(s)


def process_file_with_claude(uploaded_file):
    if not ANTHROPIC_API_KEY:
        st.error("❌ Falta configurar ANTHROPIC_API_KEY en secrets.toml")
        return []

    file_bytes = uploaded_file.getvalue()
    file_hash = get_file_hash(file_bytes)

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        # Decide envio: PDF nativo -> 'document'; escaneado -> Pipeline A + 'image' bloques
        native = pdf_is_native(file_bytes)
        if native:
            st.write("📄 PDF nativo detectado → envío directo a Claude (sin preprocesado).")
            content_blocks = [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": base64.b64encode(file_bytes).decode("utf-8"),
                    },
                }
            ]
        else:
            st.write(
                f"📷 PDF escaneado detectado → aplicando Pipeline A "
                f"({PIPELINE_DPI} DPI + grayscale + CLAHE + UnsharpMask)."
            )
            page_pngs = rasterize_and_preprocess(file_bytes, dpi=PIPELINE_DPI)
            st.write(f"  ↳ {len(page_pngs)} página(s) preprocesada(s).")
            content_blocks = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(p).decode("utf-8"),
                    },
                }
                for p in page_pngs
            ]

        content_blocks.append({
            "type": "text",
            "text": "Extract all transactions from this bank statement following the instructions.",
        })

        st.write("Enviando a Claude para extracción...")
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=40000,
            timeout=1200.0,
            thinking={
                "type": "enabled",
                "budget_tokens": 10000,
            },
            system=[
                {
                    "type": "text",
                    "text": EXTRACTION_INSTRUCTIONS,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": content_blocks}],
        )

        st.write("✅ Extracción completada.")
        # Con thinking activado, el array content puede tener bloques de tipo
        # "thinking" antes del bloque "text". Buscamos el texto explícitamente.
        text_block = next((b for b in response.content if b.type == "text"), None)
        
        if not text_block:
            st.error("❌ Claude no devolvió un bloque de texto en la respuesta.")
            st.write("Contenido de la respuesta:")
            st.write(response.content)
            return []
        
        raw_text = text_block.text.strip()

        raw_data = robust_json_loads(raw_text)

        if raw_data:
            save_to_history(uploaded_file.name, file_hash, raw_data)
            return parse_response(raw_data)

        return []

    except json.JSONDecodeError as e:
        st.error(f"❌ Error al parsear la respuesta JSON de Claude: {e}")
        st.code(raw_text)
        return []
    except Exception as e:
        st.error(f"Error general: {e}")
        st.code(traceback.format_exc())
        return []


def run_ai_validation(uploaded_file, current_json_data):
    if not ANTHROPIC_API_KEY:
        st.error("❌ Falta configurar ANTHROPIC_API_KEY en secrets.toml")
        return []

    file_bytes = uploaded_file.getvalue()
    file_base64 = base64.b64encode(file_bytes).decode("utf-8")
    
    current_json_str = json.dumps(current_json_data, indent=2)

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        st.write("Enviando archivo y datos actuales a Claude para auditoría...")

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=40000,
            timeout=1200.0,
            thinking={
                "type": "enabled",
                "budget_tokens": 10000,
            },
            system=[
                {
                    "type": "text",
                    "text": AUDITOR_INSTRUCTIONS,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": file_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": f"Here is the CURRENT JSON extraction:\n{current_json_str}\n\nPlease review it against the PDF document, fix any errors, add missing transactions, and output the CORRECTED JSON array.",
                        },
                    ],
                }
            ],
        )

        st.write("✅ Auditoría completada.")
        text_block = next((b for b in response.content if b.type == "text"), None)
        
        if not text_block:
            st.error("❌ Claude no devolvió un texto válido en la auditoría.")
            return []
        
        raw_text = text_block.text.strip()
        raw_data = robust_json_loads(raw_text)

        if raw_data:
            return parse_response(raw_data)

        return []

    except Exception as e:
        st.error(f"Error general en validación: {e}")
        st.code(traceback.format_exc())
        return []


def clean_currency(value_str):
    if pd.isna(value_str):
        return None
    s = str(value_str).strip()
    s = s.replace('$', '').replace('€', '').replace(' ', '')
    s = s.replace('O', '0').replace('o', '0')
    s = s.replace('l', '1').replace('I', '1')
    s = s.replace('S', '5')
    s = s.replace(',', '')
    match = re.search(r'-?\d+(\.\d+)?', s)
    if match:
        try:
            return float(match.group())
        except Exception:
            return None
    return None


def parse_response(raw_records):
    if isinstance(raw_records, str):
        try:
            raw_records = robust_json_loads(raw_records)
        except Exception:
            pass

    if raw_records is None:
        return []
    if isinstance(raw_records, dict) and 'data' in raw_records:
        raw_records = raw_records['data']
    if not isinstance(raw_records, list):
        return []

    processed_data = []
    for item in raw_records:
        if isinstance(item, dict):
            # Extract date (flexible keys)
            tx_date = item.get('Date') or item.get('Transaction date') or item.get('date') or ""
            
            # Extract description
            tx_desc = item.get('Description') or item.get('Transaction description') or item.get('description') or ""
            
            # Extract source / names
            tx_source = item.get('Source') or item.get('Names') or item.get('names') or item.get('source') or ""
            
            # Extract ref number
            ref_num = item.get('Ref Number') or item.get('Ref number') or item.get('ref_number') or ""
            
            # Extract account number
            acc_num = item.get('Account Number') or item.get('Account number') or item.get('account_number') or ""
            
            # Extract class / type
            tx_class = item.get('Class') or item.get('Transaction type') or item.get('class') or ""
            
            # Extract debit and credit amounts
            amount_debit = item.get('Amount Debit') or item.get('Amount debit') or item.get('amount_debit') or ""
            amount_credit = item.get('Amount Credit') or item.get('Amount credit') or item.get('amount_credit') or ""
            
            # Check if older format key 'Amount' was used as fallback
            old_amount = item.get('Amount') or item.get('amount')
            if old_amount and not amount_debit and not amount_credit:
                cleaned_amt = clean_currency(old_amount)
                if tx_class.lower() in ['credit card', 'deposit']:
                    amount_credit = cleaned_amt
                else:
                    amount_debit = cleaned_amt
        elif isinstance(item, list):
            tx_date = str(item[0]) if len(item) >= 1 else ""
            tx_desc = str(item[1]) if len(item) >= 2 else ""
            tx_source = str(item[2]) if len(item) >= 3 else ""
            ref_num = str(item[3]) if len(item) >= 4 else ""
            acc_num = str(item[4]) if len(item) >= 5 else ""
            amount_debit = str(item[5]) if len(item) >= 6 else ""
            amount_credit = str(item[6]) if len(item) >= 7 else ""
            tx_class = str(item[7]) if len(item) >= 8 else ""
        else:
            continue

        formatted_date = tx_date
        if tx_date:
            try:
                formatted_date = pd.to_datetime(tx_date).strftime('%m/%d/%y')
            except Exception:
                pass

        processed_data.append({
            "Date": formatted_date,
            "Description": tx_desc,
            "Source": tx_source,
            "Ref Number": ref_num,
            "Account Number": acc_num,
            "Amount Debit": clean_currency(amount_debit) if amount_debit != "" else "",
            "Amount Credit": clean_currency(amount_credit) if amount_credit != "" else "",
            "Class": tx_class,
        })

    return processed_data


def main():
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    if not st.session_state['authenticated']:
        st.header("🔐 Iniciar Sesión")
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Contraseña", type="password")
            if st.form_submit_button("Entrar"):
                user = login_user(email, password)
                if user:
                    st.session_state['authenticated'] = True
                    st.session_state['user_email'] = email
                    st.rerun()
        return

    st.sidebar.title(f"Usuario: {st.session_state.get('user_email')}")
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state['authenticated'] = False
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Cliente")
    clients = fetch_clients()
    if clients:
        client_options = {c["name"]: c["id"] for c in clients}
        labels = ["— Selecciona un cliente —"] + list(client_options.keys())
        current_id = st.session_state.get("selected_client_id")
        current_label = next(
            (name for name, cid in client_options.items() if cid == current_id),
            labels[0],
        )
        selected_label = st.sidebar.selectbox(
            "Cliente para asignación de cuentas",
            labels,
            index=labels.index(current_label) if current_label in labels else 0,
        )
        new_id = client_options.get(selected_label)
        if new_id != st.session_state.get("selected_client_id"):
            st.session_state["selected_client_id"] = new_id
            st.session_state.pop("assignment_df", None)
    else:
        st.sidebar.caption("Sin clientes disponibles.")

    st.title("📄 Procesador de Bank Statements")
    st.subheader("Carga de Bank Statements (PDF)")
    uploaded_file = st.file_uploader("Sube el archivo PDF", type=['pdf'])

    if uploaded_file:
        if st.button("Procesar"):
            with st.spinner("Procesando con Claude..."):
                raw_data = process_file_with_claude(uploaded_file)

                if not raw_data:
                    st.warning("No se obtuvieron datos.")
                else:
                    st.session_state['processed_results'] = pd.DataFrame(raw_data)
                    st.session_state.pop("assignment_df", None)
                    st.success(f"Registros listos: {len(st.session_state['processed_results'])}")

    if 'processed_results' in st.session_state:
        final_df = st.session_state['processed_results']
        st.divider()

        tab_full, tab_assign = st.tabs(["📋 Vista completa", "💼 Asignación de cuentas"])

        with tab_full:
            # --- MÉTRICAS DE RESUMEN ---
            total_tx = len(final_df)

            def sum_col(df, col):
                if col not in df.columns:
                    return 0.0
                numeric = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                return numeric.sum()

            total_credit = sum_col(final_df, "Amount Credit")
            total_debit  = sum_col(final_df, "Amount Debit")

            m1, m2, m3 = st.columns(3)
            m1.metric("🔢 Transacciones", f"{total_tx:,}")
            m2.metric("💚 Total Créditos", f"${total_credit:,.2f}")
            m3.metric("🔴 Total Débitos",  f"${total_debit:,.2f}")

            st.divider()

            if uploaded_file:
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("🕵️ Validar y Corregir con IA", use_container_width=True):
                        with st.spinner("Claude está auditando los datos contra el PDF original. Esto tomará un momento..."):
                            current_data = final_df.to_dict(orient='records')
                            # Limpiar NaN values para que no rompan json.dumps
                            current_data = [
                                {k: (None if pd.isna(v) else v) for k, v in row.items()}
                                for row in current_data
                            ]
                            corrected_data = run_ai_validation(uploaded_file, current_data)
                            if corrected_data:
                                st.session_state['processed_results'] = pd.DataFrame(corrected_data)
                                st.session_state.pop("assignment_df", None)
                                st.success("¡Validación terminada! La tabla ha sido actualizada con las correcciones.")
                                st.rerun()

            edited_df = st.data_editor(final_df, num_rows="dynamic", key="results_editor", use_container_width=True)

        with tab_assign:
            selected_client_id = st.session_state.get("selected_client_id")
            if not selected_client_id:
                st.info("Selecciona un cliente en la barra lateral para ver las asignaciones.")
                assignment_df = None
            else:
                vendors = fetch_vendors(selected_client_id)
                if not vendors:
                    st.warning("Este cliente no tiene vendors registrados en Supabase.")
                assignment_df = build_assignment_df(edited_df, vendors)

                total_debits = len(assignment_df)
                if total_debits == 0:
                    st.info("No hay débitos en esta extracción.")
                else:
                    assigned = (assignment_df["Score"] > 0).sum()
                    pct = (assigned / total_debits * 100) if total_debits else 0
                    c1, c2, c3 = st.columns(3)
                    c1.metric("🔢 Débitos", f"{total_debits:,}")
                    c2.metric("✅ Asignados", f"{assigned:,}")
                    c3.metric("📊 Cobertura", f"{pct:.1f}%")

                    st.dataframe(
                        assignment_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Score": st.column_config.ProgressColumn(
                                "Score",
                                help="100 = match exacto. ≥85 = fuzzy aceptado. 0 = sin asignar.",
                                min_value=0,
                                max_value=100,
                                format="%d",
                            ),
                        },
                    )

        # --- EXCEL DOWNLOAD MULTI-HOJA ---
        def autosize_columns(df, worksheet):
            for idx, col in enumerate(df.columns):
                series = df[col].astype(str)
                if not series.empty:
                    max_len = min(max(series.map(len).max(), len(str(col))) + 1, 50)
                else:
                    max_len = len(str(col)) + 1
                worksheet.column_dimensions[chr(65 + idx)].width = max_len

        def filter_with_amount(df, column):
            if column not in df.columns:
                return df.iloc[0:0]
            numeric = pd.to_numeric(df[column], errors='coerce').fillna(0)
            return df[numeric > 0].copy()

        credits_df = filter_with_amount(edited_df, "Amount Credit")
        debits_df = filter_with_amount(edited_df, "Amount Debit")

        if st.session_state.get("selected_client_id"):
            vendors_for_excel = fetch_vendors(st.session_state["selected_client_id"])
            debits_df = build_assignment_df(edited_df, vendors_for_excel)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            edited_df.to_excel(writer, index=False, sheet_name='Transactions')
            autosize_columns(edited_df, writer.sheets['Transactions'])

            credits_df.to_excel(writer, index=False, sheet_name='Credits')
            autosize_columns(credits_df, writer.sheets['Credits'])

            debits_df.to_excel(writer, index=False, sheet_name='Debits')
            autosize_columns(debits_df, writer.sheets['Debits'])

        st.download_button(
            "⬇️ Descargar XLSX",
            data=output.getvalue(),
            file_name=f"bank_statement_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        if st.button("Limpiar Resultados"):
            del st.session_state['processed_results']
            st.session_state.pop("assignment_df", None)
            st.rerun()


if __name__ == "__main__":
    main()
