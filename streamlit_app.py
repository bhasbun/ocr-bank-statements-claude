import streamlit as st
import pandas as pd
from supabase import create_client, Client
# pyrefly: ignore [missing-import]
import anthropic
import json
import io
import re
import hashlib
import base64
from datetime import datetime

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Procesador de Bank Statements", layout="wide")

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

# --- EXTRACCIÓN CON CLAUDE ---
EXTRACTION_INSTRUCTIONS = (
    "Role: Bank statement transaction extractor for business checking statements (Chase and other banks).\n\n"
    "Goals:\n"
    "- Extract ONLY transactions listed under the deposits section of the statement.\n"
    "- STOP extracting immediately when the deposits section ends.\n"
    "- The deposits section ends when any of the following section headers appear: \"CHECKS PAID\", "
    "\"ELECTRONIC WITHDRAWALS\", \"FEES\", \"OTHER WITHDRAWALS\", \"DEBITS\", or any other new section "
    "header that is not a continuation of deposits.\n"
    "- Do NOT extract any checks, withdrawals, fees, or any transaction from any other section under any circumstance.\n"
    "- Extract each deposit as a separate row.\n\n"
    "Instructions:\n"
    "1. Transaction date: Use the date shown in the DATE column. Format as MM/DD/YYYY.\n"
    "2. Transaction description: Use the full description text as shown in the statement.\n"
    "3. Amount: Always positive (deposits). Preserve exact amounts including decimals.\n"
    "4. Transaction type: Always \"Deposit\" for all rows.\n"
    "5. Account number: Use the account number shown on the statement.\n"
    "6. Names: Apply the following rules to determine the name:\n"
    "   - If the description mentions any credit card processor → \"credit card\"\n"
    "   - Credit card processors include (but are not limited to): Toast, Square, Stripe, PayPal, Heartland, "
    "Clover, Fiserv, WorldPay, Braintree, Adyen, Elavon, First Data, Global Payments, TSYS, Payroc, Paysafe, "
    "Gravity Payments, Shift4, Fattmerchant, Stax, Payanywhere, Dharma, PaySimple, iZettle, SumUp, Zettle, "
    "Poynt, Vend, Lightspeed, Revel, NCR, Aloha, Micros, Silverware, Omnivore, Breadcrumb, TouchBistro, "
    "Upserve, Lavu, Cake, Harbortouch, Talech, Shopkeep, Bindo, Epos Now, Nobly, Tillpoint, Erply, Hike, "
    "Kounta, iConnect, Springboard, AccuPOS, pcAmerica, CAM Commerce, Aldelo, Dinerware, Digital Dining, "
    "Focus POS, Future POS, Positouch, Prism, Squirrel, Xpient, Pixel Point\n"
    "   - If the description mentions \"Doordash\", \"DoorDash\", \"Uber\", \"Grubhub\", \"Instacart\", "
    "\"EZCater\", \"Caviar\", or \"Postmates\" → \"cash\"\n"
    "   - If the description is a plain deposit (e.g. \"Deposit 1295512367\") → \"cash\"\n"
    "   - If the description mentions \"ODP Transfer\", \"Online Transfer\", or any internal bank transfer → \"cash\"\n"
    "   - For any other deposit not covered above → \"cash\"\n\n"
    "High-Importance Rules:\n"
    "- ONLY extract rows from the deposits section. Stop at any new section header.\n"
    "- Do NOT extract check numbers, check amounts, or anything from the \"CHECKS PAID\" section.\n"
    "- Do not skip any deposit row within the deposits section.\n"
    "- Names must only be \"cash\" or \"credit card\" — no other values allowed.\n"
    "- Ignore completely any row that appears after the deposits section ends.\n\n"
    "Return ONLY a valid JSON array of objects. Each object must have exactly these keys:\n"
    "\"Transaction date\", \"Transaction description\", \"Amount\", \"Transaction type\", \"Account number\", \"Names\".\n"
    "No markdown, no explanation, no extra text — only the JSON array."
)


def process_file_with_claude(uploaded_file):
    if not ANTHROPIC_API_KEY:
        st.error("❌ Falta configurar ANTHROPIC_API_KEY en secrets.toml")
        return []

    file_bytes = uploaded_file.getvalue()
    file_hash = get_file_hash(file_bytes)

    try:
        file_base64 = base64.b64encode(file_bytes).decode("utf-8")
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        st.write("Enviando archivo a Claude para extracción...")

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": EXTRACTION_INSTRUCTIONS,
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
                            "text": "Extract the deposit transactions from this bank statement following the instructions.",
                        },
                    ],
                }
            ],
        )

        st.write("✅ Extracción completada.")
        raw_text = response.content[0].text.strip()

        # Strip markdown code fences if present
        raw_text = re.sub(r"^```(?:json)?\n?", "", raw_text)
        raw_text = re.sub(r"\n?```$", "", raw_text)

        raw_data = json.loads(raw_text)

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
            raw_records = json.loads(raw_records)
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
            tx_date = item.get('Transaction date') or ""
            tx_desc = item.get('Transaction description') or ""
            raw_amount = item.get('Amount') or "0"
            tx_type = item.get('Transaction type') or ""
            acc_num = item.get('Account number') or ""
            names = item.get('Names') or ""
        elif isinstance(item, list):
            tx_date = str(item[0]) if len(item) >= 1 else ""
            tx_desc = str(item[1]) if len(item) >= 2 else ""
            raw_amount = str(item[2]) if len(item) >= 3 else "0"
            tx_type = str(item[3]) if len(item) >= 4 else ""
            acc_num = str(item[4]) if len(item) >= 5 else ""
            names = str(item[5]) if len(item) >= 6 else ""
        else:
            continue

        formatted_date = tx_date
        if tx_date:
            try:
                formatted_date = pd.to_datetime(tx_date).strftime('%m/%d/%Y')
            except Exception:
                pass

        processed_data.append({
            "Transaction date": formatted_date,
            "Transaction description": tx_desc,
            "Amount": clean_currency(raw_amount),
            "Transaction type": tx_type,
            "Account number": acc_num,
            "Names": names,
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
                    st.success(f"Registros listos: {len(st.session_state['processed_results'])}")

    if 'processed_results' in st.session_state:
        final_df = st.session_state['processed_results']
        st.divider()
        st.write("### Resultados Extraídos")

        edited_df = st.data_editor(final_df, num_rows="dynamic", key="results_editor", use_container_width=True)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            edited_df.to_excel(writer, index=False, sheet_name='Deposits')
            worksheet = writer.sheets['Deposits']
            for idx, col in enumerate(edited_df.columns):
                series = edited_df[col]
                max_len = min(max(series.astype(str).map(len).max(), len(str(col))) + 1, 50)
                worksheet.column_dimensions[chr(65 + idx)].width = max_len

        st.download_button(
            "⬇️ Descargar XLSX",
            data=output.getvalue(),
            file_name=f"bank_statement_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        if st.button("Limpiar Resultados"):
            del st.session_state['processed_results']
            st.rerun()


if __name__ == "__main__":
    main()
