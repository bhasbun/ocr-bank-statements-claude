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
EXTRACTION_INSTRUCTIONS = """Extract all transactions from this bank statement and return them as a 
            table with these exact columns: Date, Description, Source, Ref Number, 
            Account Number, Amount Debit, Amount Credit, Class.
             
            Rules:
             
            - Date: use MM/DD/YY format
             
            - Description: the full transaction description as it appears in the statement
             
            - Source: extract a clean, readable name of who sent or received the money.
              For example:
              * "WITHDRAWAL -TOLTECA FOODS Tolteca FO FT423437204" → "Tolteca Foods"
              * "Orig CO Name:Toast Orig ID:1201361000..." → "Toast"
              * "Orig CO Name:Uber USA 6787..." → "Uber Eats"
              * "WITHDRAWAL -IRS USATAXPYMT" → "IRS"
              * "Zelle Payment To Juan Jose..." → "Zelle - Juan Jose"
              * "WITHDRAWAL -ATT PAYMENT..." → "AT&T"
              * "POS DB SUPERLO FO..." → "Superlo Foods"
              * "Online Transfer To Chk...3920" → "Internal Transfer - Acct 3920"
              * For checks, use the payee name visible on the check image if available,
                otherwise leave blank
              Strip out all reference numbers, trace numbers, account codes, 
              and technical identifiers. Return only the human-readable name.
             
            - Ref Number: 
              * For checks: the check number (e.g. "3245", "15087")
              * For all other transactions: leave blank
             
            - Account Number: the account number associated with this transaction 
              as it appears on the statement (e.g. "00220006508677", 
              "000000203031197"). If the document contains multiple accounts, 
              use the correct account number for each transaction. 
              If only one account exists in the document, repeat it for every row.
             
            - Amount Debit: the withdrawal/debit amount as a positive number, 
              leave blank if not applicable. If the document has columns or sections for withdrawals/debits take care from this.
              * "Online Transfer to..." → Amount Debit (money leaving the account)
              * "Zelle to..." → Amount Debit
              * "WF Direct Pay-Payment-..." → Amount Debit
              * "Business to Business ACH Debit..." → Amount Debit
             
            - Amount Credit: the deposit/credit amount as a positive number, 
              leave blank if not applicable. If the document has columns or sections for deposits/credits take care from this.
              * "Bankcard Dep..." → Amount Credit
              * "Doordash, Inc...." → Amount Credit
              * "Online Transfer From..." → Amount Credit
              * "ODP Transfer From..." → Amount Credit
             
            - Class: classify each transaction using ONLY these four values:
              * "Credit Card" — for any transaction involving payment processors 
                or credit card activity, including: Stripe, Square, Toast, EPX, 
                Merchant Bankcd, Clover, Doordash, Uber Eats, Grubhub, Chase 
                Credit Card, credit card autopay, or any merchant settlement
              * "Transfer" — for any online transfer, ACH, wire, Zelle, ODP 
                transfer, internal bank transfer between accounts, or any 
                transaction labeled "Transfer", "Online Transfer", "ACH", or "Zelle"
              * "Check" — for any transaction paid by check, identified by a 
                check number in the Checks Paid section or in the account history
              * "Cash" — for everything else, including vendor ACH payments, 
                tax payments, payroll, utilities, insurance, rent, subscriptions, 
                and ATM withdrawals
             
            Important:
            - Include every single transaction without exception
            - Do not skip any rows or summarize — one row per transaction
            - Do not include opening balance, ending balance, or summary rows
            - Checks always go in Class "Check", never in "Transfer" or "Cash"
            - Never skip the first transaction row of any page. if a page begin with a transaction row (no section header), extract it. It is a continuation of the previous section.
            - The word "FROM" in a transfer description always means money is entering the account → Amount Credit.
            - The word "TO" in a transfer description always means money is leaving the account → Amount Debit.
            - This applies regardless of how the transaction is labeled (Transfer, Zelle, Online Transfer, Wire, etc.)
            - Use the section header of the statement (Deposits/Credits vs Withdrawals/Debits) to determine the correct column for any ambiguous transaction. Example: "Deposited OR Cashed Check" please check the column or section (Debit or Credit).
            - If a page does not have a section header, assume it is a continuation of the last active section from the previous page. Use the section context to determine the correct amount column.
            -No markdown, no explanation, no extra text — only the JSON array."""



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
        file_base64 = base64.b64encode(file_bytes).decode("utf-8")
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        st.write("Enviando archivo a Claude para extracción...")

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=16000,
            thinking={
                "type": "enabled",
                "budget_tokens": 5000,
            },
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
                            "text": "Extract all transactions from this bank statement following the instructions.",
                        },
                    ],
                }
            ],
        )

        st.write("✅ Extracción completada.")
        # Con thinking activado, el array content puede tener bloques de tipo
        # "thinking" antes del bloque "text". Buscamos el texto explícitamente.
        text_block = next((b for b in response.content if b.type == "text"), None)
        if not text_block:
            st.error("❌ Claude no devolvió un bloque de texto en la respuesta.")
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

        edited_df = st.data_editor(final_df, num_rows="dynamic", key="results_editor", use_container_width=True)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            edited_df.to_excel(writer, index=False, sheet_name='Transactions')
            worksheet = writer.sheets['Transactions']
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
