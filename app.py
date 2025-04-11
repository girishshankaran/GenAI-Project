# --- START OF FILE app.py ---

import re
import os
# import requests # Keep if using API mode, remove if only local
import logging
from datetime import datetime
# Make sure to import these for the admin routes & flash messages
from flask import Flask, render_template, request, url_for, redirect, flash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from collections import Counter # Import Counter

# --- Plotly and Counter for Graphing ---
try:
    import plotly
    import plotly.graph_objs as go
    import json
    PLOTLY_AVAILABLE = True
except ImportError:
    print("Warning: Plotly not installed. Graphing feature disabled. Run: pip install plotly pandas")
    PLOTLY_AVAILABLE = False


# --- spaCy Import and Model Loading ---
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        print("spaCy model 'en_core_web_sm' loaded successfully.")
    except OSError:
        print("spaCy model 'en_core_web_sm' not found.")
        print("Please run: python -m spacy download en_core_web_sm")
        nlp = None
except ImportError:
    print("WARNING: spacy library not installed. NER detection will be disabled.")
    print("Install using: pip install spacy")
    nlp = None
# --- End spaCy Setup ---

# --- Local LLM Setup (llama-cpp-python) ---
LLM_MODE = "local" # Set to "local" or "api"
LOCAL_MODEL_PATH = "/Users/gisankar/Documents/GenAI-Project/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf" # Using Absolute Path
llm_local = None

if LLM_MODE == "local":
    try:
        from llama_cpp import Llama
        if os.path.exists(LOCAL_MODEL_PATH):
            try:
                print(f"Loading local Llama model from: {LOCAL_MODEL_PATH}")
                llm_local = Llama( model_path=LOCAL_MODEL_PATH, n_ctx=2048, n_threads=None, n_gpu_layers=0, verbose=False ) # Using n_gpu_layers=0 for CPU focus
                print(f"Local Llama model loaded successfully (CPU mode).")
            except Exception as e: print(f"!!! ERROR loading local Llama model: {e}"); import traceback; traceback.print_exc(); llm_local = None
        else: print(f"!!! ERROR: Local model file not found at {LOCAL_MODEL_PATH}"); llm_local = None
    except ImportError: print("WARNING: llama-cpp-python library not installed."); llm_local = None
# --- End Local LLM Setup ---


load_dotenv()

# --- PDF/DOCX Imports ---
try: import PyPDF2
except ImportError: PyPDF2=None; print("PyPDF2 disabled.")
try: import docx
except ImportError: docx=None; print("python-docx disabled.")

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-should-be-complex')

# --- Audit Logging Setup ---
LOG_FILE_PATH = 'audit.log'
audit_logger = logging.getLogger('audit'); audit_logger.setLevel(logging.INFO)
audit_handler = logging.FileHandler(LOG_FILE_PATH, encoding='utf-8'); audit_handler.setLevel(logging.INFO)
audit_formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'); audit_handler.setFormatter(audit_formatter)
if not audit_logger.hasHandlers(): audit_logger.addHandler(audit_handler)

# --- Configuration ---
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
def allowed_file(filename): return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Text Extraction Helper Functions ---
def extract_text_from_txt(file_stream):
    try: file_stream.seek(0); return file_stream.read().decode('utf-8', errors='ignore')
    except Exception as e: print(f"Error reading TXT: {e}"); return None
def extract_text_from_pdf(file_stream):
    if not PyPDF2: return None; text = "";
    try:
        file_stream.seek(0); reader = PyPDF2.PdfReader(file_stream)
        if reader.is_encrypted: print("PDF is encrypted, cannot extract text."); return "[ERROR: PDF is encrypted]"
        for page in reader.pages:
            page_text = page.extract_text();
            if page_text: text += page_text + "\n"
        return text if text else None
    except PyPDF2.errors.PdfReadError as e: print(f"PDF ReadError: {e}."); return None
    except Exception as e: print(f"PDF Error: {e}"); return None
def extract_text_from_docx(file_stream):
    if not docx: return None; text = "";
    try:
        file_stream.seek(0); document = docx.Document(file_stream)
        for para in document.paragraphs: text += para.text + "\n"
        return text if text else None
    except Exception as e: print(f"DOCX Error: {e}"); return None

# --- LLM Query Function (Local Llama Implementation) ---
def query_llm(text):
    global llm_local
    if LLM_MODE != "local" or not llm_local: return "[LLM Query Skipped - Local model not configured or loaded]"
    if not text or not isinstance(text, str) or not text.strip(): return "[LLM Query Skipped - Input text is empty or invalid]"
    system_prompt = "You are a helpful assistant. Process the user's text and provide a relevant response."
    user_prompt = text
    prompt_string = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    print(f"Generating response using local model: {LOCAL_MODEL_PATH}")
    generated_text = "[Error: Local LLM generation failed]"
    try:
        output = llm_local( prompt_string, max_tokens=250, stop=["<|eot_id|>", "<|end_of_text|>", "<|end_header_id|>", "assistant<|end_header_id|>"], temperature=0.7, top_p=0.9, echo=False )
        if output and 'choices' in output and output['choices']:
            response_content = output['choices'][0]['text']; print(f"Local LLM Raw Response Content: {response_content[:200]}...")
            generated_text = response_content.strip()
        else: print(f"Could not parse response from local LLM output: {output}"); generated_text = "[Error: Local LLM response format unexpected]"
    except Exception as e: print(f"An unexpected error occurred during local LLM generation: {e}"); import traceback; traceback.print_exc(); generated_text = f"[An unexpected error occurred during local LLM generation: {str(e)}]"
    return generated_text

# --- Redaction Functions ---
# *** UPDATED PLACEHOLDER_MAP: Removed direct mappings for ignored spaCy types ***
PLACEHOLDER_MAP = {
    # Specific Types (matched by precise regex or NER)
    "Email Address": "[EMAIL]", "Phone Number Format (US)": "[PHONE]", "SSN Format (US)": "[SSN]",
    "Potential SSN (Loose Format)": "[SSN_LIKE]", "Potential Credit Card": "[CREDIT_CARD]",
    "API Key Format": "[API_KEY]", "Private Key Block": "[PRIVATE_KEY]", "Internal/Local IP": "[IP_ADDRESS]",
    "UUID Format": "[UUID]", "Secret Assignment": "[SECRET_VALUE]", "Password Assignment": "[PASSWORD_VALUE]",
    "Credential Assignment": "[CREDENTIAL_VALUE]", "Username Assignment": "[USERNAME_VALUE]",
    "Bank Account Number": "[BANK_ACCOUNT]", "Date Format (MM/DD/YY or YYYY)": "[DATE]",
    "Project Codename": "[PROJECT_CODENAME]", "Potential Generic Key/Token": "[GENERIC_KEY]",
    "JSON Key Pair": "[JSON_SECRET_PAIR]", "Value in Parentheses": "[SECRET_IN_PARENS]",
    "Value near Keyword": "[POTENTIAL_VALUE]", # Kept for proximity scan logic
    "PERSON": "[PERSON_NAME]", # Keep PERSON as it's PII
    "SSN Keyword": "[SSN_MENTIONED]", # Keep keyword overrides
    "DOB Keyword": "[DOB_MENTIONED]",
    # Removed: GPE, LOC, ORG, DATE, MONEY, NORP, FAC etc. as direct keys
    # Category Mappings (fallback if Type not in map)
    "Credentials": "[CREDENTIALS_INFO]",
    "Personal Data (PII)": "[PII_INFO]",
    "Potential PII/Contextual": "[CONTEXTUAL_INFO]", # Category for things like MONEY if kept
    "Confidential/Secret": "[CONFIDENTIAL_INFO]",
    "Legal": "[LEGAL_INFO]",
    "Intellectual Property": "[IP_INFO]",
    "Network": "[NETWORK_INFO]",
    "Identifier": "[IDENTIFIER_INFO]",
    "Contextual": "[CONTEXTUAL_INFO]", # Catch-all for less sensitive NER/Keywords if needed
    "Keyword": "[SENSITIVE_KEYWORD]", # Fallback for unmapped Keywords
}
def get_placeholder(finding):
    ftype = finding.get('type', ''); fcategory = finding.get('category', '')
    if ftype in PLACEHOLDER_MAP: return PLACEHOLDER_MAP[ftype]
    if fcategory in PLACEHOLDER_MAP: return PLACEHOLDER_MAP[fcategory]
    if ftype == 'Keyword': return PLACEHOLDER_MAP.get('Keyword', '[KEYWORD]')
    sanitized_category = re.sub(r'[^\w\s-]', '', fcategory).strip().upper().replace(' ','_')
    return f"[{sanitized_category or 'INFO'}]"
def redact_text(text, findings):
    if not findings or not text: return text
    sorted_findings = sorted(findings, key=lambda f: (f['span'][0], -f['span'][1]))
    redacted = ""; last_end = 0; processed_indices = set()
    for finding in sorted_findings:
        start, end = finding['span']
        if not (0 <= start < end <= len(text)): continue
        if any(i in processed_indices for i in range(start, end)): continue
        if start >= last_end: redacted += text[last_end:start]
        else: print(f"Warning: Adjusting start due to potential finding overlap issue. Finding: {finding}")
        placeholder = get_placeholder(finding); redacted += placeholder; last_end = end
        for i in range(start, end): processed_indices.add(i)
    if last_end < len(text): redacted += text[last_end:]
    return redacted

# --- Confidentiality Check Function ---
def perform_confidentiality_check(text): # Includes skipping CARDINAL & Ignore List
    if not text or not isinstance(text, str): return {"found_issues": False, "findings": []}
    findings = []; processed_text_positions = set()
    original_text = text; lower_text = original_text.lower()
    def add_finding(category, type, value, description, start_index, end_index):
        if not (0 <= start_index < end_index <= len(original_text)): return
        span = (start_index, end_index)
        if any(i in processed_text_positions for i in range(start_index, end_index)): return
        findings.append({"category": category, "type": type, "value": value, "description": description, "span": span})
        for i in range(start_index, end_index): processed_text_positions.add(i)

    # *** Ignore list for common technical/generic terms ***
    IGNORE_TERMS_SPACY = {"lcm", "gcd", "api", "cpu", "gpu", "ram", "http", "https",
                          "json", "xml", "sql", "html", "css", "k8s", "url", "uri",
                          "fed", } # Added "fed" based on example
    # *** Labels to ignore from spaCy unless overridden ***
    IGNORE_SPACY_LABELS = {"CARDINAL", "ORG", "MONEY", "GPE", "LOC", "DATE",
                           "NORP", "FAC", "QUANTITY", "PRODUCT", "EVENT",
                           "WORK_OF_ART", "LAW", "LANGUAGE"}

    # 1. spaCy NER Processing
    spacy_findings_temp = []
    if nlp:
        try:
            doc = nlp(original_text)
            for ent in doc.ents:
                label = ent.label_; category = "Unknown"; finding_type = label; entity_text = ent.text; entity_text_lower = entity_text.lower()

                # --- Rule-based Override / Ignore List ---
                if entity_text_lower in IGNORE_TERMS_SPACY:
                    print(f"DEBUG: Ignoring common technical term/acronym classified by spaCy as {label}: '{entity_text}'")
                    continue # Skip this entity entirely

                # Handle specific known sensitive acronyms if misclassified
                # This check runs *before* the general label ignore check
                is_overridden_sensitive = False
                if label in ["ORG", "MISC", "PRODUCT"] :
                    if entity_text_lower == "ssn":
                        category = "Personal Data (PII)"; finding_type = "SSN Keyword"
                        is_overridden_sensitive = True
                    elif entity_text_lower == "dob":
                         category = "Personal Data (PII)"; finding_type = "DOB Keyword"
                         is_overridden_sensitive = True

                # --- Skip common entity labels UNLESS overridden ---
                if label in IGNORE_SPACY_LABELS and not is_overridden_sensitive:
                    print(f"DEBUG: Skipping common spaCy {label} entity: '{entity_text}'")
                    continue # Don't process this entity further
                # --- End Skip Logic ---

                # Map remaining essential labels (only PERSON left as sensitive by default)
                if label == "PERSON": category = "Personal Data (PII)"; finding_type = "PERSON"
                # Add other labels ONLY if they haven't been categorized yet AND you want them for context (not sensitive)
                elif category == "Unknown": # Only if not already categorized by override
                    # You could potentially map other non-ignored labels to "Contextual" here if needed
                    # category = "Contextual"
                    pass # For now, we ignore unmapped labels if not caught by overrides/specific list

                # Add if category was determined (i.e., PERSON or an override like SSN/DOB)
                if category != "Unknown":
                    spacy_findings_temp.append({"category": category, "type": finding_type, "value": entity_text, "description": f"Detected '{label}' entity (via spaCy NER).", "span": (ent.start_char, ent.end_char)})
        except Exception as e: print(f"Error during spaCy processing: {e}")
    else: print("spaCy NER skipped as model is not loaded.")

    # 2. Keyword Scanning
    keyword_categories = { # Full lists
        "Legal": [r"attorney-client\s+privilege", r"legal\s+hold", r"litigation", r"confidential\s+settlement", r"under\s+seal", r"privileged\s+and\s+confidential", r"cease\s+and\s+desist", r"nda", r"non-disclosure\s+agreement"],
        "Confidential/Secret": [r"confidential", r"proprietary", r"secrets?", r"internal\s+use\s+only", r"trade\s+secret", r"classified", r"sensitive", r"do\s+not\s+distribute", r"private", r"restricted", r"not\s+for\s+public\s+release", r"passwords?", r"secret keys?", r"api keys?", r"credentials?", r"tokens?"],
        "Intellectual Property": [r"patent\s+pending", r"patent\s+application", r"trademark", r"copyright", r"invention\s+disclosure", r"prototype", r"roadmap", r"research\s+findings", r"algorithm", r"proprietary\s+algorithm"],
        "Personal Data (PII)": [r"dob", r"date\s+of\s+birth", r"passport\s+number", r"driver'?s\s+license", r"address", r"ssn", r"social security numbers?", r"credit cards?", r"bank accounts?", r"email addresses?", r"phone numbers?"],
    }
    temp_keyword_findings = []; keyword_processed_positions_check = set()
    if lower_text:
        for category, keywords in keyword_categories.items():
            for keyword_pattern in keywords:
                try:
                    for match in re.finditer(r'\b' + keyword_pattern + r'\b', lower_text, re.IGNORECASE):
                        start, end = match.start(), match.end()
                        if not any(pos in keyword_processed_positions_check for pos in range(start, end)):
                             temp_keyword_findings.append({"category": category, "type": "Keyword", "value": original_text[start:end], "description": f"Detected keyword: '{original_text[start:end]}'.", "span": (start, end)})
                             for i in range(start, end): keyword_processed_positions_check.add(i)
                except re.error as e: print(f"Warning: Skipping invalid regex keyword: '{keyword_pattern}' - {e}")

    # --- 3. Keyword Proximity Scan REMOVED ---
    potential_value_findings = [] # Keep list empty

    # --- 4. Regex Pattern Matching ---
    regex_patterns = [ # Full list
        (r'\b(password|passwd|secret|pwd|pass)\b(?:\s+(?:is|was|are|be)\s+|\s*[:=]\s*|\s+)(\S+)', "Password Assignment", "Credentials", "Potential password assignment.", re.IGNORECASE),
        (r'\b(api[\._]?key|access[\._]?key|secret[\._]?key|token|credential|auth_token)\b\s*[:=]\s*(?:\{|"|\'|)([^\s;\'"\}]+)(?:\}|"|\'|;)', "Credential Assignment", "Credentials", "Potential hardcoded key/token assignment in code.", re.IGNORECASE),
        (r'\b(user(?: |_)name|user|login|user_?id)\s*[:=]\s*\S+', "Username Assignment", "Credentials", "Potential username or login ID assignment.", re.IGNORECASE),
        (r'\b(secrets?|keys?|tokens?|passwords?|credentials?|values?)\b\s*\(([a-zA-Z0-9\-_+=]{6,})\)', "Value in Parentheses", "Credentials", "Potential secret value enclosed in parentheses after keyword.", re.IGNORECASE),
        (r'\b(sk_live|pk_live|rk_live|sk_test|pk_test|rk_test)_[0-9a-zA-Z]{24,}\b', "API Key Format", "Credentials", "Common API key format (e.g., Stripe)."),
        (r'-----BEGIN (RSA|OPENSSH|PGP|DSA|EC) PRIVATE KEY-----.*?-----END \1 PRIVATE KEY-----', "Private Key Block", "Credentials", "Private key block detected.", re.DOTALL | re.IGNORECASE),
        (r'\b[a-zA-Z0-9\+\/]{40,}\b', "Potential Generic Key/Token", "Credentials", "High entropy string (potential key/token)."),
        (r'"(?:access_key|secret_key|api_key|token)"\s*:\s*"([^"]+)"', "JSON Key Pair", "Credentials", "Potential sensitive keys in JSON.", re.IGNORECASE),
        (r'\b(ssn|social security(?: number)?)\s*(?:is|:|=)?\s*(\d{3}-?\d{2}-?\d{4})\b', "SSN Format (US)", "Personal Data (PII)", "US SSN pattern with indicator.", re.IGNORECASE),
        (r'(?<!\d-)\b(\d{3}-\d{2}-\d{4})\b(?!\d)', "SSN Format (US)", "Personal Data (PII)", "US SSN pattern (standalone)."),
        (r'(?<!\d-)\b(\d{7}|\d{9})\b(?!\d)', "Potential SSN (Loose Format)", "Personal Data (PII)", "7 or 9 digits (standalone, potential SSN)."),
        (r'\b(credit card|cc|card number|cc#|card no)\s*(?:is|:|=)?\s*(\d[\d -]{11,18}\d)\b', "Potential Credit Card", "Personal Data (PII)", "Potential Credit Card Number with indicator.", re.IGNORECASE),
        (r'\b(\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4})\b', "Potential Credit Card", "Personal Data (PII)", "16 digits in groups of 4."),
        (r'\b(bank account(?: number| no)?|account no|account number|acct no)\s*(?:is|:|=|,?\s+which\s+is)?\s*(\d[\d\s-]{5,}\d?)', "Bank Account Number", "Personal Data (PII)", "Potential bank account number.", re.IGNORECASE),
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "Email Address", "Personal Data (PII)", "Email address format."),
        (r'\(?\b\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', "Phone Number Format (US)", "Personal Data (PII)", "US phone number format."),
        (r'\b(0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])[-/](?:19|20)?\d{2}\b', "Date Format (MM/DD/YY or YYYY)", "Personal Data (PII)", "Date format (MM/DD/YYYY)."),
        (r'\b(192\.168(?:\.\d{1,3}){2}|10(?:\.\d{1,3}){3}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}|127(?:\.\d{1,3}){3})\b', "Internal/Local IP", "Network", "Internal/localhost IP address."),
        (r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b', "UUID Format", "Identifier", "UUID format."),
        (r'\b(?:Project|Codename|Initiative)[ -_]([A-Z][a-zA-Z0-9]+(?:[ -_][A-Z][a-zA-Z0-9]+)*)\b', "Project Codename", "Intellectual Property", "Potential project codename."),
    ]
    regex_findings_temp = []
    if original_text:
        for item in regex_patterns:
            pattern, type, category, description = item[0], item[1], item[2], item[3]; flags = item[4] if len(item) > 4 else 0
            try:
                 for match in re.finditer(pattern, original_text, flags=flags):
                      value = match.group(0); start, end = match.start(), match.end()
                      if value and start < end: regex_findings_temp.append({"category": category, "type": type, "value": value, "description": description, "span": (start, end)})
            except re.error as e: print(f"Warning: Skipping invalid regex pattern: {pattern} - {e}")
            except IndexError as e: print(f"Warning: Index error during regex: {pattern} - {e}")

    # --- 5. Add Findings in Simplified Priority Order ---
    # Priority: Specific Regex -> Keywords -> spaCy NER (Proximity scan removed)
    for finding in regex_findings_temp: add_finding(finding["category"], finding["type"], finding["value"], finding["description"], finding["span"][0], finding["span"][1])
    for finding in temp_keyword_findings: add_finding(finding["category"], finding["type"], finding["value"], finding["description"], finding["span"][0], finding["span"][1])
    # Proximity values removed
    for finding in spacy_findings_temp: add_finding(finding["category"], finding["type"], finding["value"], finding["description"], finding["span"][0], finding["span"][1])
    # --- End Finding Addition ---

    sorted_findings = sorted(findings, key=lambda f: (f['span'][0], -f['span'][1]))
    return {"found_issues": len(sorted_findings) > 0, "findings": sorted_findings}


# --- Helper function to read and parse audit log ---
def read_parse_audit_log():
    log_entries = []; error = None
    try:
        if os.path.exists(LOG_FILE_PATH):
            with open(LOG_FILE_PATH, 'r', encoding='utf-8') as f:
                lines = f.readlines(); lines.reverse()
                for line_num, line in enumerate(lines, 1):
                    clean_line = line.strip();
                    if not clean_line: continue
                    entry = {'timestamp': 'N/A', 'user': 'N/A', 'model': 'N/A', 'findings': 'N/A', 'original_excerpt': 'N/A', 'redacted_excerpt': 'N/A', 'error_msg': 'N/A'}
                    try:
                        ts_part, rest_of_line = clean_line.split(' | ', 1); entry['timestamp'] = ts_part.strip()
                        pattern = re.compile(r"User:(?P<user>.*?)\s*\|\s*Model:(?P<model>.*?)\s*\|\s*Findings:(?P<findings>.*?)\s*\|\s*OriginalExcerpt:(?P<original_excerpt>.*?)\s*\|\s*RedactedExcerpt:(?P<redacted_excerpt>.*?)\s*\|\s*Error:(?P<error_msg>.*)")
                        match = pattern.search(rest_of_line)
                        if match:
                            gd = match.groupdict()
                            entry['user'] = gd.get('user', 'Parse Error').strip(); entry['model'] = gd.get('model', 'Parse Error').strip(); entry['findings'] = gd.get('findings', 'Parse Error').strip()
                            entry['original_excerpt'] = gd.get('original_excerpt', 'Parse Error').strip(); entry['redacted_excerpt'] = gd.get('redacted_excerpt', 'Parse Error').strip(); entry['error_msg'] = gd.get('error_msg', 'Parse Error').strip()
                            log_entries.append(entry)
                        else: print(f"Warning: Regex failed to parse fields on log line {line_num}: '{rest_of_line}'")
                    except ValueError: print(f"Warning: Skipping malformed log line {line_num} (missing initial ' | '): '{clean_line}'")
                    except Exception as parse_e: print(f"Warning: Could not parse log line {line_num}: '{clean_line}' - Error: {parse_e}")
        else: print(f"Audit log file '{LOG_FILE_PATH}' not found.")
    except Exception as e: print(f"Error reading audit log file: {e}"); error = f"Could not read audit log file: {str(e)}"
    return log_entries, error

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def home():
    results = None; original_text_display = ""; uploaded_filename = None; error = None; llm_response = None; text_sent_to_llm = None; source_description = "No input processed yet."; placeholders_used = None; user_identifier = "anonymous"
    if request.method == 'POST':
        text_input_from_area = request.form.get('text_to_analyze', ''); file = request.files.get('file_to_analyze'); content_to_analyze = None
        # --- Input Processing ---
        if file and file.filename:
            if allowed_file(file.filename):
                uploaded_filename = secure_filename(file.filename); print(f"Processing uploaded file: {uploaded_filename}"); source_description = f"Uploaded file '{uploaded_filename}'"; file_ext = uploaded_filename.rsplit('.', 1)[1].lower()
                try:
                    if file_ext == 'txt': content_to_analyze = extract_text_from_txt(file.stream)
                    elif file_ext == 'pdf': content_to_analyze = extract_text_from_pdf(file.stream)
                    elif file_ext == 'docx': content_to_analyze = extract_text_from_docx(file.stream)
                    if content_to_analyze is None or content_to_analyze == "[ERROR: PDF is encrypted]":
                        error_reason = "Check file content/corruption/libraries.";
                        if content_to_analyze == "[ERROR: PDF is encrypted]": error_reason = "PDF is password-protected."
                        error = f"Failed to extract text from '{uploaded_filename}'. {error_reason}"; original_text_display = text_input_from_area; content_to_analyze = None
                    else: original_text_display = ""
                except Exception as e: print(f"Error processing file {uploaded_filename}: {e}"); error = f"An error occurred processing file '{uploaded_filename}'."; original_text_display = text_input_from_area
            else: error = f"Invalid file type '{file.filename}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}."; original_text_display = text_input_from_area
        elif text_input_from_area: print("Processing text from input area."); content_to_analyze = text_input_from_area; original_text_display = text_input_from_area; source_description = "Text Input"
        else: error = "Please enter text or upload an allowed document (.txt, .pdf, .docx)."

        log_model = "N/A"; log_orig_excerpt = "N/A"; log_red_excerpt = "N/A"; log_num_findings = 0; log_error = error if error else ""
        if content_to_analyze is not None and error is None:
            if not isinstance(content_to_analyze, str): content_to_analyze = str(content_to_analyze)
            log_orig_excerpt = (content_to_analyze[:100] + '...') if len(content_to_analyze) > 100 else content_to_analyze
            print(f"Performing confidentiality check on content from {source_description}..."); results = perform_confidentiality_check(content_to_analyze); log_num_findings = len(results.get('findings', [])); print(f"Check complete. Found issues: {results['found_issues']}")
            if results['found_issues']:
                print("Redacting text before sending to LLM..."); used_placeholders_set = set(); findings_to_process = results.get('findings', [])
                if findings_to_process:
                    for finding in findings_to_process: used_placeholders_set.add(get_placeholder(finding))
                placeholders_used = sorted(list(used_placeholders_set)); text_sent_to_llm = redact_text(content_to_analyze, results['findings']); log_red_excerpt = (text_sent_to_llm[:100] + '...') if len(text_sent_to_llm) > 100 else text_sent_to_llm
            else: print("No issues found needing redaction, sending original text."); text_sent_to_llm = content_to_analyze; log_red_excerpt = log_orig_excerpt; placeholders_used = None
            log_model = "Local Llama" if LLM_MODE == "local" and llm_local else ("API_" + API_URL.split('/')[-1] if LLM_MODE == "api" and HF_API_KEY and '/models/' in API_URL else "LLM_Disabled/Unknown")
            llm_response = query_llm(text_sent_to_llm)
            if llm_response and llm_response.startswith(("[Error", "[LLM Info")): log_error += f" LLM Query: {llm_response}"
            print("LLM query finished.")
        elif error: results = None; llm_response = None; text_sent_to_llm = None; placeholders_used = None; log_orig_excerpt = "Error processing input"; log_red_excerpt = "Error processing input"
        # Audit Logging
        try:
            safe_orig_excerpt = str(log_orig_excerpt or '').replace('|', '/').replace('\n', ' ').replace('\r', '')
            safe_red_excerpt = str(log_red_excerpt or '').replace('|', '/').replace('\n', ' ').replace('\r', '')
            safe_error_msg = str(log_error or '').replace('|', '/').replace('\n', ' ').replace('\r', '')
            log_message = (f"User:{user_identifier} | Model:{log_model} | Findings:{log_num_findings} | OriginalExcerpt:{safe_orig_excerpt} | RedactedExcerpt:{safe_red_excerpt} | Error:{safe_error_msg}"); audit_logger.info(log_message)
        except Exception as log_e: print(f"!!! FAILED TO WRITE AUDIT LOG: {log_e} !!!")

    # Don't pass audit log data to the main index template anymore
    return render_template('index.html',
                           results=results, original_text=original_text_display,
                           uploaded_filename=uploaded_filename, error=error,
                           redacted_text=text_sent_to_llm, llm_response=llm_response,
                           placeholders_used=placeholders_used)


# --- ADMIN ROUTES (NO SECURITY) ---
@app.route('/admin/user_report')
def user_activity_report():
    log_entries, error = read_parse_audit_log(); report_data = {}
    if not error:
        total_requests = len(log_entries); total_findings_overall = 0; model_usage = Counter()
        for entry in log_entries:
            try: total_findings_overall += int(entry.get('findings', 0))
            except ValueError: pass
            model = entry.get('model', 'Unknown')
            if model != 'N/A': model_usage[model] += 1
        report_data = {'total_requests': total_requests, 'total_findings': total_findings_overall, 'model_usage': dict(model_usage.most_common()), 'log_entries': log_entries[:100]}
    return render_template('admin_user_report.html', report_data=report_data, error=error)

@app.route('/admin/findings_graph')
def findings_graph():
    graphJSON = None; error = None
    if not PLOTLY_AVAILABLE: flash("Plotly library not installed. Cannot generate graph.", "warning"); return render_template('admin_findings_graph.html', graphJSON=None, error="Plotly not installed")
    log_entries, error = read_parse_audit_log()
    if not error:
        try:
            findings_summary = Counter({'PII': 0, 'Credentials': 0, 'Contextual': 0, 'Keyword': 0, 'Other': 0})
            for entry in log_entries:
                try:
                    count = int(entry.get('findings', 0))
                    # Crude estimation logic
                    if count > 0: findings_summary['PII'] += int(count * 0.5)
                    if count > 1: findings_summary['Credentials'] += int(count * 0.2)
                    if count > 2: findings_summary['Keyword'] += int(count * 0.2)
                    if count > 3: findings_summary['Contextual'] += int(count * 0.1)
                    if 0 < count <= 3: findings_summary['Other'] += 1
                except ValueError: continue
            findings_summary = Counter({k: v for k, v in findings_summary.items() if v > 0})
            if findings_summary:
                 fig = go.Figure(data=[go.Bar( x=list(findings_summary.keys()), y=list(findings_summary.values()) )])
                 fig.update_layout(title_text='Approximate Finding Category Frequency (Estimate from Logs)')
                 graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            else: graphJSON = None; error = "No finding data to graph."
        except Exception as e: print(f"Error generating graph: {e}"); flash(f"Error generating graph: {e}", "danger"); graphJSON = None; error = str(e)
    return render_template('admin_findings_graph.html', graphJSON=graphJSON, error=error)
# --- End Admin Routes ---


# --- Main Execution ---
if __name__ == '__main__':
    print("-" * 50); print("Starting GenAI Shield Application...")
    print("Ensure required libraries are installed:"); print("  pip install Flask requests PyPDF2 python-docx python-dotenv spacy plotly pandas llama-cpp-python huggingface_hub")
    print(f"Allowed file extensions: {ALLOWED_EXTENSIONS}")
    if not PyPDF2: print("  (PDF processing disabled)")
    if not docx: print("  (DOCX processing disabled)")
    if not nlp: print("  (spaCy NER processing disabled)")
    else: print("  (spaCy NER processing enabled)")
    if not PLOTLY_AVAILABLE: print("  (Graphing disabled)")
    else: print("  (Graphing enabled)")
    # Updated LLM status message
    if LLM_MODE == 'local':
        if llm_local: print(f"  (Local LLM mode enabled - using {LOCAL_MODEL_PATH})")
        else: print(f"  (Local LLM mode selected but FAILED to load model from {LOCAL_MODEL_PATH})")
    elif LLM_MODE == 'api':
        print(f"  (API LLM mode enabled - using {API_URL})")
        if not HF_API_KEY: print("     WARNING: Hugging Face API Key (HF_API_KEY) not found."); print("              API LLM functionality will be disabled.")
        else: print("     Ensure you have accepted terms for the selected model on huggingface.co (if applicable)")
    else: print(f"  (Unknown LLM_MODE: {LLM_MODE})")
    print("Audit logs will be written to 'audit.log'"); print("-" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)

# --- END OF FILE app.py ---