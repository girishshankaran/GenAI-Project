import re
import os
import requests
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# Import file reading libraries (ensure installed: pip install PyPDF2 python-docx)
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import docx
except ImportError:
    docx = None

app = Flask(__name__)

# --- Configuration ---
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Text Extraction Helper Functions ---
def extract_text_from_txt(file_stream):
    try:
        return file_stream.read().decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"Error reading TXT file: {e}")
        return None

def extract_text_from_pdf(file_stream):
    if not PyPDF2:
        print("PyPDF2 library not installed. Cannot process PDF.")
        return None
    text = ""
    try:
        reader = PyPDF2.PdfReader(file_stream)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text if text else None
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return None

def extract_text_from_docx(file_stream):
    if not docx:
        print("python-docx library not installed. Cannot process DOCX.")
        return None
    text = ""
    try:
        document = docx.Document(file_stream)
        for para in document.paragraphs:
            text += para.text + "\n"
        return text if text else None
    except Exception as e:
        print(f"Error reading DOCX file: {e}")
        return None

# --- Hugging Face API Setup ---
HF_API_KEY = os.environ.get("HF_API_KEY")
if not HF_API_KEY:
    raise ValueError("HF_API_KEY environment variable is not set. Set it with your Hugging Face API key.")
API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-125m"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# --- LLM Query Function ---
def query_llm(text):
    try:
        payload = {"inputs": text}
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()[0]['generated_text']
    except requests.exceptions.RequestException as e:
        return f"Error querying LLM: {str(e)}"
    except (IndexError, KeyError):
        return "Error: Invalid response from LLM."

# --- Redaction Functions ---
PLACEHOLDER_MAP = {
    "Email Address": "[EMAIL]",
    "Phone Number Format (US)": "[PHONE]",
    "SSN Format (US)": "[SSN]",
    "Potential Credit Card": "[CREDIT_CARD]",
    "API Key Format": "[API_KEY]",
    "Private Key Block": "[PRIVATE_KEY]",
    "Internal/Local IP": "[IP_ADDRESS]",
    "UUID Format": "[UUID]",
}

def get_placeholder(finding):
    if finding['type'] in PLACEHOLDER_MAP:
        return PLACEHOLDER_MAP[finding['type']]
    else:
        return f"[{finding['category'].upper()}]"

def redact_text(text, findings):
    sorted_findings = sorted(findings, key=lambda f: f['span'][0])
    redacted = ""
    last_end = 0
    for finding in sorted_findings:
        start, end = finding['span']
        redacted += text[last_end:start]
        redacted += get_placeholder(finding)
        last_end = end
    redacted += text[last_end:]
    return redacted

# --- Confidentiality Check Function ---
def perform_confidentiality_check(text):
    findings = []
    processed_text_positions = set()

    def add_finding(category, type, value, description, start_index, end_index):
        span = (start_index, end_index)
        if not any(pos in processed_text_positions for pos in range(start_index, end_index)):
            findings.append({
                "category": category,
                "type": type,
                "value": value,
                "description": description,
                "span": span
            })
            for i in range(start_index, end_index):
                processed_text_positions.add(i)

    keyword_categories = {
        "Legal": [
            "attorney-client privilege", "legal hold", "litigation", "settlement agreement",
            "confidential settlement", "under seal", "privileged and confidential",
            "cease and desist", "nda", "non-disclosure agreement", "terms of service", "privacy policy"
        ],
        "Confidential/Secret": [
            "confidential", "proprietary", "secret", "internal use only",
            "trade secret", "classified", "sensitive", "do not distribute",
            "private", "restricted", "not for public release", "password", "secret key",
            "api key", "credential", "token", "access code", "passphrase"
        ],
        "Intellectual Property": [
            "patent pending", "patent application", "trademark", "copyright",
            "invention disclosure", "prototype", "roadmap", "research findings",
            "algorithm", "source code", "proprietary algorithm",
            "Project Phoenix", "Zephyr Initiative"
        ],
        "Personal Data (PII)": [
            "ssn", "social security number", "dob", "date of birth", "passport number",
            "driver's license", "address", "phone number", "email address", "bank account",
            "credit card"
        ]
    }

    lower_text = text.lower() if text else ""
    if lower_text:
        for category, keywords in keyword_categories.items():
            for keyword in keywords:
                try:
                    escaped_keyword = re.escape(keyword)
                    for match in re.finditer(r'\b' + escaped_keyword + r'\b', lower_text, re.IGNORECASE):
                        add_finding(category, "Keyword", match.group(0), f"Detected keyword indicating potential {category.lower()} information.", match.start(), match.end())
                except re.error as e:
                    print(f"Warning: Skipping invalid regex generated from keyword: '{keyword}' - Error: {e}")

    regex_patterns = {
        (r'(password|passwd|secret|token|credential|api[_-]?key)\s*[:=]\s*["\']?([^\s"\' ]{8,})["\']?', "Secret Assignment", "Credentials"):
            "Potential hardcoded secret or credential assignment.",
        (r'\b(sk_live|pk_live|rk_live|sk_test|pk_test|rk_test)_[0-9a-zA-Z]{24,}\b', "API Key Format", "Credentials"):
            "Pattern matches common API key format.",
        (r'-----BEGIN (RSA|OPENSSH|PGP|DSA|EC) PRIVATE KEY-----.*?-----END \1 PRIVATE KEY-----', "Private Key Block", "Credentials", re.DOTALL):
            "Detected block resembling a private key.",
        (r'\b\d{3}-?\d{2}-?\d{4}\b', "SSN Format (US)", "Personal Data (PII)"):
            "Pattern matches US Social Security Number format.",
        (r'\b(?:\d[ -]*?){13,16}\b', "Potential Credit Card", "Personal Data (PII)"):
            "Sequence of 13-16 digits, potentially a Credit Card Number.",
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "Email Address", "Personal Data (PII)"):
            "Detected email address format.",
        (r'\(?\b\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', "Phone Number Format (US)", "Personal Data (PII)"):
            "Pattern matches common US phone number formats.",
        (r'\b(0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])[-/](?:19|20)?\d{2}\b', "Date Format (MM/DD/YY or YYYY)", "Personal Data (PII)"):
            "Detected common date format (e.g., MM/DD/YYYY).",
        (r'\b(192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2[0-9]|3[0-1])\.\d{1,3}\.\d{1,3}|127\.0\.0\.1)\b', "Internal/Local IP", "Network"):
            "Detected potential internal network or localhost IP address.",
        (r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b', "UUID Format", "Identifier"):
            "Detected UUID format, potentially an internal identifier.",
        (r'\b(Project|Codename)[ -_][A-Z][a-zA-Z0-9]+)\b', "Project Codename", "Intellectual Property"):
            "Detected pattern possibly indicating a project codename.",
    }

    if text:
        for pattern_key, description in regex_patterns.items():
            flags = 0
            pattern_type = "Regex Match"
            category = "Pattern"
            if isinstance(pattern_key, tuple):
                pattern = pattern_key[0]
                if len(pattern_key) > 1: pattern_type = pattern_key[1]
                if len(pattern_key) > 2: category = pattern_key[2]
                if len(pattern_key) > 3: flags = pattern_key[3]
            else:
                pattern = pattern_key

            try:
                for match in re.finditer(pattern, text, flags=flags):
                    value = match.group(0) if match.group(0) else ""
                    if value:
                        add_finding(category, pattern_type, value, description, match.start(), match.end())
            except re.error as e:
                print(f"Warning: Skipping invalid regex pattern: {pattern} - Error: {e}")

    return {
        "found_issues": len(findings) > 0,
        "findings": findings
    }

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def home():
    results = None
    original_text = ""
    uploaded_filename = None
    error = None
    llm_response = None

    if request.method == 'POST':
        text_input_from_area = request.form.get('text_to_analyze', '')
        file = request.files.get('file_to_analyze')

        content_to_analyze = None
        source_description = "Unknown"

        if file and file.filename:
            if allowed_file(file.filename):
                try:
                    uploaded_filename = secure_filename(file.filename)
                    print(f"Processing uploaded file: {uploaded_filename}")
                    source_description = f"Uploaded file '{uploaded_filename}'"
                    file_ext = uploaded_filename.rsplit('.', 1)[1].lower()

                    if file_ext == 'txt':
                        content_to_analyze = extract_text_from_txt(file.stream)
                    elif file_ext == 'pdf':
                        content_to_analyze = extract_text_from_pdf(file.stream)
                    elif file_ext == 'docx':
                        content_to_analyze = extract_text_from_docx(file.stream)

                    if content_to_analyze is None:
                        error = f"Failed to extract text from '{uploaded_filename}'."
                except Exception as e:
                    print(f"Error processing file: {e}")
                    error = "An error occurred while processing the uploaded file."
            else:
                error = f"Invalid file type '{file.filename}'. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}."

        elif text_input_from_area:
            print("Processing text from input area.")
            content_to_analyze = text_input_from_area
            original_text = text_input_from_area
            source_description = "Text Input"

        else:
            error = "Please enter text or upload an allowed document (.txt, .pdf, .docx)."

        if content_to_analyze is not None:
            results = perform_confidentiality_check(content_to_analyze)
            if results['found_issues']:
                text_to_send = redact_text(content_to_analyze, results['findings'])
            else:
                text_to_send = content_to_analyze
            llm_response = query_llm(text_to_send)

    return render_template('index.html',
                           results=results,
                           original_text=original_text,
                           uploaded_filename=uploaded_filename,
                           error=error,
                           llm_response=llm_response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)