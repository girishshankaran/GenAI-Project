# --- START OF FILE app.py ---

import re
import os
import requests
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from dotenv import load_dotenv # Import load_dotenv

load_dotenv() # Load variables from .env file into environment (if using .env method)

# Import file reading libraries (ensure installed: pip install PyPDF2 python-docx requests Flask python-dotenv)
try:
    import PyPDF2
except ImportError:
    print("WARNING: PyPDF2 library not installed. PDF processing will be disabled.")
    print("Install using: pip install PyPDF2")
    PyPDF2 = None

try:
    import docx
except ImportError:
    print("WARNING: python-docx library not installed. DOCX processing will be disabled.")
    print("Install using: pip install python-docx")
    docx = None

app = Flask(__name__)

# --- Configuration ---
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

def allowed_file(filename):
    """Checks if the filename has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Text Extraction Helper Functions ---
def extract_text_from_txt(file_stream):
    """Extracts text from a TXT file stream."""
    try:
        file_stream.seek(0) # Reset stream position
        return file_stream.read().decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"Error reading TXT file: {e}")
        return None

def extract_text_from_pdf(file_stream):
    """Extracts text from a PDF file stream using PyPDF2."""
    if not PyPDF2:
        print("PyPDF2 library not installed. Cannot process PDF.")
        return None
    text = ""
    try:
        file_stream.seek(0) # Reset stream position
        reader = PyPDF2.PdfReader(file_stream)
        if reader.is_encrypted:
            print("Warning: PDF is encrypted and cannot be read without a password.")
            return "[ERROR: PDF is encrypted]"

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text if text else None # Return None if no text extracted
    except PyPDF2.errors.PdfReadError as e:
        print(f"Error reading PDF file (PdfReadError): {e}. File might be corrupted or incompatible.")
        return None
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return None

def extract_text_from_docx(file_stream):
    """Extracts text from a DOCX file stream using python-docx."""
    if not docx:
        print("python-docx library not installed. Cannot process DOCX.")
        return None
    text = ""
    try:
        file_stream.seek(0) # Reset stream position
        document = docx.Document(file_stream)
        for para in document.paragraphs:
            text += para.text + "\n"
        return text if text else None # Return None if no text extracted
    except Exception as e:
        print(f"Error reading DOCX file: {e}")
        return None

# --- Hugging Face API Setup ---
HF_API_KEY = os.environ.get("HF_API_KEY")

if not HF_API_KEY:
    print("WARNING: HF_API_KEY environment variable not set. LLM functionality will be disabled.")

# *** UPDATED MODEL URL ***
# Switch to a model suitable for the free Inference API tier
# Note: Ensure you have accepted terms for this model on Hugging Face website!
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
# API_URL = "https://api-inference.huggingface.co/models/google/gemma-7b-it" # Gemma is too large for free tier

headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# --- LLM Query Function ---
def query_llm(text):
    """Queries the Hugging Face Inference API with the provided text."""
    if not HF_API_KEY or not API_URL:
        return "[LLM Query Skipped - API Key or URL not configured]"
    if not text or not isinstance(text, str) or not text.strip():
        return "[LLM Query Skipped - Input text is empty or invalid]"

    # Adjust prompt slightly for Mistral Instruct format if needed, though generic often works
    prompt = f"Process the following text and provide a relevant response:\n\n---\n{text}\n---\n\nResponse:"
    # Mistral official format uses [INST] [/INST] tags, but often works without for simple tasks.
    # prompt = f"[INST] Process the following text and provide a relevant response:\n\n{text} [/INST]"

    try:
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 250, # Adjust as needed
                "return_full_text": False,
                "temperature": 0.7,
                "top_p": 0.9,
            }
            # Mistral might not need do_sample explicitly if temp/top_p are set
        }
        print(f"Sending request to LLM: {API_URL}")
        response = requests.post(API_URL, headers=headers, json=payload, timeout=90) # Increased timeout slightly
        response.raise_for_status()

        result = response.json()
        print(f"LLM Raw Response: {result}")

        generated_text = None
        if isinstance(result, list) and result:
            if isinstance(result[0], dict) and 'generated_text' in result[0]:
                generated_text = result[0]['generated_text']
            elif isinstance(result[0], str):
                 generated_text = result[0] # Some simpler models might return string directly
        elif isinstance(result, dict) and 'generated_text' in result:
             generated_text = result.get('generated_text') # Check top-level dict too

        if generated_text:
             # Sometimes models might include the prompt in the response even with return_full_text=False
             # Basic check to remove prompt if it appears at the start (adapt if needed)
            if generated_text.strip().startswith(prompt.strip()):
                 generated_text = generated_text[len(prompt):].strip()
            elif "Response:" in generated_text: # Find the response marker
                 generated_text = generated_text.split("Response:", 1)[-1].strip()

            return generated_text.strip()
        else:
            print(f"Could not extract 'generated_text' from LLM response: {result}")
            # Check for specific error messages from HF API within the response
            if isinstance(result, dict) and 'error' in result:
                 api_error = result['error']
                 print(f"Hugging Face API Error Message: {api_error}")
                 # Check if it's the 'model is loading' error
                 if isinstance(api_error, str) and "currently loading" in api_error.lower():
                      estimated_time = result.get('estimated_time', 'unknown')
                      return f"[LLM Info: Model is currently loading, please wait ({estimated_time:.1f}s estimated) and try again.]"
                 else:
                     return f"[Error: Received error from API: {api_error}]" # Return the specific API error
            return "[Error: LLM response format unexpected or empty]"

    except requests.exceptions.Timeout:
        print(f"Error querying LLM: Request timed out.")
        return "[Error: LLM request timed out]"
    except requests.exceptions.RequestException as e:
        error_detail = str(e); status_code = "N/A"; resp_text = "N/A"
        if e.response is not None:
            status_code = e.response.status_code; resp_text = e.response.text[:500] # Get first 500 chars of response
        print(f"Error querying LLM ({API_URL}): {error_detail} | Status Code: {status_code} | Response: {resp_text}")
        if status_code == 401: return "[Error querying LLM: Authorization failed (401). Check your HF_API_KEY.]"
        elif status_code == 403: return f"[Error querying LLM: Forbidden (403). Ensure you accepted the terms for model {API_URL.split('/')[-1]} on Hugging Face.]"
        elif status_code == 429: return "[Error querying LLM: Rate limit possibly exceeded (429).]"
        # Handle 503 Service Unavailable (often means model loading)
        elif status_code == 503: return "[Error querying LLM: Service Unavailable (503). Model might be loading or temporarily down. Try again later.]"
        elif status_code >= 500: return f"[Error querying LLM: Server error ({status_code}). Try again later.]"
        else: return f"[Error querying LLM: {str(e)}]"
    except (IndexError, KeyError, TypeError) as e:
        # Include result in the print for better debugging
        print(f"Error parsing LLM response: {e}. Response was: {result if 'result' in locals() else 'Not available'}")
        return "[Error: Invalid response structure from LLM]"
    except Exception as e:
        print(f"An unexpected error occurred during LLM query: {e}")
        return f"[An unexpected error occurred during LLM query: {str(e)}]"


# --- Redaction Functions ---
PLACEHOLDER_MAP = {
    "Email Address": "[EMAIL]", "Phone Number Format (US)": "[PHONE]",
    "SSN Format (US)": "[SSN]", "Potential SSN (Loose Format)": "[SSN_LIKE]",
    "Potential Credit Card": "[CREDIT_CARD]", "API Key Format": "[API_KEY]",
    "Private Key Block": "[PRIVATE_KEY]", "Internal/Local IP": "[IP_ADDRESS]",
    "UUID Format": "[UUID]", "Secret Assignment": "[SECRET_VALUE]",
    "Password Assignment": "[PASSWORD_VALUE]", "Credential Assignment": "[CREDENTIAL_VALUE]",
    "Username Assignment": "[USERNAME_VALUE]", "Bank Account Number": "[BANK_ACCOUNT]",
    "Date Format (MM/DD/YY or YYYY)": "[DATE]", "Project Codename": "[PROJECT_CODENAME]",
    "Potential Generic Key/Token": "[GENERIC_KEY]", "JSON Key Pair": "[JSON_SECRET_PAIR]",
    "Credentials": "[CREDENTIAL]", "Personal Data (PII)": "[PII]",
    "Confidential/Secret": "[CONFIDENTIAL_INFO]", "Legal": "[LEGAL_TERM]",
    "Intellectual Property": "[IP_INFO]", "Network": "[NETWORK_INFO]",
    "Identifier": "[IDENTIFIER]", "Keyword": "[KEYWORD]" # Added generic keyword placeholder
}

def get_placeholder(finding):
    """Gets the appropriate placeholder for a finding."""
    ftype = finding.get('type', ''); fcategory = finding.get('category', '')
    if ftype in PLACEHOLDER_MAP: return PLACEHOLDER_MAP[ftype]
    elif fcategory in PLACEHOLDER_MAP: return PLACEHOLDER_MAP[fcategory]
    else:
        sanitized_category = re.sub(r'[^\w\s-]', '', fcategory).strip().upper().replace(' ','_')
        return f"[{sanitized_category or 'INFO'}]"

def redact_text(text, findings):
    """Redacts the text based on the list of findings, handling overlaps."""
    if not findings or not text: return text
    sorted_findings = sorted(findings, key=lambda f: (f['span'][0], -f['span'][1]))
    redacted = ""
    last_end = 0
    processed_indices = set()
    for finding in sorted_findings:
        start, end = finding['span']
        if not (0 <= start < end <= len(text)): continue
        is_fully_covered = all(i in processed_indices for i in range(start, end))
        if is_fully_covered: continue
        overlaps = any(i in processed_indices for i in range(start, end))
        if overlaps: continue
        if start >= last_end: redacted += text[last_end:start]
        else: print(f"Warning: Overlap detected - Adjusting start. Finding: {finding}")
        placeholder = get_placeholder(finding)
        redacted += placeholder
        last_end = end
        for i in range(start, end): processed_indices.add(i)
    if last_end < len(text): redacted += text[last_end:]
    return redacted

# --- Confidentiality Check Function ---
def perform_confidentiality_check(text):
    """Performs confidentiality checks using keywords and regex patterns."""
    if not text or not isinstance(text, str): return {"found_issues": False, "findings": []}
    findings = []; processed_text_positions = set()
    def add_finding(category, type, value, description, start_index, end_index):
        # Check span validity relative to text length *before* proceeding
        if not (0 <= start_index < end_index <= len(text)):
             print(f"Warning: Invalid span ({start_index},{end_index}) for value '{value[:50]}...' - Text length {len(text)}. Skipping.")
             return
        span = (start_index, end_index)
        # Check for overlap with already added findings
        if any(i in processed_text_positions for i in range(start_index, end_index)):
             # Optional: Add logging here if you want to see skipped overlaps
             # print(f"Debug: Skipping overlapping finding {type} at {span}")
             return
        findings.append({"category": category, "type": type, "value": value, "description": description, "span": span})
        for i in range(start_index, end_index): processed_text_positions.add(i)

    keyword_categories = {
        "Legal": [r"attorney-client\s+privilege", r"legal\s+hold", r"litigation", r"confidential\s+settlement", r"under\s+seal", r"privileged\s+and\s+confidential", r"cease\s+and\s+desist", r"nda", r"non-disclosure\s+agreement"],
        "Confidential/Secret": [r"confidential", r"proprietary", r"secret", r"internal\s+use\s+only", r"trade\s+secret", r"classified", r"sensitive", r"do\s+not\s+distribute", r"private", r"restricted", r"not\s+for\s+public\s+release",
                              r"password", r"secret key", r"api key", r"credential", r"token"],
        "Intellectual Property": [r"patent\s+pending", r"patent\s+application", r"trademark", r"copyright", r"invention\s+disclosure", r"prototype", r"roadmap", r"research\s+findings", r"algorithm", r"proprietary\s+algorithm"],
        "Personal Data (PII)": [r"dob", r"date\s+of\s+birth", r"passport\s+number", r"driver'?s\s+license", r"address",
                                r"ssn", r"social security number", r"credit card", r"bank account", r"email address", r"phone number"],
    }
    original_text = text; lower_text = original_text.lower()
    temp_keyword_findings = []; keyword_processed_positions_check = set()
    if lower_text: # Find potential keyword positions first
        for category, keywords in keyword_categories.items():
            for keyword_pattern in keywords:
                try:
                    for match in re.finditer(r'\b' + keyword_pattern + r'\b', lower_text, re.IGNORECASE):
                        start, end = match.start(), match.end()
                        # Check for self-overlap during this phase
                        if not any(pos in keyword_processed_positions_check for pos in range(start, end)):
                            # Store finding details with original casing value
                            temp_keyword_findings.append({"category": category, "type": "Keyword", "value": original_text[start:end], "description": f"Detected keyword: '{original_text[start:end]}'.", "span": (start, end)})
                            for i in range(start, end): keyword_processed_positions_check.add(i)
                except re.error as e: print(f"Warning: Skipping invalid regex keyword: '{keyword_pattern}' - {e}")

    # *** REGEX LIST including the corrected Private Key pattern ***
    regex_patterns = [
        # Credentials / Secrets
        (r'\b(password|passwd|secret|pwd|pass)\b(?:\s+(?:is|was|are|be)\s+|\s*[:=]\s*|\s+)(\S+)', "Password Assignment", "Credentials", "Potential password assignment.", re.IGNORECASE),
        (r'\b(api_?key|access_?key|secret_?key|token|credential|auth_token)\s*[:=]\s*(["\']?\S+["\']?)', "Credential Assignment", "Credentials", "Potential API key/token/secret assignment.", re.IGNORECASE),
        (r'\b(user(?: |_)name|user|login|user_?id)\s*[:=]\s*\S+', "Username Assignment", "Credentials", "Potential username or login ID assignment.", re.IGNORECASE),
        (r'\b(sk_live|pk_live|rk_live|sk_test|pk_test|rk_test)_[0-9a-zA-Z]{24,}\b', "API Key Format", "Credentials", "Common API key format (e.g., Stripe)."),
        # Corrected: Use capturing group for \1 backreference
        (r'-----BEGIN (RSA|OPENSSH|PGP|DSA|EC) PRIVATE KEY-----.*?-----END \1 PRIVATE KEY-----', "Private Key Block", "Credentials", "Private key block detected.", re.DOTALL | re.IGNORECASE),
        (r'\b[a-zA-Z0-9\+\/]{40,}\b', "Potential Generic Key/Token", "Credentials", "High entropy string (potential key/token)."),
        (r'"(?:access_key|secret_key|api_key|token)"\s*:\s*"([^"]+)"', "JSON Key Pair", "Credentials", "Potential sensitive keys in JSON.", re.IGNORECASE),

        # Personal Data (PII)
        (r'\b(ssn|social security(?: number)?)\s*(?:is|:|=)?\s*(\d{3}-?\d{2}-?\d{4})\b', "SSN Format (US)", "Personal Data (PII)", "US SSN pattern with indicator.", re.IGNORECASE),
        (r'(?<!\d-)\b(\d{3}-\d{2}-\d{4})\b(?!\d)', "SSN Format (US)", "Personal Data (PII)", "US SSN pattern (standalone)."),
        (r'(?<!\d-)\b(\d{7}|\d{9})\b(?!\d)', "Potential SSN (Loose Format)", "Personal Data (PII)", "7 or 9 digits (standalone, potential SSN)."),
        (r'\b(credit card|cc|card number|cc#|card no)\s*(?:is|:|=)?\s*(\d[\d -]{11,18}\d)\b', "Potential Credit Card", "Personal Data (PII)", "Potential Credit Card Number with indicator.", re.IGNORECASE),
        (r'\b(\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4})\b', "Potential Credit Card", "Personal Data (PII)", "16 digits in groups of 4."),
        (r'\b(bank account(?: number| no)?|account no|account number|acct no)\s*(?:is|:|=|,?\s+which\s+is)?\s*(\d[\d\s-]{5,}\d?)', "Bank Account Number", "Personal Data (PII)", "Potential bank account number.", re.IGNORECASE),
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "Email Address", "Personal Data (PII)", "Email address format."),
        (r'\(?\b\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', "Phone Number Format (US)", "Personal Data (PII)", "US phone number format."),
        (r'\b(0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])[-/](?:19|20)?\d{2}\b', "Date Format (MM/DD/YY or YYYY)", "Personal Data (PII)", "Date format (MM/DD/YYYY)."),

        # Network / Identifiers
        (r'\b(192\.168(?:\.\d{1,3}){2}|10(?:\.\d{1,3}){3}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}|127(?:\.\d{1,3}){3})\b', "Internal/Local IP", "Network", "Internal/localhost IP address."),
        (r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b', "UUID Format", "Identifier", "UUID format."),

        # Project / Internal Naming
        (r'\b(?:Project|Codename|Initiative)[ -_]([A-Z][a-zA-Z0-9]+(?:[ -_][A-Z][a-zA-Z0-9]+)*)\b', "Project Codename", "Intellectual Property", "Potential project codename."),
    ]
    # *** END OF REGEX LIST ***

    if original_text: # Process Regex patterns first - they are generally more specific
        for item in regex_patterns:
            pattern, type, category, description = item[0], item[1], item[2], item[3]
            flags = item[4] if len(item) > 4 else 0
            try:
                for match in re.finditer(pattern, original_text, flags=flags):
                    # Extract value and span from the match object
                    value = match.group(0) # Get the entire matched string
                    start, end = match.start(), match.end()
                    # Add finding if value is not empty and span is valid
                    if value and start < end:
                        add_finding(category, type, value, description, start, end)
            except re.error as e: print(f"Warning: Skipping invalid regex pattern: {pattern} - {e}")
            except IndexError as e: print(f"Warning: Index error during regex {pattern} - {e}") # Catch potential group index errors
    # Now add the keyword findings *only if* their positions haven't already been covered by a regex finding
    for kw_finding in temp_keyword_findings:
         # Use the add_finding function which handles overlap checks
         add_finding(kw_finding["category"], kw_finding["type"], kw_finding["value"], kw_finding["description"], kw_finding["span"][0], kw_finding["span"][1])
    # Final sort before returning (helps redaction function, though add_finding prevents most overlaps)
    sorted_findings = sorted(findings, key=lambda f: (f['span'][0], -f['span'][1]))
    return {"found_issues": len(sorted_findings) > 0, "findings": sorted_findings}


# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def home():
    """Handles GET and POST requests for the main page."""
    results = None; original_text_display = ""; uploaded_filename = None
    error = None; llm_response = None; text_sent_to_llm = None
    source_description = "No input processed yet."

    if request.method == 'POST':
        text_input_from_area = request.form.get('text_to_analyze', '')
        file = request.files.get('file_to_analyze')
        content_to_analyze = None

        if file and file.filename:
            if allowed_file(file.filename):
                uploaded_filename = secure_filename(file.filename)
                print(f"Processing uploaded file: {uploaded_filename}")
                source_description = f"Uploaded file '{uploaded_filename}'"
                file_ext = uploaded_filename.rsplit('.', 1)[1].lower()
                try:
                    if file_ext == 'txt': content_to_analyze = extract_text_from_txt(file.stream)
                    elif file_ext == 'pdf': content_to_analyze = extract_text_from_pdf(file.stream)
                    elif file_ext == 'docx': content_to_analyze = extract_text_from_docx(file.stream)
                    if content_to_analyze is None or content_to_analyze == "[ERROR: PDF is encrypted]":
                        error_reason = "Check file content, corruption, or required libraries."
                        if content_to_analyze == "[ERROR: PDF is encrypted]": error_reason = "PDF is password-protected."
                        error = f"Failed to extract text from '{uploaded_filename}'. {error_reason}"
                        original_text_display = text_input_from_area
                        content_to_analyze = None
                    else: original_text_display = ""
                except Exception as e:
                    print(f"Error processing file {uploaded_filename}: {e}")
                    error = f"An error occurred processing file '{uploaded_filename}'."
                    original_text_display = text_input_from_area
            else:
                error = f"Invalid file type '{file.filename}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}."
                original_text_display = text_input_from_area
        elif text_input_from_area:
            print("Processing text from input area.")
            content_to_analyze = text_input_from_area
            original_text_display = text_input_from_area
            source_description = "Text Input"
        else: error = "Please enter text or upload an allowed document (.txt, .pdf, .docx)."

        if content_to_analyze is not None and error is None:
            if not isinstance(content_to_analyze, str): content_to_analyze = str(content_to_analyze)
            print(f"Performing confidentiality check on content from {source_description}...")
            results = perform_confidentiality_check(content_to_analyze)
            print(f"Check complete. Found issues: {results['found_issues']}")
            if results['found_issues']:
                print("Redacting text before sending to LLM...")
                text_sent_to_llm = redact_text(content_to_analyze, results['findings'])
            else:
                print("No issues found needing redaction, sending original text.")
                text_sent_to_llm = content_to_analyze
            llm_response = query_llm(text_sent_to_llm)
            print("LLM query finished.")
        elif error: results = None; llm_response = None; text_sent_to_llm = None

    return render_template('index.html',
                           results=results,
                           original_text=original_text_display,
                           uploaded_filename=uploaded_filename,
                           error=error,
                           redacted_text=text_sent_to_llm,
                           llm_response=llm_response)

# --- Main Execution ---
if __name__ == '__main__':
    print("-" * 50)
    print("Starting GenAI Shield Application...")
    print("Ensure required libraries are installed:")
    print("  pip install Flask requests PyPDF2 python-docx python-dotenv")
    print(f"Allowed file extensions: {ALLOWED_EXTENSIONS}")
    if not PyPDF2: print("  (PDF processing disabled)")
    if not docx: print("  (DOCX processing disabled)")
    print("-" * 50)
    if HF_API_KEY:
        print(f"Using Hugging Face API Key: YES")
        print(f"Using LLM Endpoint: {API_URL}")
        print("Ensure you have accepted terms for the selected model on huggingface.co") # Reminder
    else:
        print("WARNING: Hugging Face API Key (HF_API_KEY) not found.")
        print("         LLM functionality will be disabled.")
    print("-" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)

# --- END OF FILE app.py ---