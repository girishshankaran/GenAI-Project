# app.py

import re
import os # Needed for file operations like getting extension
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename # For safe filename handling

# Import file reading libraries (ensure installed: pip install PyPDF2 python-docx)
# Use try-except to handle cases where they might not be installed
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None # Flag that library is missing

try:
    import docx
except ImportError:
    docx = None # Flag that library is missing


app = Flask(__name__)

# --- Configuration ---
# Define allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

def allowed_file(filename):
    """Checks if the filename has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Text Extraction Helper Functions ---
def extract_text_from_txt(file_stream):
    """Reads text from a file stream assumed to be TXT."""
    try:
        # Read bytes and decode, ignoring errors for robustness
        return file_stream.read().decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"Error reading TXT file: {e}")
        return None

def extract_text_from_pdf(file_stream):
    """Extracts text from a PDF file stream using PyPDF2."""
    if not PyPDF2:
        print("PyPDF2 library not installed. Cannot process PDF.")
        return None # Indicate library missing
    text = ""
    try:
        # Create a PdfReader object
        reader = PyPDF2.PdfReader(file_stream)
        # Iterate through each page and extract text
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text: # Add text only if extraction was successful for the page
                text += page_text + "\n"
        return text if text else None # Return None if no text extracted at all
    except PyPDF2.errors.PdfReadError as e:
        # Specific PyPDF2 read error (e.g., encrypted, malformed)
        print(f"PyPDF2 Error reading PDF file: {e}")
        return None # Indicate extraction failure
    except Exception as e:
        # Catch other potential errors during processing
        print(f"General Error reading PDF file with PyPDF2: {e}")
        return None

def extract_text_from_docx(file_stream):
    """Extracts text from a DOCX file stream using python-docx."""
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


# --- Confidentiality Check Function ---
def perform_confidentiality_check(text):
    """
    Analyzes text for potential confidential information across various categories
    using keywords and regex patterns.

    Returns:
        dict: A dictionary containing 'found_issues' (boolean) and 'findings' (list of dicts).
              Each finding dict has 'category', 'type', 'value', 'description', 'span'.
    """
    findings = []
    processed_text_positions = set() # To avoid duplicate reporting on same text span

    # Helper to add findings and track processed text spans
    def add_finding(category, type, value, description, start_index, end_index):
        span = (start_index, end_index)
        # Basic check to prevent adding findings fully contained within an already processed span
        if not any(pos in processed_text_positions for pos in range(start_index, end_index)):
             findings.append({
                 "category": category,
                 "type": type,
                 "value": value,
                 "description": description,
                 "span": span
             })
             # Mark positions as processed
             for i in range(start_index, end_index):
                 processed_text_positions.add(i)

    # 1. Keyword Matching (Case-insensitive)
    # ****** VERIFIED: NO '...' ELLIPSIS OBJECTS IN THESE LISTS ******
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
            "Project Phoenix", "Zephyr Initiative" # Example codenames
        ],
         "Personal Data (PII)": [
             "ssn", "social security number", "dob", "date of birth", "passport number",
             "driver's license", "address", "phone number", "email address", "bank account",
             "credit card"
         ]
    }

    lower_text = text.lower() if text else "" # Ensure lower_text is string even if input is None

    # Perform keyword matching only if text exists
    if lower_text:
        for category, keywords in keyword_categories.items():
            for keyword in keywords: # 'keyword' is guaranteed to be a string here
                try:
                    escaped_keyword = re.escape(keyword)
                    # Use finditer to get match objects with positions
                    for match in re.finditer(r'\b' + escaped_keyword + r'\b', lower_text, re.IGNORECASE):
                        add_finding(category, "Keyword", match.group(0), f"Detected keyword indicating potential {category.lower()} information.", match.start(), match.end())
                except re.error as e:
                    # This error is less likely now but kept for robustness
                    print(f"Warning: Skipping invalid regex generated from keyword: '{keyword}' - Error: {e}")
                except Exception as e:
                    print(f"Warning: Unexpected error during keyword matching for '{keyword}': {e}")


    # 2. Regex Pattern Matching
    # (Keep your existing regex patterns here)
    regex_patterns = {
        # --- Secrets / Credentials ---
        (r'(password|passwd|secret|token|credential|api[_-]?key)\s*[:=]\s*["\']?([^\s"\' ]{8,})["\']?', "Secret Assignment", "Credentials"):
            "Potential hardcoded secret or credential assignment.",
        (r'\b(sk_live|pk_live|rk_live|sk_test|pk_test|rk_test)_[0-9a-zA-Z]{24,}\b', "API Key Format", "Credentials"):
            "Pattern matches common API key format.",
        (r'-----BEGIN (RSA|OPENSSH|PGP|DSA|EC) PRIVATE KEY-----.*?-----END \1 PRIVATE KEY-----', "Private Key Block", "Credentials", re.DOTALL):
            "Detected block resembling a private key.",
        # --- PII (Personal Identifiable Information) ---
        (r'\b\d{3}-?\d{2}-?\d{4}\b', "SSN Format (US)", "PII"): # Made dashes optional
            "Pattern matches US Social Security Number format.",
        (r'\b(?:\d[ -]*?){13,16}\b', "Potential Credit Card", "PII"):
             "Sequence of 13-16 digits, potentially a Credit Card Number.",
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "Email Address", "PII"):
            "Detected email address format.",
        (r'\(?\b\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', "Phone Number Format (US)", "PII"):
             "Pattern matches common US phone number formats.",
        (r'\b(0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])[-/](?:19|20)?\d{2}\b', "Date Format (MM/DD/YY or YYYY)", "PII"): # Optional century
             "Detected common date format (e.g., MM/DD/YYYY).",
         # --- Network / Internal ---
         (r'\b(192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2[0-9]|3[0-1])\.\d{1,3}\.\d{1,3}|127\.0\.0\.1)\b', "Internal/Local IP", "Network"): # Corrected 172 range
             "Detected potential internal network or localhost IP address.",
         # --- Identifiers ---
         (r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b', "UUID Format", "Identifier"):
             "Detected UUID format, potentially an internal identifier.",
         # --- Legal/Project ---
         (r'\b(Project|Codename)[ -_][A-Z][a-zA-Z0-9]+)\b', "Project Codename", "Intellectual Property"):
             "Detected pattern possibly indicating a project codename.",
    }

    # Perform regex matching only if text exists
    if text:
        for pattern_key, description in regex_patterns.items():
            flags = 0
            pattern_type = "Regex Match"
            category = "Pattern" # Defaults
            if isinstance(pattern_key, tuple):
                pattern = pattern_key[0]
                if len(pattern_key) > 1: pattern_type = pattern_key[1]
                if len(pattern_key) > 2: category = pattern_key[2]
                if len(pattern_key) > 3: flags = pattern_key[3]
            else:
                pattern = pattern_key

            try:
                # Use finditer to get match objects with positions
                for match in re.finditer(pattern, text, flags=flags):
                    value = match.group(0) if match.group(0) else ""
                    if value: # Only add if we matched something tangible
                        add_finding(category, pattern_type, value, description, match.start(), match.end())
            except re.error as e:
                print(f"Warning: Skipping invalid regex pattern: {pattern} - Error: {e}")
            except Exception as e:
                print(f"Warning: Error matching regex pattern: {pattern} - Error: {e}")

    return {
        "found_issues": len(findings) > 0,
        "findings": findings
    }
# --- End of perform_confidentiality_check ---


# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def home():
    """Handles homepage display and form submission including file uploads."""
    results = None
    original_text = ""
    uploaded_filename = None
    error = None # Variable to hold error messages for the template

    if request.method == 'POST':
        text_input_from_area = request.form.get('text_to_analyze', '') # Get text from textarea
        file = request.files.get('file_to_analyze') # Get file object

        content_to_analyze = None
        source_description = "Unknown" # Description of where content came from

        # --- Input Priority: File > Text Area ---
        if file and file.filename: # Check if a file was actually selected
            if allowed_file(file.filename):
                try:
                    uploaded_filename = secure_filename(file.filename) # Sanitize filename
                    print(f"Processing uploaded file: {uploaded_filename}")
                    source_description = f"Uploaded file '{uploaded_filename}'"
                    file_ext = uploaded_filename.rsplit('.', 1)[1].lower()

                    # Extract text based on extension
                    if file_ext == 'txt':
                        content_to_analyze = extract_text_from_txt(file.stream)
                        if content_to_analyze is None: error = f"Failed to read text content from TXT file '{uploaded_filename}'."
                    elif file_ext == 'pdf':
                        content_to_analyze = extract_text_from_pdf(file.stream)
                        if content_to_analyze is None:
                            if PyPDF2 is None: error = "PDF processing requires PyPDF2. Install with: pip install PyPDF2"
                            else: error = f"Could not extract text from PDF '{uploaded_filename}'. It might be encrypted, image-based, or corrupted."
                    elif file_ext == 'docx':
                        content_to_analyze = extract_text_from_docx(file.stream)
                        if content_to_analyze is None:
                             if docx is None: error = "DOCX processing requires python-docx. Install with: pip install python-docx"
                             else: error = f"Could not extract text from DOCX '{uploaded_filename}'. The file might be corrupted."

                    if content_to_analyze is None and not error: # Fallback if extraction failed unexpectedly
                        error = f"Failed to extract text content from '{uploaded_filename}'."

                except Exception as e:
                    print(f"An unexpected error occurred during file processing: {e}")
                    error = "An unexpected error occurred while processing the uploaded file."
            else:
                # File was selected but has disallowed extension
                error = f"Invalid file type '{file.filename}'. Allowed types are: {', '.join(ALLOWED_EXTENSIONS)}."

        # If no valid file content obtained, try the text area
        if content_to_analyze is None and not error and text_input_from_area:
            print("Processing text from input area.")
            content_to_analyze = text_input_from_area
            original_text = text_input_from_area # Keep text area content for refill
            source_description = "Text Input"

        # Handle case where neither valid file nor text was provided
        elif content_to_analyze is None and not error:
            # Only show this error if no previous error was set (like invalid extension)
             error = "Please enter text or upload an allowed document (.txt, .pdf, .docx) to analyze."


        # --- Perform analysis if content was successfully obtained ---
        if content_to_analyze is not None:
            print(f"Analyzing content from: {source_description}")
            results = perform_confidentiality_check(content_to_analyze)
            # We don't set original_text here if content came from file, to avoid huge text in textarea
        elif not error: # If content is None but no specific error was set
            error = "Could not retrieve content for analysis." # Should be rare now


    # Render the template, passing all necessary variables
    return render_template('index.html',
                           results=results,
                           original_text=original_text, # For refilling text area
                           uploaded_filename=uploaded_filename, # Filename if upload was source
                           error=error) # Pass error message to template

# --- Run the App ---
if __name__ == '__main__':
    # Set host='0.0.0.0' to make it accessible on your local network (optional)
    app.run(debug=True, host='0.0.0.0', port=5000)