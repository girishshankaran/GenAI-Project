# GenAI Shield

## Overview

GenAI Shield is a Flask-based web application designed to enhance the safety of interacting with Large Language Models (LLMs). It acts as a pre-processor for user prompts and uploaded documents, identifying and redacting potential confidential or sensitive information before submitting the sanitized content to an LLM for processing. The primary goal is to mitigate the risk of accidentally exposing private data (PII, credentials, secrets) during generative AI interactions.

This application utilizes a hybrid approach for detection, combining:
*   **Regular Expressions (Regex):** For finding specific, well-defined patterns (SSNs, API keys, credit cards, etc.).
*   **Keyword Matching:** For identifying common sensitive terms.
*   **Named Entity Recognition (NER):** Using spaCy's `en_core_web_sm` model to recognize entities like person names, organizations, and locations within their context (with overrides for common misclassifications).

Currently, it's configured to use a **locally run quantized Llama 3 model** (via `llama-cpp-python`) for the LLM interaction step, avoiding reliance on external cloud APIs and associated costs/privacy concerns.

## Features

*   **Flexible Input:** Accepts text input via a textarea or document uploads (`.txt`, `.pdf`, `.docx`).
*   **Multi-Method Detection:** Scans input using Regex, Keywords, and spaCy NER to identify a range of potentially sensitive data types:
    *   Credentials (Passwords, API Keys, Tokens, Private Key Blocks)
    *   Personal Data (PII) (SSN, Credit Cards, Emails, Phone Numbers, Names via NER)
    *   Network Identifiers (Internal IPs, UUIDs)
    *   Contextual Entities (Organizations, Locations, Dates, Monetary Values - currently ignored by default to reduce false positives)
    *   Generic Keywords indicating confidentiality or legal status.
*   **Automated Redaction:** Replaces detected sensitive information with clear placeholders (e.g., `[SSN]`, `[PASSWORD_VALUE]`, `[PERSON_NAME]`).
*   **Local LLM Integration:** Configured to send the redacted (or original) text to a locally hosted GGUF model (tested with Llama 3 3B) using `llama-cpp-python` for text generation/processing.
*   **Web Interface:** Provides a simple UI for input, viewing analysis results (findings list), comparing original vs. redacted text sent to the LLM, and displaying the LLM's response.
*   **UI Placeholders:** Includes non-functional UI elements for selecting different LLM models and configuring admin policies (Alerts, Blocking).
*   **File-Based Audit Logging:** Records details of each analysis request (timestamp, user='anonymous', model used, findings count, excerpts, errors) to an `audit.log` file.
*   **Basic Admin Reports (Unsecured):** Includes separate pages (`/admin/user_report`, `/admin/findings_graph`) accessible via the header menu to display basic statistics and an approximate findings graph derived from the `audit.log` file. **Note:** These pages currently lack any authentication.

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd GenAI-Shield
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install Flask PyPDF2 python-docx python-dotenv spacy plotly pandas llama-cpp-python huggingface_hub requests
    # Note: 'requests' is only needed if you switch LLM_MODE back to 'api'
    ```
    *   **macOS Specific `llama-cpp-python`:** For Metal GPU support on M-series Macs, it's often better to install `llama-cpp-python` separately *after* installing Xcode Command Line Tools (`xcode-select --install`):
        ```bash
        CMAKE_ARGS="-DLLAMA_METAL=on" pip install --force-reinstall --upgrade --no-cache-dir llama-cpp-python
        ```
        If this fails, install without the `CMAKE_ARGS` for CPU-only mode and ensure `n_gpu_layers=0` is set in `app.py`.

4.  **Download spaCy Model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  **Accept Llama 3 License & Login:**
    *   Go to a Llama 3 model page on Hugging Face (e.g., [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)) and accept the license terms.
    *   Log in via your terminal:
        ```bash
        huggingface-cli login
        ```
        (You'll need a Hugging Face access token).

6.  **Download Local LLM Model (GGUF):**
    *   Search the Hugging Face Hub for a quantized GGUF version of the Llama 3 3B Instruct model (e.g., from user `bartowski`).
    *   Download a suitable quantization file (e.g., `...Q4_K_M.gguf`) recommended for your hardware (Q4_K_M is good for ~16GB RAM on CPU/Metal).
    *   Create a `models` subfolder in your project directory: `mkdir models`
    *   Place the downloaded `.gguf` file inside the `models` folder.

7.  **Configure `app.py`:**
    *   Open `app.py`.
    *   Verify/Update the `LOCAL_MODEL_PATH` variable to match the **exact absolute or relative path and filename** of the `.gguf` model you downloaded.
    *   Ensure `LLM_MODE = "local"` is set near the top.
    *   Adjust `n_gpu_layers` in the `Llama(...)` call if needed (0 for CPU, -1 for max Metal offload).

8.  **Create `.env` File (Optional but Recommended):**
    *   Create a file named `.env` in the project root.
    *   Add a secret key for Flask session management (required for flash messages):
        ```dotenv
        SECRET_KEY='a_very_long_random_and_secret_string_here'
        ```
        *(Generate a real secret key for any non-personal use)*
    *   You *could* also add your `HF_API_KEY` here if needed for other purposes or switching back to API mode later.

## Running the Application

1.  Make sure your virtual environment is activated (if used).
2.  Run the Flask development server from the project root directory:
    ```bash
    python3 app.py
    ```
3.  Open your web browser and navigate to `http://127.0.0.1:5000` (or the address shown in the console).

## Configuration

*   **Local LLM Path:** Modify `LOCAL_MODEL_PATH` in `app.py` to point to your downloaded GGUF model file.
*   **LLM Mode:** Change `LLM_MODE` in `app.py` to `"api"` if you want to switch back to using the Hugging Face API (requires setting `API_URL` and ensuring the `HF_API_KEY` env variable is set).
*   **Secret Key:** Set a strong `SECRET_KEY` in a `.env` file or as an environment variable for session security.

## Notes and Limitations

*   **No User Authentication:** The application currently lacks user accounts. All actions are logged as "anonymous". Admin pages are not secured.
*   **Local LLM Performance:** Performance depends heavily on your machine's CPU and RAM. Generating responses can take several seconds or more. `n_gpu_layers` can be adjusted for Metal support on Macs.
*   **Detection Accuracy:** The detection uses a combination of methods and is not foolproof. False positives (flagging non-sensitive data) and false negatives (missing sensitive data) are possible. It should be used as an aid, not a guarantee.
*   **File-Based Logging/Reporting:** Audit logging writes to a simple text file (`audit.log`). The admin reports parse this file, which can be inefficient for large logs and limits reporting capabilities.
*   **Dummy UI Elements:** The "Select Model" and "Policy" configuration features in the UI are currently non-functional placeholders.

## Potential Future Enhancements

*   Implement User Authentication and Roles (e.g., using Flask-Login).
*   Store audit logs and findings details in a database (e.g., SQLite, PostgreSQL) for better querying and reporting.
*   Build out functional admin pages (using Flask-Admin or custom routes) for user management (suspension), policy configuration, and viewing detailed reports/graphs from the database.
*   Implement the policy engine for triggering email alerts or user blocking based on defined thresholds.
*   Integrate email sending capabilities.
*   Fine-tune detection logic (add/refine regex, potentially use larger/fine-tuned NER models).
*   Add options to select different local or cloud LLM models dynamically.
*   Improve UI/UX.
