---
layout: post
author: Lakshmi Balasubramaniam
tags: [cscc]
---

### [AI for NLP Final Project: Q&A Chatbot]

## Installation & Steps
Before doing below, create folder to place code & make sure that have **python 3.11**, **Anaconda Prompt**, a **code editor** installed

1. Run Anaconda Prompt as administrator and run following commands:
#### 1. Navigate to your desired project directory
```bash
cd path/to/your/project  # replace with your actual folder path
```

#### 2. Create a new conda environment
```bash
conda create -n GithubQAchatbot python=3.11 -y  # replace with name of conda environment want to use if want & can replace python version if want
```

#### 3. Activate the environment
```bash
conda activate GithubQAchatbot  # replace with name of conda environment want to use if want
```

#### 4. Install required packages
```bash
pip install streamlit
```

```bash
pip install transformers torch sentencepiece pypdf beautifulsoup4 python-docx scikit-learn
```

#### 5. Open the project in preferred editor or IDE
- Visual Studio Code > code .
- JupyterLab > jupyter lab
- Jupyter Notebook >jupyter notebook
- PyCharm > charm .

#### 6. Create the main file
Create a file named githubQAchatbot.py (any filename ending with .py works)

#### 7. Add code to githubQAchatbot.py
Paste the provided code below into the file

#### 8. Run the streamlit app
```bash
streamlit run githubQAchatbot.py  # replace with name of file created in step 6
```

## Files that can upload
- PDF, text, html, docx

## Running Project quick steps

#### 1. Navigate to your desired project directory
```bash
cd path/to/your/project  # replace with your actual folder path
```

#### 2. Activate the environment
```bash
conda activate GithubQAchatbot  # replace with your chosen environment name if different
```

#### 3. Run the streamlit app
```bash
streamlit run githubQAchatbot.py  # replace with name of file created in step 6 above
```

## Code in githubQAchatbot.py with description
```python
# --- Import necessary libraries ---
import streamlit as st # Streamlit framework for building the chatbot UI
from transformers import pipeline # Hugging Face Transformers pipeline for question answering
from bs4 import BeautifulSoup # HTML parsing for URL and file content
from pypdf import PdfReader # PDF text extraction
from docx import Document # DOCX file reading
from urllib.parse import urlparse # URL parsing and validation
from sklearn.feature_extraction.text import TfidfVectorizer # Keyword extraction
import requests # HTTP requests for URL fetching
import re # Regular expressions for URL validation
import time # Used for time delay
import os # For terminating the Streamlit process
import random # Used for random follow-up question suggestions

# --- Load the QA model once then cache it for performance ---
# @st.cache_resource ensures the model does not reload every time the application refreshes
@st.cache_resource
def load_qa_model():
    """
    - Load and cache the Hugging Face QA pipeline
    - Uses 'deepset/roberta-base-squad2' for handling unanswerable questions
    - Cached so it only loads once (~30s on first run)
    - Can switch models here if desired
    """
    # Old model used previously:
    # return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    
    # New model with better handling of unanswerable questions
    return pipeline("question-answering", model="deepset/roberta-base-squad2")

# --- Initialize the QA pipeline ---
qa_pipeline = load_qa_model()

# Set up Streamlit page configuration
st.set_page_config(page_title="Q&A Chatbot", page_icon="🤖", layout="centered")

# Title and welcome message
st.title("Final Project - Q&A Chatbot")
st.write("Welcome! Upload a file or enter a URL to begin. Then, ask questions about the content")

# Information box for showing default context is below and usage instructions
st.info("See the full default context below for more details on how the chatbot works.")

# First-time user instructions
st.info("""
### First-Time User Tip

Welcome! Here is how to get started:

1. **Upload a file** *(PDF, TXT, HTML, DOCX)* or **enter a URL**. Only one at a time
2. If no file or URL is provided or extraction fails, the chatbot will use the **default context**, shown below
3. Once your content is loaded, you will see a **preview of the first 1000 characters**
4. Ask a question about the content, the chatbot will extract or generate an answer
5. To end your session, scroll down and check the **End Chat Session** checkbox then click the **Click to confirm** button
""")

# Visual separator
st.markdown(">>>")

# --- Follow-up suggestions to keep the conversation going ---
follow_up_options = [
    "Do you have more questions?",
    "Do you want to ask any related questions?",
    "Do you need clarification or more details?",
    "Want to explore another section?"
]

# --- 2 functions for keyword extraction and show keyword insights ---

# Function to extract keywords using TF-IDF
def extract_keywords(text, top_n=5):
    """
    - Extract the top N keywords from the given text using TF-IDF
    - TF-IDF (Term Frequency–Inverse Document Frequency) highlights words that are important
      in the text but not common across general language
    - Returns a list of keywords sorted by importance
    """
    # Ignore common English stopwords
    vectorizer = TfidfVectorizer(stop_words="english")
    
    # Fit and transform the text
    tfidf_matrix = vectorizer.fit_transform([text])

    # Pair words with scores
    scores = zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0])

    # Sort by score descending
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Return the top N keywords
    return [word for word, score in sorted_scores[:top_n]]

# Function to show keyword insights
def show_keyword_insights(context_text, key_suffix):
    """
    - Display keyword insights in the UI
    - Help users understand the most relevant terms extracted from the context.
    - Adds a checkbox, so the user can choose whether to see keywords
    - Uses extract_keywords() to compute top terms
    - key_suffix ensures unique widget keys when multiple contexts are displayed
    - Handles errors gracefully with st.error()
    """
    st.markdown("### Keyword Insights")
    if st.checkbox("Show top keywords from context", key=f"keywords_{key_suffix}"):
        try:
            keywords = extract_keywords(context_text)
            st.markdown("**Top Keywords from Context:**")

            # Display keywords as a comma-separated string
            st.write(", ".join(keywords))
        except Exception as e:
            st.error(f"Keyword extraction failed: {e}")

# --- Sidebar to handle file upload and URL input ---
with st.sidebar:
    st.subheader("Upload a document or enter a URL")
    st.sidebar.markdown("Please provide a document or webpage for the chatbot to read. Only one input at a time.")
    st.sidebar.markdown( """
    To fully end the chat:
    - Scroll to the bottom of the page
    - Check the **End Chat Session** checkbox
    - Click the **Click to confirm** button
""")
    
    # Visual separator
    st.markdown(">>>")
    st.markdown(">>>")

    # Initialize URL validation flag
    url_validated = False

    # Handle File Upload
    uploaded_file = st.file_uploader(
        label="Upload a PDF, text, HTML, docx file",
        type=["pdf", "txt", "html", "htm", "docx"]
    )

    # Handle URL Input
    url_input = st.text_input("Or enter a webpage URL")
    normalized_url = None

    # URL normalization and validation
    if url_input:
        # Normalize common URL formats (e.g., www.example.com → http://www.example.com)        
        if re.match(r"^www\.[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}$", url_input):
            normalized_url = "http://" + url_input
        elif re.match(r"^[a-zA-Z0-9\-]+(\.[a-zA-Z0-9\-]+)+$", url_input):
            normalized_url = "https://" + url_input
        else:
            normalized_url = url_input

        # Validate URL structure
        parsed = urlparse(normalized_url)

        # Reject invalid domains early
        if parsed.netloc in ["www", ".", ""]:
            st.sidebar.warning("Invalid domain. Please enter a full domain like 'https://example.com'.")
            normalized_url = None
        else:
            # Accept only http/https with valid domain
            if parsed.scheme in ["http", "https"] and parsed.netloc and "." in parsed.netloc:
                try:
                    response = requests.get(normalized_url, timeout=10)
                    url_validated = True
                    # Only show success if no file is uploaded
                    if not uploaded_file:
                        st.sidebar.success("URL fetched successfully!")
                        st.sidebar.markdown(f"**URL:** {normalized_url}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not resolve the domain. Please check if the site exists or try another URL.")
                    normalized_url = None
                except Exception as e:
                    st.error(f"Error fetching URL: {e}")
                    normalized_url = None
            else:
                st.sidebar.warning("Invalid URL format. Please enter a full domain like 'https://example.com'.")

    # Conflict warning and file success
    if uploaded_file and url_validated:
        st.sidebar.warning("Please choose either a file or a URL, not both.")
    elif uploaded_file and not url_validated:
        st.sidebar.success("File loaded successfully!")
        st.sidebar.markdown(f"**File name:** {uploaded_file.name}")
    
    # Sidebar context source summary
    if uploaded_file and not url_validated:
        st.sidebar.markdown("Using uploaded file as context")
    elif url_validated and not uploaded_file:
        st.sidebar.markdown("Using webpage URL as context.")

    # Default context fallback checkmark toggle ensures chatbot always has context, even if extraction returns nothing
    st.sidebar.markdown("Uses default context if file or url extraction fails or no input is provided:")
    use_default_if_empty = st.checkbox("Use default context if file or url extraction fails", value=True)

# --- Initialize context_text ---
context_text = None

# --- Default context (used when no file/URL is provided or extraction fails) ---
default_context = """
The purpose of this chatbot is designed to answer user questions based on uploaded documents or webpage content.

To begin, users can upload a file (PDF, TXT, HTML, or DOCX) or enter a valid URL. Only one input should be
provided at a time. If no input is provided or if the file or URL cannot be processed, the chatbot will use
this default context to demonstrate its functionality.

The chatbot previously used a pre-trained extractive question-answering model (distilbert-base-uncased-distilled-squad),
which sometimes returned unrelated words for short or vague queries. To improve accuracy and relevance, the model was
updated to roberta-base-squad2, a more robust extractive QA model that identifies precise answers from the provided
context.

Once your content is loaded, the chatbot will display a preview of the first 1000 characters. You can then ask questions
about the content, and the chatbot will extract the most relevant answer.

This chatbot is built using Python and Streamlit for the interactive interface. It leverages the Hugging Face Transformers
library to load and run question-answering models and uses supporting libraries such as Requests (for URL handling), Regex
(for input validation) and standard Python utilities for text processing.

Key features include:
- Robust error handling for invalid files or URLs
- Automatic fallback to a default context if extraction fails
- Context preview truncated to the first 1000 characters for clarity
- Session state management to preserve chat history
- Keyword insights to highlight important terms in the context
- Sidebar controls for file upload, URL input, and default context toggling
- Clear instructions for ending the chat session

To end your session, scroll to the bottom of the page, check the End Chat Session checkbox, and click the confirmation button.
"""

# --- 2 Functions for context extraction from uploaded file and URL ---

# Function: Extract text from uploaded file
def extract_context_from_file(uploaded_file):
    """
    - Extract text content from an uploaded file
    - Supported formats: PDF, text, HTML/ HTM, DOCX

    - Parameters: uploaded_file which is a file-like object from the Streamlit file_uploader
    - Returns the extracted text as a string or None if unsupported/error
    """
    # Initialize context_text
    context_text = None

    # Inform user of processing file
    st.info(f"Processing file {uploaded_file.name}")

    try:
        file_type = uploaded_file.type
        if file_type == "application/pdf":
            # Read PDF text page by page
            reader = PdfReader(uploaded_file)
            pages = [page.extract_text() or "" for page in reader.pages]
            context_text = "\n".join(pages)
        elif file_type == "text/plain":
            # Read plain text file
            context_text = uploaded_file.read().decode("utf-8")
        elif file_type in ["text/html", "application/xhtml+xml"] or uploaded_file.name.endswith((".html", ".htm")):
            # Read HTML file and strip tags
            soup = BeautifulSoup(uploaded_file.read(), "html.parser")
            context_text = soup.get_text()
        elif uploaded_file.name.endswith(".docx"):
            # Read DOCX file
            doc = Document(uploaded_file)
            paragraphs = [p.text for p in doc.paragraphs]
            context_text = "\n".join(paragraphs)
        else:
            # Gives user feedback for unsupported formats
            st.warning("Unsupported file type.")        
    except Exception as e:
        # Error handling
        st.error(f"Error reading file: {e}")
        
    return context_text

# Function: Extract text from URL
def extract_context_from_url(url_input):
    """
    - Extract text content from a webpage URL

    - Parameters: url_input which is a validated URL string
    - Returns the extracted text as a string or None if error
    """

    # Initialize context_text
    context_text = None
    
    try:
        response = requests.get(url_input, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        context_text = soup.get_text()
    except Exception as e:
        # Error handling
        st.error(f"Error fetching URL: {e}")
    
    return context_text

# --- Four functions dealing with context and checking for conflict ---
# 1) Function: Show truncated context preview
def show_context_preview(context_text, label="Context Preview", length=500, height=150, key=None):
    """
    - Display a truncated preview of the context text in a text area
    
    - Parameters:
    - context_text (str): The full context text to preview
    - label (str): Label for the text area
    - length (int): Number of characters to show in the preview
    - height (int): Height of the text area
    - key (str): Unique key for the Streamlit widget
    """
    if context_text:
        st.text_area(label, context_text[:length], height=height, key=key)

# 2) Function: Handle empty context with default fallback
def handle_empty_context(context_text, use_default_if_empty):
    """
    - Fallback to default context if extraction fails
    - Returns either the default context or the original context

    - Parameters:
    - context_text: Extracted context text (may be None or empty)
    - use_default_if_empty: Boolean flag to use default context if extraction fails

    """
    if not context_text and use_default_if_empty:
        st.info("Using default context instead.")
        return default_context
    return context_text

# 3) Function: Check for conflict between file and URL input
def check_conflict(uploaded_file, url_input):
    """
    - Warns the user if both file and URL are provided

    - Parameters:
    - uploaded_file: uploaded file object
    - url_input (str): URL string input by the user

    - Returns a boolean indicating if there is a conflict
    """
    if uploaded_file and url_input:
        st.warning("Please choose either a file or a URL, not both")
        return True
    return False

# 4) Function: Display default context
def show_default_context(default_context):
    """
    - Display default context preview and keyword insights

    - Parameters: default_context (str): Default context text
    """
    st.info("Default context is being used.")
    
    # Short preview
    show_context_preview(default_context, "Default Context (Preview)", length=200, height=100, key="default_preview")
    
    # Expandable full context
    with st.expander("View full default context", expanded=False):
        st.text_area("Default Context (Full)", default_context, height=300, key="default_preview_full")
    
    # Keyword insights
    show_keyword_insights(default_context, "default")

# --- Conflict warning and branching ---
# Decide which context source to use → file, URL, or default fallback
# 4 cases shown below

# Conflict case: Checks if both file and URL provided → warn user and skip context extraction
if check_conflict(uploaded_file, url_input):
    context_text = None

# Case 1: File uploaded
elif uploaded_file is not None:
    context_text = extract_context_from_file(uploaded_file)
    context_text = handle_empty_context(context_text, use_default_if_empty)

    # If valid file context extracted → show preview and keyword insights
    if context_text and context_text != default_context:
        show_context_preview(context_text, "Extracted File Text (Preview)", length=1000, height=200, key="file_preview")
        with st.expander("View full extracted file context", expanded=False):
            st.text_area("Extracted File Text (Full)", context_text, height=300, key="file_full")
        show_keyword_insights(context_text, "file")

# Case 2: Valid URL provided
elif normalized_url is not None:
    st.write(f"Using normalized URL: {normalized_url}")
    context_text = extract_context_from_url(normalized_url)

    # Handle empty context from URL with fallback
    context_text = handle_empty_context(context_text, use_default_if_empty)

    # If valid URL context extracted → show preview and keyword insights
    if context_text and context_text != default_context:
        show_context_preview(context_text, "Extracted URL Text (Preview)", length=1000, height=200, key="url_preview")
        with st.expander("View full extracted URL context", expanded=False):
            st.text_area("Extracted URL Text (Full)", context_text, height=300, key="url_full")

        # Show keywords automatically
        show_keyword_insights(context_text, "url")

# Case 3: Invalid URL entered → warn user and fallback to default
elif url_input and normalized_url is None:
    st.warning("URL could not be used due to resolution or format error.")
    context_text = handle_empty_context(None, use_default_if_empty)

# Case 4: No input provided → fallback to default context if enabled
elif not uploaded_file and not url_input:
    context_text = handle_empty_context(None, use_default_if_empty)

# --- Default context display ---
# Show default context info if it is being used
if context_text == default_context:
    show_default_context(default_context)

# Sidebar summary flag
default_context_used = context_text == default_context

# Check for default context used → show sidebar info
if default_context_used:
    st.sidebar.info("Default context is being used.")

# --- Initialize session state for chat history ---
# session_state preserves chat history and context across interactions
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Defensive cleanup ---
# Defensive cleanup → remove leftover widget keys from previous runs
# Prevents duplicate key errors and ensures a clean session state
for key in list(st.session_state.keys()):
    if key.startswith("follow_up_") and not isinstance(st.session_state[key], str):
        del st.session_state[key]

# --- Display chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
if prompt := st.chat_input("Ask me something..."):
    # User submits message → save to session history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # User message saved → display in chat window
    with st.chat_message("user"):
        st.markdown(prompt)

    # Context preview toggle → show preview before answering
    if st.checkbox("Show context preview before answer", value=True, key="preview_toggle"):
        show_context_preview(context_text, "Context Preview", length=500, height=150, key="chat_preview")

    # Choose context source → file, URL, or default fallback
    if context_text:
        context_to_use = context_text
    elif use_default_if_empty:
        context_to_use = default_context
    else:
        # No usable context → stop execution
        st.warning("No context available. Please upload a file or enter a URL.")
        st.stop()

    # --- Block to handle user question and generate answer ---
    # Handle short/vague question → show friendly prompt
    if len(prompt.split()) < 3:
        response = "Your question seems too short or vague. Try asking in more detail for better results."
    # --- QA model answer extraction ---
    else:
        # Runs the extractive QA pipeline with the chosen context
        result = qa_pipeline(question=prompt, context=context_to_use)

        # Extract and clean answer and confidence score
        answer = result.get("answer", "").strip()
        score = result.get("score", 0)

        # Fallback if answer is empty or confidence is low
        # Confidence threshold: if score < 0.2 or answer is empty, show fallback message
        if not answer or score < 0.2:
            response = "I couldn’t find a direct answer in the provided content. You might try rephrasing your question or uploading a more detailed source."
        else:
            response = f"**Answer:** {answer}"
    
    # Assistant response generated → save to session history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Assistant response saved → display reply with follow-up suggestion
    with st.chat_message("assistant"):
        st.markdown(response)
        st.markdown(f"_{random.choice(follow_up_options)}_")

# Visual separator
st.markdown(">>>")
st.markdown(">>>")

# --- End Chat button with confirmation ---
st.subheader("End Chat Session")
col1, col2 = st.columns([2, 1])
with col1:
    confirm = st.checkbox("End chat session")
with col2:
    if confirm and st.button("Click to confirm"):
        # End chat confirmed → clear session state
        st.session_state.messages = []
        st.session_state.context_text = None
        
        # End chat confirmed → inform user of shutdown
        st.success("Chat ended. The server will now shut down.")
        st.markdown("You may see a 'Connection error' popup. This means the session has ended successfully.")
        st.markdown("The server will shut down automatically. You may now close this browser tab.")

        # Shutdown initiated → pause to let user read message
        time.sleep(2)

        # Shutdown complete → terminate Streamlit process
        os._exit(0)

# Visual separator
st.markdown(">>>")
st.markdown(">>>")

# --- Footer → Developer attribution ---
st.markdown("Developed by: Lakshmi Balasubramaniam")
```
