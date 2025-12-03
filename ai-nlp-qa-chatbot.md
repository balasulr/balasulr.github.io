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

## Code with description

