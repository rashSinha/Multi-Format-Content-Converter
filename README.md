# Mistral-Powered Local RAG + Content Transformer

A full-featured Streamlit app that integrates a **locally-run Mistral LLM**, **LangChain**, and **multi-format content conversion tools**. It supports:

* Intelligent content rewriting (blog-to-email, blog-to-Twitter, etc.)
* Local file-based RAG for answering questions on PDF documents
* Vision support (image-to-text via local OCR)
* All LLM inference runs fully **offline**

---

## Features

### File-based PDF RAG

* Upload PDFs and query them with Mistral
* Uses FAISS for local vector search

### Content Transformer

* Paste any content (e.g. blog post)
* Select output style (e.g. Email, Summary, Thread)
* Get rephrased result from Mistral locally

### PDF Question Answering (Vision and Text)

* Upload PDF (e.g. text or diagram)
* Ask Mistral about the PDF
* Extracts text via Tesseract OCR first

---

## Folder Structure

```
📦multi_format_content_converter
├── app.py                       # Main Streamlit app
├── models/                      # Local GGUF model directory (download manually)
├── .streamlit/config.toml       # Streamlit configuration
├── requirements.txt             # Python dependencies
├── README.md                    # README
└── .github/workflows/deploy.yml # GitHub Actions workflow for CI
```

---

## Local Setup Instructions

1. **Create virtual environment**

```bash
conda create -n venv python=3.10 -y
conda activate venv
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Download the Mistral GGUF model**

```bash
mkdir -p ~/models/mistral
curl -L https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf -o ~/models/mistral/mistral-7b-instruct-v0.1.Q4_K_M.gguf
```

4. **Run Streamlit app**

```bash
streamlit run app.py
```

---

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Log into [Streamlit Community Cloud](https://streamlit.io/cloud)
3. Click **New App**
4. Connect GitHub repo and select `app.py`
5. Click **Deploy**

Ensure the **GGUF model is not pushed to GitHub** due to file size. You'll need to manually upload it via [Streamlit Secrets Manager](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app#manage-secrets-and-files).

---

## Requirements (see `requirements.txt`)

* `streamlit`
* `llama-cpp-python`
* `langchain`
* `faiss-cpu`
* `tesseract`
* `pytesseract`
* `pdfminer.six`

---

## Model

This app uses **Mistral-7B-Instruct-v0.1 (GGUF Q4\_K\_M)** via `llama-cpp-python`.

You can replace it with another GGUF-compatible model by placing it in `~/models/mistral/` and updating the filename in the app.

---

## License

MIT License. You are free to use, modify, and distribute this project.

---

## Acknowledgements

* [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
* [LangChain](https://python.langchain.com/)
* [Streamlit](https://streamlit.io)
* [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
