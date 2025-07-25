
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os

st.set_page_config(page_title="Multi-Format Content Converter + PDF QA", layout="wide")

st.title("Multi-Format Content Converter and PDF Question Answering")

# Load Mistral model locally
@st.cache_resource
def load_llm():
    return LlamaCpp(
        model_path="/Users/rashmilsinha/models/mistral/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        temperature=0.7,
        max_tokens=256,
        top_p=0.95,
        n_ctx=2048,
        verbose=False,
        stop=["</s>"],
        streaming=False
    )

llm = load_llm()

# Define content transformation prompt templates
TEMPLATES = {
    "blog_to_tweet": """Rewrite the following blog post into a concise 3-tweet thread:

{content}""",
    "academic_to_linkedin": """Summarise the following academic abstract into a friendly LinkedIn post:

{content}""",
    "news_to_email": """Rewrite this news article into a short internal email:

{content}""",
    "informal_to_formal": """Rewrite the following text in a formal, professional tone:

{content}"""
}


# Sidebar: Task selector
task = st.sidebar.selectbox("Choose content transformation task", list(TEMPLATES.keys()))

st.subheader("Content Transformer")
input_text = st.text_area("Enter text for transformation", height=200)
if st.button("Transform Content"):
    if input_text:
        prompt = PromptTemplate.from_template(TEMPLATES[task])
        final_prompt = prompt.format(content=input_text.strip())
        output = llm.invoke(final_prompt)
        st.text_area("Transformed Output", value=output.strip(), height=200)
    else:
        st.warning("Please enter some text to transform.")

# PDF Upload and QA
st.subheader("Ask Questions About a PDF")
uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
query = st.text_input("Ask a question about the PDF:")

if uploaded_pdf and query:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.read())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    result = qa_chain.invoke(query)
    st.markdown("###Answer")
    st.write(result["result"])

    with st.expander("Source"):
        for doc in result["source_documents"]:
            st.markdown(doc.page_content)

