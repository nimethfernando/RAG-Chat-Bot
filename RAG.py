!pip install -q langchain langchain-community langchain-huggingface chromadb pypdf sentence-transformers faiss-cpu transformers

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import os

# Load and process documents
loader = PyPDFLoader("/content/Nimeth_Log_report_3.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(texts, embeddings)

# Load local model instead of using API
model_name = "google/flan-t5-small"  # Smaller but works reliably
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.5
)

llm = HuggingFacePipeline(pipeline=pipe)

# Create retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

def ask_question(question):
    try:
        result = qa_chain.invoke({"query": question})
        print("Answer:", result["result"])
        print("\nSources:")
        for doc in result["source_documents"]:
            print(doc.metadata["source"], "- Page", doc.metadata.get("page", "N/A"))
    except Exception as e:
        print(f"Error: {str(e)}")

# Test it
ask_question("What happened on April")