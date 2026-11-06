import os
from dotenv import load_dotenv
import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# Cargar variables de entorno desde .env
load_dotenv()

# 1. Cargar PDF directamente con pypdf
reader = pypdf.PdfReader("Preguntas_frecuentes__FAQ_.pdf")
docs = [
    Document(
        page_content=page.extract_text() or "",
        metadata={"source": "Preguntas_frecuentes__FAQ_.pdf", "page": i}
    )
    for i, page in enumerate(reader.pages)
]

# 2. Trocear en chunks de ~800 caracteres
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# 3. Embeddings (Ollama nomic-embed-text) + vectorstore persistente (Chroma)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(
    chunks, embeddings,
    persist_directory="./chroma_db")

# 4. LLM con Groq usando la variable de entorno GROQ_API_KEY
llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0)

def rag(pregunta: str) -> str:
    docs_relevantes = vectorstore.similarity_search(pregunta, k=4)
    contexto = "\n\n".join(d.page_content for d in docs_relevantes)
    prompt = f"""Responde la pregunta usando solo el contexto siguiente.
Si la respuesta no está en el contexto, di "No tengo esa información".

CONTEXTO:
{contexto}

PREGUNTA: {pregunta}

RESPUESTA:"""
    return llm.invoke(prompt).content

print(rag("¿Qué es un Sistema Informático de Facturación?"))
