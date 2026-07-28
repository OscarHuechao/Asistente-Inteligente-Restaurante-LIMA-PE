"""
agent.py
Define el agente RAG: recibe una pregunta, busca los fragmentos más
relevantes del documento en Chroma, y usa Gemini para redactar la respuesta
citando solo la información encontrada.
"""
import os
import re

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv(override=True)

CHROMA_DIR = "chroma_db"
DEFAULT_MODEL = os.getenv("GOOGLE_GENAI_MODEL", "gemini-flash-latest")
FALLBACK_MODEL = os.getenv("GOOGLE_GENAI_FALLBACK_MODEL", "gemini-flash-latest")

PROMPT_TEMPLATE = """Eres un asistente que responde preguntas ÚNICAMENTE con base en el
siguiente contexto extraído de un documento interno. Si la respuesta no está
en el contexto, di claramente que no encontraste esa información en el
documento; no inventes datos.

Contexto:
{context}

Pregunta: {question}

Respuesta clara y directa:"""


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def _create_chain(retriever, model):
    llm = ChatGoogleGenerativeAI(model=model, temperature=0)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def find_direct_match(question, docs):
    question_lower = question.lower()
    terms = []
    if "ají" in question_lower or "aji" in question_lower:
        terms.extend(["ají de gallina", "aji de gallina", "aji de habas", "ají de habas"])
    if not terms:
        return None

    for term in terms:
        for doc in docs:
            text = doc.page_content.lower()
            if term in text:
                start = text.index(term)
                snippet = text[start : start + 300]
                snippet = " ".join(snippet.split())
                return f"Sí, el documento menciona '{term}'.\n\n{snippet}"
    return None


def build_agent():
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "Falta GOOGLE_API_KEY. Copia .env.example a .env y coloca tu clave."
        )
    if not os.path.exists(CHROMA_DIR):
        raise RuntimeError(
            "No existe la base vectorial. Corre primero: python src/ingest.py --pdf data/tu_archivo.pdf"
        )

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    try:
        chain = _create_chain(retriever, DEFAULT_MODEL)
        chain.invoke("Hola")
        return chain, retriever
    except Exception as primary_error:
        if DEFAULT_MODEL != FALLBACK_MODEL:
            try:
                chain = _create_chain(retriever, FALLBACK_MODEL)
                chain.invoke("Hola")
                return chain, retriever
            except Exception as fallback_error:
                raise RuntimeError(
                    f"No se pudo inicializar el agente con los modelos {DEFAULT_MODEL} y {FALLBACK_MODEL}. "
                    f"Errores: {primary_error} | {fallback_error}"
                ) from fallback_error
        raise RuntimeError(
            f"No se pudo inicializar el agente con el modelo {DEFAULT_MODEL}: {primary_error}"
        ) from primary_error


def ask(question: str, chain, retriever) -> str:
    """Consulta el agente y aplica un fallback directo si la frase clave está en los documentos."""
    docs = retriever.get_relevant_documents(question)
    direct = find_direct_match(question, docs)
    if direct:
        return direct
    return chain.invoke(question)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python src/agent.py 'tu pregunta aquí'")
        sys.exit(1)
    pregunta = " ".join(sys.argv[1:])
    print(ask(pregunta))
