"""
app.py
Interfaz web (Streamlit) para conversar con el agente sobre el PDF cargado.

Ejecutar localmente:
    streamlit run app.py
"""
import os
import sys

import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from agent import ask, build_agent, CHROMA_DIR  # noqa: E402
from ingest import ingest_pdf  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
AGENT_IMAGE = os.path.join(ASSETS_DIR, "ai_agent.svg")

st.set_page_config(
    page_title="Alura Agente",
    page_icon="🤖",
    layout="wide",
)


def get_pdf_list():
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")])


def resolve_api_key():
    if not os.getenv("GOOGLE_API_KEY") and "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    return bool(os.getenv("GOOGLE_API_KEY"))


def main():
    col1, col2 = st.columns([1, 3])

    with col1:
        if os.path.exists(AGENT_IMAGE):
            st.image(AGENT_IMAGE, width=140)
            st.caption("Asistente AI del menú")
        else:
            st.markdown("## 🤖")

    with col2:
        st.markdown(
            "# Bienvenido al restaurante LIMA-PE🍽️\n\n"
            "## Tu asistente de menú inteligente\n\n"
            "📍 Gran Avenida 1994, San Miguel, Santiago de Chile\n\n"
            "📋 Menú y Carta del día\n\n"
            "🥇 Restaurante 5 estrellas ⭐⭐⭐⭐⭐\n\n"
            "Pregunta lo que quieras saber sobre nuestro menú, ofertas y servicios."
        )

    st.markdown("---")
    st.markdown("### Instrucciones rápidas")
    st.markdown(
        "Escribe preguntas como:\n\n"
        "- ¿Cuál es el horario de atención?\n"
        "- ¿Qué opciones vegetarianas ofrece el restaurante?\n"
        "- ¿Tienen ají de gallina?"
    )

    api_key_ok = resolve_api_key()
    pdfs = get_pdf_list()

    if not api_key_ok:
        st.error(
            "Falta la variable de entorno GOOGLE_API_KEY. "
            "Configúrala en tu archivo .env o en los secretos de la plataforma de deploy."
        )
        return

    if not pdfs:
        st.error(
            "No se encontró ningún PDF en la carpeta data/. "
            "Sube un PDF a esa carpeta y vuelve a recargar."
        )
        return

    if not os.path.exists(CHROMA_DIR):
        with st.spinner("Generando la base vectorial a partir de los PDFs..."):
            try:
                ingest_pdf([os.path.join(DATA_DIR, pdf) for pdf in pdfs], force_ocr=True)
            except Exception as exc:
                st.error("No se pudo generar la base vectorial inicial.")
                st.error(str(exc))
                return

    if "chain" not in st.session_state or "retriever" not in st.session_state:
        try:
            chain, retriever = build_agent()
            st.session_state.chain = chain
            st.session_state.retriever = retriever
        except Exception as e:
            st.error("No se pudo iniciar el agente.")
            st.error(str(e))
            return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pregunta = st.chat_input("Escribe tu pregunta sobre el documento...", key="chat_input_main")

    if pregunta:
        st.session_state.messages.append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)

        with st.chat_message("assistant"):
            with st.spinner("Buscando en el documento..."):
                try:
                    respuesta = ask(pregunta, st.session_state.chain, st.session_state.retriever)
                except Exception as e:
                    st.error(
                        "Hubo un error al generar la respuesta. "
                        "Revisa tu clave API, la cuota o el modelo configurado."
                    )
                    st.error(str(e))
                    respuesta = None

                if respuesta is not None:
                    st.markdown(respuesta)
                    st.session_state.messages.append({"role": "assistant", "content": respuesta})


if __name__ == "__main__":
    main()
