# 🤖 Alura Agente

Interfaz de chat para consultar un menú en PDF del restaurante Lima-Pe.
El proyecto usa OCR + embeddings + LLM para responder solo con información
extraída del documento.

## Qué incluye

- `app.py`: interfaz Streamlit.
- `src/ingest.py`: carga PDFs, aplica OCR si es necesario, divide el texto y crea la base vectorial local.
- `src/agent.py`: busca contexto en Chroma y genera respuestas con Google Gemini.
- `data/`: PDFs fuente.
- `requirements.txt`: dependencias necesarias.
- `.env.example`: plantilla para la API key.

## Requisitos mínimos

- Python 3.11 o superior
- Tesseract OCR instalado en Windows
- Clave de Google Gemini en `GOOGLE_API_KEY`

## Instalación

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Edita `.env` y agrega tu clave:

```text
GOOGLE_API_KEY=tu_clave_aqui
```

## Uso

1. Genera la base vectorial:
   ```powershell
   python src/ingest.py --pdf-dir data/ --force-ocr
   ```

2. Inicia la app:
   ```powershell
   streamlit run app.py
   ```

3. Abre:
   ```text
   http://192.168.8.114:8501
   ```

## Notas

- Usa `--force-ocr` cuando el PDF es escaneado y no contiene texto seleccionable.
- Si `GOOGLE_API_KEY` no está configurada, la app no se inicia.

## Estructura del proyecto

```
app.py
src/
  ingest.py
  agent.py
data/
requirements.txt
.env.example
README.md
```

## Deploy
La Aplicación está desplegada:
URL http://192.168.8.114:8501
Proyecto desarrollado por Oscar Huechao, para el Challenge Final Alura Agente
