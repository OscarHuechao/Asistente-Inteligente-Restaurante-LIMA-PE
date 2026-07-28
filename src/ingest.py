"""
ingest.py
Lee uno o varios PDFs, los divide en fragmentos (chunks) y crea una base de
datos vectorial local con Chroma, usando embeddings de Google Gemini.

Uso:
    python src/ingest.py --pdf data/mi_documento.pdf
    python src/ingest.py --pdf-dir data/            # procesa todos los PDFs de la carpeta
"""
import argparse
import glob
import os
import re
import shutil

from dotenv import load_dotenv
from PIL import Image
import fitz
import pytesseract
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

load_dotenv()

CHROMA_DIR = "chroma_db"


def pdf_requires_ocr(docs):
    if not docs:
        return True
    text_length = sum(len(doc.page_content.strip()) for doc in docs)
    empty_pages = sum(1 for doc in docs if len(doc.page_content.strip()) < 20)
    return text_length < 100 or empty_pages >= max(1, len(docs) // 2)


def resolve_tesseract_cmd():
    current = pytesseract.pytesseract.tesseract_cmd
    if current and os.path.exists(current):
        return current

    candidate_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for candidate in candidate_paths:
        if os.path.exists(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            return candidate

    return current


def ensure_tessdata_language(lang):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    local_tessdata = os.path.join(base_dir, ".tessdata")
    os.makedirs(local_tessdata, exist_ok=True)

    # Prefer system tessdata if the requested language is already installed there.
    if os.environ.get("TESSDATA_PREFIX"):
        system_tessdata = os.path.join(os.environ["TESSDATA_PREFIX"].rstrip(os.sep))
        if os.path.exists(os.path.join(system_tessdata, f"{lang}.traineddata")):
            return system_tessdata

    candidate_paths = [
        r"C:\Program Files\Tesseract-OCR\tessdata",
        r"C:\Program Files (x86)\Tesseract-OCR\tessdata",
    ]
    for candidate in candidate_paths:
        if os.path.exists(os.path.join(candidate, f"{lang}.traineddata")):
            os.environ["TESSDATA_PREFIX"] = candidate
            return candidate

    local_file = os.path.join(local_tessdata, f"{lang}.traineddata")
    if not os.path.exists(local_file):
        try:
            import urllib.request

            url = f"https://github.com/tesseract-ocr/tessdata/raw/main/{lang}.traineddata"
            urllib.request.urlretrieve(url, local_file)
            print(f"⚙️  Descargado {lang}.traineddata a {local_file}")
        except Exception:
            if os.path.exists(local_file):
                os.remove(local_file)
            raise RuntimeError(
                f"No se pudo descargar {lang}.traineddata. Instala el archivo de idioma "
                "manualmente en tu carpeta de tessdata o usa el lenguaje 'eng'."
            )

    os.environ["TESSDATA_PREFIX"] = local_tessdata
    return local_tessdata


def normalize_ocr_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\s+", " ", text)

    # Correcciones comunes de OCR para texto de menú
    text = re.sub(r"\bAJI?\b(?=\s+de\s+Gallina)", "Ají", text, flags=re.IGNORECASE)
    text = re.sub(r"\bAJI?\b(?=\s+de\s+Habas)", "Ají", text, flags=re.IGNORECASE)
    text = re.sub(r"\bPallo\b", "Pollo", text, flags=re.IGNORECASE)
    text = re.sub(r"\bYegetales\b", "Vegetales", text, flags=re.IGNORECASE)
    text = re.sub(r"\bHuewo\b", "Huevo", text, flags=re.IGNORECASE)
    text = re.sub(r"\bLacteos\b", "Lácteos", text, flags=re.IGNORECASE)
    text = re.sub(r"\bA\s+LACARTA\b", "A LA CARTA", text, flags=re.IGNORECASE)
    return text.strip()


def ocr_pdf(pdf_path):
    resolved = resolve_tesseract_cmd()
    if not resolved or not os.path.exists(resolved):
        raise RuntimeError(
            "Tesseract OCR no se encontró. Instala Tesseract y asegúrate de que "
            "C:\\Program Files\\Tesseract-OCR\\tesseract.exe exista o esté en PATH."
        )

    language = "spa"
    try:
        ensure_tessdata_language(language)
    except RuntimeError as exc:
        print(f"⚠️  {exc}")
        language = "eng"
        ensure_tessdata_language(language)
        print("⚠️  Usando idioma inglés (eng) para OCR en lugar de spa.")

    print("⚙️  El PDF parece escaneado. Usando OCR para extraer texto...")
    documents = []
    with fitz.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img, lang=language, config="--psm 6")
            text = normalize_ocr_text(text)
            documents.append(Document(page_content=text, metadata={"source": pdf_path, "page": page_number}))
    return documents


def ingest_pdf(pdf_path, force_ocr=False):
    """Acepta la ruta a un único PDF (str) o una lista de rutas.

    Si force_ocr es True, se aplica OCR en todos los PDFs aunque extraigan texto.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "Falta GOOGLE_API_KEY. Copia .env.example a .env y coloca tu clave "
            "(consíguela gratis en https://aistudio.google.com/apikey)."
        )

    pdf_paths = [pdf_path] if isinstance(pdf_path, str) else list(pdf_path)

    documents = []
    for path in pdf_paths:
        print(f"📄 Cargando PDF: {path}")
        if force_ocr:
            docs = ocr_pdf(path)
        else:
            loader = PyPDFLoader(path)
            docs = loader.load()
            if pdf_requires_ocr(docs):
                docs = ocr_pdf(path)
        print(f"   -> {len(docs)} páginas cargadas")
        documents.extend(docs)

    print("✂️  Dividiendo el documento en fragmentos...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"   -> {len(chunks)} fragmentos generados")

    print("🧠 Generando embeddings y guardando en Chroma...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # Si ya existe una base previa, la eliminamos para reconstruirla limpia
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"✅ Base vectorial creada en ./{CHROMA_DIR}")
    return vectorstore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingesta de PDF(s) para el agente")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf", help="Ruta a un único archivo PDF a procesar")
    group.add_argument(
        "--pdf-dir", help="Carpeta con uno o más PDFs a procesar (ej. data/)"
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Forzar OCR en todos los PDFs aunque el texto se extraiga correctamente.",
    )
    args = parser.parse_args()

    if args.pdf_dir:
        pdfs = sorted(glob.glob(os.path.join(args.pdf_dir, "*.pdf")))
        if not pdfs:
            raise SystemExit(f"No se encontraron PDFs en {args.pdf_dir}")
        ingest_pdf(pdfs, force_ocr=args.force_ocr)
    else:
        ingest_pdf(args.pdf, force_ocr=args.force_ocr)
