from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from src.ingest import ocr_pdf, ingest_pdf

pdf_path = Path('data/Carta_de_platos_criollos-Restaurante Lima-Pe.pdf')
print('PDF existe:', pdf_path.exists())
print('\n--- Extracción con PyPDFLoader ---')
loader = PyPDFLoader(str(pdf_path))
docs = loader.load()
print('Páginas cargadas (PyPDFLoader):', len(docs))
for d in docs:
    text = d.page_content
    print('--- Página', d.metadata.get('page'), '---')
    print(repr(text[:1000]))
    print()

print('\n--- Extracción con OCR ---')
docs_ocr = ocr_pdf(str(pdf_path))
print('Páginas cargadas (OCR):', len(docs_ocr))
for d in docs_ocr:
    text = d.page_content
    print('--- Página', d.metadata.get('page'), '---')
    print(repr(text[:1000]))
    print()

search_terms = ['ají de gallina', 'aji de gallina', 'ajídegallina', 'ajidegallina', 'aji de gallina?']
for mode, docs in [('PyPDFLoader', docs), ('OCR', docs_ocr)]:
    found = False
    for d in docs:
        lower = d.page_content.lower()
        for term in search_terms:
            if term in lower:
                print(f"Encontrado '{term}' en {mode} página {d.metadata.get('page')}")
                print(repr(lower[lower.find(term):lower.find(term)+200]))
                found = True
    if not found:
        print(f"No aparece ninguna variación de ají de gallina en {mode}")

print('\n--- Reconstruyendo la base vectorial con OCR forzado ---')
ingest_pdf([str(pdf_path)], force_ocr=True)
embeddings = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')
vectorstore = Chroma(persist_directory='chroma_db', embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={'k': 10})
query = '¿Tienen ají de gallina?'
results = retriever.get_relevant_documents(query)
print('Documentos recuperados:', len(results))
for i, d in enumerate(results, start=1):
    print('--- Doc', i, 'page', d.metadata.get('page'), 'source', d.metadata.get('source'))
    print(repr(d.page_content[:500]))
    print()
