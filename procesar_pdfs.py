import os
import pdfplumber
import requests
import time
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

CARPETA_PDFS = "pdfs"
TAMANO_CHUNK = 1000
TRASLAPE = 200


def extraer_texto(ruta_pdf):
    texto_completo = ""
    with pdfplumber.open(ruta_pdf) as pdf:
        for pagina in pdf.pages:
            texto_pagina = pagina.extract_text()
            if texto_pagina:
                texto_completo += texto_pagina + "\n"
    return texto_completo


def trocear_texto(texto, tamano=TAMANO_CHUNK, traslape=TRASLAPE):
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fin = inicio + tamano
        chunks.append(texto[inicio:fin])
        inicio += tamano - traslape
    return chunks


def generar_embedding(texto, intentos=5):
    for intento in range(intentos):
        respuesta = requests.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {VOYAGE_API_KEY}"},
            json={"input": texto, "model": "voyage-3"}
        )
        if respuesta.status_code == 200:
            return respuesta.json()["data"][0]["embedding"]
        if respuesta.status_code == 429:
            print(f"Límite alcanzado, esperando 20 segundos... (intento {intento + 1}/{intentos})")
            time.sleep(20)
            continue
        print(f"ERROR de Voyage ({respuesta.status_code}): {respuesta.text}")
        raise Exception("Fallo en generar embedding")
    raise Exception("Se agotaron los intentos por límite de velocidad")


def procesar_pdf(ruta_pdf, nombre_documento):
    print(f"Procesando: {nombre_documento}")
    texto = extraer_texto(ruta_pdf)
    chunks = trocear_texto(texto)

    for i, chunk in enumerate(chunks):
        embedding = generar_embedding(chunk)
        supabase.table("document_chunks").insert({
            "document_name": nombre_documento,
            "content": chunk,
            "embedding": embedding
        }).execute()
        print(f"  Chunk {i + 1}/{len(chunks)} guardado")


def main():
    archivos = [f for f in os.listdir(CARPETA_PDFS) if f.endswith(".pdf")]
    print(f"Encontrados {len(archivos)} PDFs")

    for archivo in archivos:
        ruta_completa = os.path.join(CARPETA_PDFS, archivo)
        procesar_pdf(ruta_completa, archivo)

    print("¡Listo! Todos los PDFs procesados.")


if __name__ == "__main__":
    main()