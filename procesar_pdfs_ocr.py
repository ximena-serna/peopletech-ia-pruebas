import os
import io
import base64
import requests
from pdf2image import convert_from_path
from supabase import create_client
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
claude = Anthropic(api_key=ANTHROPIC_API_KEY)

CARPETA_PDFS = "pdfs"
TAMANO_CHUNK = 1000
TRASLAPE = 200

# Lista de PDFs que quedaron en 0 chunks (los que necesitan OCR completo)
PDFS_OCR = [
    "Anexo 20 RMF.pdf",
]


def imagen_a_base64(imagen_pil):
    buffer = io.BytesIO()
    imagen_pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def extraer_texto_con_claude(imagen_pil, numero_pagina):
    img_b64 = imagen_a_base64(imagen_pil)
    mensaje = claude.messages.create(
        model="claude-sonnet-5",
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Transcribe todo el texto visible en esta imagen de un documento. Incluye tablas, encabezados y cualquier dato relevante. Responde SOLO con el texto transcrito, sin comentarios adicionales.",
                    },
                ],
            }
        ],
    )
    for bloque in mensaje.content:
        if bloque.type == "text":
            return bloque.text
    raise Exception("No se encontró texto en la respuesta de Claude")


def trocear_texto(texto, tamano=TAMANO_CHUNK, traslape=TRASLAPE):
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fin = inicio + tamano
        chunks.append(texto[inicio:fin])
        inicio += tamano - traslape
    return chunks


def generar_embedding(texto, intentos=5):
    import time
    for intento in range(intentos):
        respuesta = requests.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {VOYAGE_API_KEY}"},
            json={"input": texto, "model": "voyage-3"}
        )
        if respuesta.status_code == 200:
            return respuesta.json()["data"][0]["embedding"]
        if respuesta.status_code == 429:
            print(f"Límite alcanzado, esperando 20 segundos...")
            time.sleep(20)
            continue
        print(f"ERROR de Voyage ({respuesta.status_code}): {respuesta.text}")
        raise Exception("Fallo en generar embedding")
    raise Exception("Se agotaron los intentos")


def procesar_pdf_con_ocr(nombre_pdf):
    ruta_pdf = os.path.join(CARPETA_PDFS, nombre_pdf)
    print(f"Convirtiendo a imágenes: {nombre_pdf}")
    paginas = convert_from_path(ruta_pdf, dpi=150)
    print(f"  {len(paginas)} páginas encontradas")

    texto_completo = ""
    for i, pagina in enumerate(paginas):
        print(f"  Procesando página {i + 1}/{len(paginas)} con Claude...")
        texto_pagina = extraer_texto_con_claude(pagina, i + 1)
        texto_completo += texto_pagina + "\n"

    chunks = trocear_texto(texto_completo)
    for i, chunk in enumerate(chunks):
        embedding = generar_embedding(chunk)
        supabase.table("document_chunks").insert({
            "document_name": nombre_pdf,
            "content": chunk,
            "embedding": embedding
        }).execute()
        print(f"  Chunk {i + 1}/{len(chunks)} guardado")


def main():
    for nombre_pdf in PDFS_OCR:
        procesar_pdf_con_ocr(nombre_pdf)
    print("¡Listo! PDFs con OCR procesados.")


if __name__ == "__main__":
    main()