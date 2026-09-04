import os
import requests
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

NUM_CHUNKS_CONTEXTO = 5


def generar_embedding_pregunta(pregunta):
    respuesta = requests.post(
        "https://api.voyageai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {VOYAGE_API_KEY}"},
        json={"input": pregunta, "model": "voyage-3", "input_type": "query"}
    )
    return respuesta.json()["data"][0]["embedding"]


def buscar_chunks_relevantes(embedding_pregunta):
    resultado = supabase.rpc("match_document_chunks", {
        "query_embedding": embedding_pregunta,
        "match_count": NUM_CHUNKS_CONTEXTO
    }).execute()
    return resultado.data


def generar_respuesta(pregunta, chunks):
    contexto = "\n\n---\n\n".join(
        f"[Fuente: {c['document_name']}]\n{c['content']}" for c in chunks
    )

    mensaje = claude.messages.create(
        model="claude-sonnet-5",
        max_tokens=1500,
        messages=[
            {
                "role": "user",
                "content": f"""Eres un asistente que ayuda a responder preguntas y generar casos de prueba basándote ÚNICAMENTE en los siguientes documentos de referencia. Si la información no está en los documentos, dilo claramente.

DOCUMENTOS DE REFERENCIA:
{contexto}

PREGUNTA DEL USUARIO:
{pregunta}

Responde de forma clara y cita de qué documento sacaste cada dato relevante."""
            }
        ],
    )

    for bloque in mensaje.content:
        if bloque.type == "text":
            return bloque.text
    return "No se pudo generar una respuesta."


def main():
    print("Interfaz de pruebas con IA - escribe 'salir' para terminar")
    while True:
        pregunta = input("\nTu pregunta: ")
        if pregunta.lower() == "salir":
            break

        print("Buscando información relevante...")
        embedding = generar_embedding_pregunta(pregunta)
        chunks = buscar_chunks_relevantes(embedding)

        if not chunks:
            print("No encontré información relevante en los documentos.")
            continue

        print(f"Encontrados {len(chunks)} fragmentos relevantes. Generando respuesta...\n")
        respuesta = generar_respuesta(pregunta, chunks)
        print(respuesta)


if __name__ == "__main__":
    main()