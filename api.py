import os
import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

NUM_CHUNKS_CONTEXTO = 20

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PreguntaRequest(BaseModel):
    pregunta: str


def generar_embedding_pregunta(pregunta):
    respuesta = requests.post(
        "https://api.voyageai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {VOYAGE_API_KEY}"},
        json={"input": pregunta, "model": "voyage-3", "input_type": "query"}
    )
    return respuesta.json()["data"][0]["embedding"]


def buscar_chunks_relevantes(embedding_pregunta, texto_pregunta):
    resultado = supabase.rpc("match_document_chunks_hybrid", {
        "query_embedding": embedding_pregunta,
        "query_text": texto_pregunta,
        "match_count": NUM_CHUNKS_CONTEXTO
    }).execute()
    return resultado.data


def generar_respuesta(pregunta, chunks):
    contexto = "\n\n---\n\n".join(
        f"[Fuente: {c['document_name']}]\n{c['content']}" for c in chunks
    )

    mensaje = claude.messages.create(
        model="claude-sonnet-5",
        max_tokens=3000,
        messages=[
            {
                "role": "user",
                "content": f"""Eres un asistente de QA que ayuda a un equipo a probar el sistema de nómina que PeopleTech implementó para su empresa. Tu trabajo es ayudar a validar que el sistema funcione correctamente según la normativa mexicana aplicable (LFT, LSS, ISR, salarios mínimos, etc.) y según los manuales funcionales del sistema.

Usa ÚNICAMENTE los siguientes documentos de referencia (manuales del sistema y leyes/reglamentos) para responder. Si el usuario pide un caso de prueba, estructura tu respuesta con: Objetivo, Precondiciones, Pasos, Resultado esperado. Si pide una explicación o valida una regla de negocio, respóndelo apoyándote en el manual y/o la ley correspondiente. Si la información no está en los documentos, dilo claramente y no inventes.

DOCUMENTOS DE REFERENCIA:
{contexto}

PREGUNTA DEL USUARIO:
{pregunta}

Cita de qué documento (manual o ley) sacaste cada dato relevante."""
            }
        ],
    )

    for bloque in mensaje.content:
        if bloque.type == "text":
            return bloque.text
    return "No se pudo generar una respuesta."


@app.post("/preguntar")
def preguntar(request: PreguntaRequest):
    embedding = generar_embedding_pregunta(request.pregunta)
    chunks = buscar_chunks_relevantes(embedding, request.pregunta)

    if not chunks:
        return {"respuesta": "No encontre informacion relevante en los documentos.", "fuentes": []}

    respuesta = generar_respuesta(request.pregunta, chunks)
    fuentes = list(set(c["document_name"] for c in chunks))

    return {"respuesta": respuesta, "fuentes": fuentes}


@app.get("/")
def health_check():
    return {"status": "ok"}