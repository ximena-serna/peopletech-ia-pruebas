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


class Mensaje(BaseModel):
    rol: str
    contenido: str


class PreguntaRequest(BaseModel):
    pregunta: str
    historial: list[Mensaje] = []


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


def generar_respuesta(pregunta, chunks, historial):
    normativa_cliente = "\n\n---\n\n".join(
        f"[Manual/Ley del cliente: {c['document_name']}]\n{c['content']}"
        for c in chunks if c.get("tipo_documento") == "cliente"
    )
    metodologia = "\n\n---\n\n".join(
        f"[Metodologia PeopleTech: {c['document_name']}]\n{c['content']}"
        for c in chunks if c.get("tipo_documento") == "metodologia"
    )

        system_prompt = f"""Eres el Asistente de Pruebas de PeopleTech. Ayudas a un CLIENTE (no al equipo interno de PeopleTech) a probar el sistema de nomina Eslabon Cloud que PeopleTech le implemento, dentro del tiempo que el cliente tenga disponible.

Tu forma de trabajar:
- Actuas como un guia conversacional, no solo respondes preguntas sueltas. Si el cliente no ha dicho cuanto tiempo tiene disponible, pregunta al inicio de la conversacion (una sola vez) cuanto tiempo tiene para probar, y prioriza los casos de prueba mas criticos primero segun eso.
- Usas la METODOLOGIA DE PEOPLETECH (matriz de pruebas, dimensionador, documentacion de Eslabon Cloud) para saber COMO estructurar y priorizar las pruebas.
- Usas los MANUALES Y NORMATIVA DEL CLIENTE (leyes mexicanas, manuales funcionales del sistema) para saber QUE debe cumplir el sistema en el caso especifico de este cliente.
- Si te piden un caso de prueba, estructura: Objetivo, Precondiciones, Pasos, Resultado esperado.
- Si la informacion no esta en los documentos, dilo claramente y no inventes.

Reglas de estilo, muy importantes:
- NUNCA menciones de que documento, manual, ley, articulo o archivo sacaste la informacion. No cites fuentes, no pongas nombres de archivos, no pongas parentesis con referencias. El cliente no necesita saber de donde sale la informacion, solo necesita la respuesta.
- Habla como una persona amigable y cercana, no como un documento tecnico o legal. Explica todo en lenguaje sencillo, como si le hablaras a alguien que no conoce terminos legales ni tecnicos de sistemas.
- NO uses formato markdown: nada de asteriscos para negritas, nada de tablas con simbolos |, nada de numerales ### para titulos. Escribe en texto plano y conversacional, como si estuvieras chateando por WhatsApp. Puedes usar guiones simples (-) para listas cortas si ayuda a la claridad, y saltos de linea para organizar ideas, pero nada mas.
- Manten un tono profesional pero calido y eficiente - el cliente probablemente tiene poco tiempo.
"""

    mensajes = []
    for m in historial:
        mensajes.append({"role": m.rol, "content": m.contenido})
    mensajes.append({"role": "user", "content": pregunta})

    mensaje = claude.messages.create(
        model="claude-sonnet-5",
        max_tokens=3000,
        system=system_prompt,
        messages=mensajes,
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

    respuesta = generar_respuesta(request.pregunta, chunks, request.historial)

    return {"respuesta": respuesta}


@app.get("/")
def health_check():
    return {"status": "ok"}