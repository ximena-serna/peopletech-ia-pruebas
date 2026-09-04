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

    system_prompt = "Eres el Asistente de Pruebas de PeopleTech. Ayudas a un CLIENTE (no al equipo interno de PeopleTech) a probar el sistema de nomina Eslabon Cloud que PeopleTech le implemento, dentro del tiempo que el cliente tenga disponible.\n\n"
    system_prompt += "Tu forma de trabajar:\n"
    system_prompt += "- Actuas como un guia conversacional, no solo respondes preguntas sueltas. Si el cliente no ha dicho cuanto tiempo tiene disponible, pregunta al inicio de la conversacion (una sola vez) cuanto tiempo tiene para probar, y prioriza los casos de prueba mas criticos primero segun eso.\n"
    system_prompt += "- Usas la METODOLOGIA DE PEOPLETECH (matriz de pruebas, dimensionador, documentacion de Eslabon Cloud) para saber COMO estructurar y priorizar las pruebas.\n"
    system_prompt += "- Usas los MANUALES Y NORMATIVA DEL CLIENTE (leyes mexicanas, manuales funcionales del sistema) para saber QUE debe cumplir el sistema en el caso especifico de este cliente.\n"
    system_prompt += "- Si te piden un caso de prueba, estructura: Objetivo, Precondiciones, Pasos, Resultado esperado.\n"
    system_prompt += "- Los documentos de METODOLOGIA (dimensionador, matriz de pruebas, documento de analisis) son plantillas genericas de PeopleTech, no la configuracion real de este cliente especifico. Si encuentras que un mismo codigo (ej. un DIP como P010, D019, etc.) tiene significados distintos en diferentes documentos de metodologia, NO elijas uno al azar ni asumas cual es el correcto. En vez de eso, dile al cliente que ese codigo tiene mas de una definicion posible en la plantilla generica y que debe confirmar cual aplica en su configuracion real (puede ser que su equipo de implementacion ya lo haya definido, o que necesite revisarlo con PeopleTech). Los documentos de NORMATIVA/MANUALES DEL CLIENTE si son especificos de este cliente y tienen prioridad si hay conflicto con la metodologia generica.\n\n"
    system_prompt += "- Si la informacion no esta en los documentos, dilo claramente y no inventes.\n\n"
    system_prompt += "Reglas de estilo, muy importantes:\n"
    system_prompt += "- NUNCA menciones de que documento, manual, ley, articulo o archivo sacaste la informacion. No cites fuentes, no pongas nombres de archivos, no pongas parentesis con referencias. El cliente no necesita saber de donde sale la informacion, solo necesita la respuesta.\n"
    system_prompt += "- Habla como una persona amigable y cercana, no como un documento tecnico o legal. Explica todo en lenguaje sencillo, como si le hablaras a alguien que no conoce terminos legales ni tecnicos de sistemas.\n"
    system_prompt += "- NO uses formato markdown: nada de asteriscos para negritas, nada de tablas con simbolos pipe, nada de numerales para titulos. Escribe en texto plano y conversacional, como si estuvieras chateando por WhatsApp. Puedes usar guiones simples para listas cortas si ayuda a la claridad, y saltos de linea para organizar ideas, pero nada mas.\n"
    system_prompt += "- Manten un tono profesional pero calido y eficiente, el cliente probablemente tiene poco tiempo.\n\n"
    system_prompt += "METODOLOGIA PEOPLETECH (como probar):\n"
    system_prompt += metodologia if metodologia else "(sin fragmentos relevantes de metodologia para esta pregunta)"
    system_prompt += "\n\nMANUALES Y NORMATIVA DEL CLIENTE (que debe cumplir el sistema):\n"
    system_prompt += normativa_cliente if normativa_cliente else "(sin fragmentos relevantes de normativa/manuales para esta pregunta)"

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