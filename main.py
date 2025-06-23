from fastapi import FastAPI
from pydantic import BaseModel
from vector import retriever
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
import os
import logging

# Configuración de logs
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Cargar variables de entorno
load_dotenv()

# -------- CONFIGURACIÓN --------
MODEL_NAME = "gemma2:2b"
app = FastAPI()


class Query(BaseModel):
    question: str
    session_id: str = "default"


model = OllamaLLM(model=MODEL_NAME)

prompt = ChatPromptTemplate.from_template("""
Eres un asistente académico especializado en la Universidad de Alicante. Tu tarea es responder preguntas con información relevante y fiable extraída exclusivamente del contexto proporcionado por los documentos de la UA.

⚠️ Muy importante:
- Si no encuentras la información en el contexto, responde: "No dispongo de información suficiente para responder a esta pregunta". En ese caso, **no añadas una lista de fuentes utilizadas**.
- No inventes leyes, reglamentos, fechas ni procedimientos.
- No completes con datos genéricos si no están presentes en el contexto.

🎓 Ámbitos que puedes cubrir: titulaciones, matrícula, normativa de permanencia, calendario académico, becas, prácticas, movilidad, etc.

Pregunta del usuario:
{question}

📚 Contexto recuperado:
{context}

✍️ Redacta una respuesta clara, breve y bien estructurada, con un tono profesional y accesible.
""")

chat_chain = RunnableSequence(prompt | model | StrOutputParser())

# Cargar token desde variable de entorno
cohere_token = os.getenv("COHERE_API_KEY")
if cohere_token is None:
    raise ValueError("❌ No se ha definido la variable de entorno COHERE_API_KEY")

compressor = CohereRerank(
    model="rerank-multilingual-v3.0", top_n=5, cohere_api_key=cohere_token
)

reranked_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)


@app.post("/chat")
def chat_endpoint(query: Query):
    """Endpoint de generación de respuesta basada en contexto RAG."""
    try:
        context = "\n\n".join(
            [doc.page_content for doc in reranked_retriever.invoke(query.question)]
        )
        response = chat_chain.invoke({"context": context, "question": query.question})
        return {"response": response}
    except Exception as e:
        logging.error(f"❌ Error en API /chat: {e}")
        return {"response": "⚠️ Error al procesar la consulta."}
