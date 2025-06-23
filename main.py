from fastapi import FastAPI
from pydantic import BaseModel
from vector import retriever
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

# -------- CONFIGURACIÓN --------
MODEL_NAME = "llama3.2:1b"
app = FastAPI()

class Query(BaseModel):
    question: str
    session_id: str = "default"

model = OllamaLLM(model="llama3.2:1b")

prompt = ChatPromptTemplate.from_template("""
Eres un asistente académico especializado en la Universidad de Alicante. Tu tarea es responder con precisión y claridad a preguntas (Pregunta) sobre titulaciones, asignaturas, normativa, servicios universitarios, calendarios académicos, prácticas externas, movilidad, acceso y matrícula, entre otros temas relevantes de la UA.

Antes de generar la respuesta, consulta los documentos proporcionados (Contexto) y extrae solo la información más relevante y actual. Si no encuentras una respuesta en las fuentes, indica educadamente que no dispones de datos sobre ello.

Responde de forma clara, ordenada y con un tono profesional pero cercano. Si es útil, estructura tu respuesta con viñetas o apartados.

Pregunta:
{question}

Contexto:
{context}

💬 Respuesta:
""")


chat_chain = RunnableSequence(
    prompt | model | StrOutputParser()
)

compressor = CohereRerank(
    model="rerank-multilingual-v3.0",
    top_n=10,
    cohere_api_key="5XonK5ImNmrWZFGE9yL2MPHAv74oBoKElF4RXC8S"
)
reranked_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

@app.post("/chat")
def chat_endpoint(query: Query):
    context = "\n\n".join([doc.page_content for doc in reranked_retriever.invoke(query.question)])
    response = chat_chain.invoke({
        "context": context,
        "question": query.question
    })
    return {"response": response}

