from fastapi import FastAPI, Request
from pydantic import BaseModel
from vector import retriever
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_core.output_parsers import StrOutputParser

# -------- CONFIGURACIÓN --------
SESSION_ID = "default"
MODEL_NAME = "llama3.2:1b"
app = FastAPI()

class Query(BaseModel):
    question: str
    session_id: str = "default"

model = OllamaLLM(model="llama3.2:1b")

prompt = ChatPromptTemplate.from_template("""
Eres un asistente académico de la Universidad de Alicante.

Usa SOLO el contexto para responder.  
No inventes. Si no tienes suficiente información, responde: "No lo sé con certeza".

📄 Cuando la respuesta esté en el contexto, incluye la frase exacta entre comillas.

❓ Pregunta:
{question}

📄 Contexto:
{context}

💬 Respuesta:
""")


def memory_factory() -> BaseChatMessageHistory:
    return ChatMessageHistory()

chat_with_history = RunnableWithMessageHistory(
    prompt | model | StrOutputParser(),
    lambda session_id: memory_factory(),
    input_messages_key="question",
    history_messages_key="history",
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
    response = chat_with_history.invoke(
        {"context": context, "question": query.question},
        config={"configurable": {"session_id": query.session_id}},
    )
    return {"response": response}

