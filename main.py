from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from vector import retriever

# -------- CONFIGURACIÓN --------
SESSION_ID = "default"
MODEL_NAME = "llama3.2:1b"

# --- Modelo y prompt ---
model = OllamaLLM(model=MODEL_NAME)

prompt = ChatPromptTemplate.from_template("""
Eres un asistente experto en temas universitarios de la Universidad de Alicante (UA).
Usa solo el contexto que se te da. No inventes ni hagas suposiciones.
Responde de forma clara y concisa.

{context}

Historial:
{history}

Pregunta: {question}

""")


# --- Cadena básica ---
chain = prompt | model

# --- Fábrica de historial por sesión ---

def memory_factory() -> BaseChatMessageHistory:
    return ChatMessageHistory()

# --- Runnable con historial ---
chat_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: memory_factory(),
    input_messages_key="question",
    history_messages_key="history",
)

# --------- Bucle principal ---------
while True:
    print("-------------------------------")
    question = input("Haz tu pregunta (q to quit): ")
    print("")

    if question.lower().strip() == "q":
        break

    # Recuperar contexto con RAG
    context = retriever.invoke(question)

    # Ejecutar cadena con historial
    response = chat_with_history.invoke(
        {"context": context, "question": question},
        config={"configurable": {"session_id": SESSION_ID}},
    )

    print(response)
