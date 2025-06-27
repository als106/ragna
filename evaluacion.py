from main import chat_chain, reranked_retriever
import pandas as pd

# Cargar las preguntas y respuestas reales
df = pd.read_csv("data/ua/faq.csv")
preguntas = df["Pregunta"].tolist()
respuestas_reales = df["Respuesta"].tolist()
respuestas_generadas = []

for pregunta in preguntas:
    docs = reranked_retriever.invoke(pregunta)
    contexto = "\n\n".join([doc.page_content for doc in docs])
    respuesta = chat_chain.invoke({"context": contexto, "question": pregunta})
    respuestas_generadas.append(respuesta.strip())

# Guardar resultados para comparación
df_resultados = pd.DataFrame({
    "Pregunta": preguntas,
    "Respuesta_Real": respuestas_reales,
    "Respuesta_Generada": respuestas_generadas
})
df_resultados.to_csv("resultados_evaluacion.csv", index=False)
