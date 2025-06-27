import pandas as pd
import evaluate

# Cargar métricas
bertscore = evaluate.load("bertscore")
exact_match_metric = evaluate.load("exact_match")
rouge = evaluate.load("rouge")

# Cargar CSV
df = pd.read_csv("resultados_evaluacion.csv")
df = df.dropna(subset=["Respuesta_Real", "Respuesta_Generada"]).astype(str)

# Extraer listas
predictions = df["Respuesta_Generada"].tolist()
references = df["Respuesta_Real"].tolist()

# BERTScore
bertscore_results = bertscore.compute(predictions=predictions, references=references, lang="es")
bertscore_f1 = sum(bertscore_results['f1']) / len(bertscore_results['f1']) * 100

# Exact Match
em_result = exact_match_metric.compute(predictions=predictions, references=references)
exact_match_score = em_result["exact_match"] * 100

# ROUGE-L
rouge_result = rouge.compute(predictions=predictions, references=references)
rouge_l = rouge_result["rougeL"] * 100

# Mostrar resultados
print("========== MÉTRICAS DE EVALUACIÓN ==========")
print(f"🔹 BERTScore F1 promedio: {bertscore_f1:.2f} %")
print(f"🔹 Exact Match (EM):       {exact_match_score:.2f} %")
print(f"🔹 ROUGE-L:                {rouge_l:.2f} %")
