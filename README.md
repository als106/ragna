# 🧠 RAG-NA: Asistente Conversacional para Estudiantes de la UA

Este proyecto implementa un sistema de generación aumentada por recuperación (RAG) para asistir a estudiantes universitarios mediante una interfaz conversacional basada en modelos de lenguaje.

## 🧱 Tecnologías principales

- Python 3.12
- FastAPI (API REST para integración con Dialogflow)
- LangChain + Chroma (gestión RAG y almacenamiento vectorial)
- Gradio (interfaz local alternativa)
- Modelo de generación: `gemma2:2b` mediante Ollama
- Embeddings: `intfloat/multilingual-e5-small`
- Transformers y Accelerate (para gestión eficiente en GPU)
- Evaluación con BERTScore, ROUGE y Exact Match

## 📦 Estructura del proyecto

- `app_gradio_mod.py` — Interfaz de chat con Gradio.
- `main_mod.py` — Backend de generación de respuestas con FastAPI.
- `vector_mod.py` — Procesamiento de PDFs y CSVs para actualizar la base vectorial.
- `evaluacion.py` — Generación automática de respuestas para evaluación.
- `metricas.py` — Cálculo de métricas (ROUGE, BERTScore, EM).
- `data/` — Carpeta donde se guardan los documentos fuente.
- `.env` — Archivo con variables de entorno (token de Cohere).
- `chrome_langchain_db/` — Base de datos semántica de ChromaDB (generada automáticamente).

## ⚙️ Requisitos

Instala las dependencias necesarias con:

```bash
pip install -r requirements.txt
```
## 🔐 Variables de entorno

Crea un archivo `.env` en la raíz del proyecto con el siguiente contenido:

```
COHERE_API_KEY=TU_API_KEY_AQUI
```
## 🚀 Ejecución

### 1. Actualizar la base vectorial

Antes de ejecutar la interfaz, debes procesar los documentos en `data/`:

```bash
python app_gradio_mod.py --update
```

### 2. Lanzar la interfaz conversacional

```bash
python app_gradio_mod.py
```

### 3. API REST (opcional)

Puedes lanzar el servidor de backend con FastAPI:

```bash
uvicorn main_mod:app --reload
```

---

## 🧪 Formatos soportados

- **PDF**: se segmenta y se guarda por fragmentos con metadatos.
- **CSV**: debe contener columnas `Pregunta` y `Respuesta`.

---

## 🛡️ Control de errores y logs

El sistema está instrumentado con `logging` y captura errores con `try/except`.
Los logs se muestran por consola e incluyen etiquetas informativas como ✅, ❌ y ⚠️.

---

## ✍️ Autor

Álvaro Lario Sánchez  
TFG - Grado en Ingeniería Informática  
Universidad de Alicante (2025)