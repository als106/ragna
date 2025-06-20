# Ragna – Chatbot académico con RAG + Dialogflow

Este proyecto es un asistente conversacional inteligente diseñado para estudiantes universitarios. Utiliza una arquitectura **Retrieval-Augmented Generation (RAG)** integrada con **LangChain**, **Hugging Face Transformers**, **ChromaDB** y **Dialogflow** para proporcionar respuestas precisas basadas en documentos reales como guías docentes y archivos CSV de preguntas frecuentes.

## 🧱 Tecnologías principales

- Python 3.12
- FastAPI (webhook para Dialogflow)
- LangChain + Chroma
- flan-t5-large (modelo de Hugging Face)
- Transformers, Accelerate
- Hugging Face Embeddings (`intfloat/multilingual-e5-base`)

## 📂 Estructura
```plaintext
.
├── build_index.py
├── data/
│ └── qa.csv
├── dialogflow_webhook.py
├── loaders/
│ ├── init.py
│ ├── pdf_loader.py
│ └── qa_loader.py
├── processing/
│ ├── init.py
│ ├── splitter.py
│ └── metadata.py
├── embeddings/
│ ├── init.py
│ └── embedder.py
├── vectorstore/
│ ├── init.py
│ ├── indexer.py
│ └── cleaner.py
└── requirements.txt
```


## 🚀 Cómo usar

1. Instala dependencias:
   ```bash
   pip install -r requirements.txt

python build_index.py --reset

2. Indexa documentos:
    ```bash
    python build_index.py --reset

3. Lanza el webhook para Dialogflow:
    ```bash
    uvicorn dialogflow_webhook:app --port 8080
