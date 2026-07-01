# NayePankh Chatbot

A RAG-based chatbot built as a hiring task for NayePankh Foundation, a UP-registered NGO. Answers questions about the foundation using retrieval-augmented generation, not hardcoded FAQs.

**Live demo:** [add HF Spaces link]

## What it does

Answers questions about NayePankh Foundation — history, mission, registration status — by retrieving relevant context from the foundation's knowledge base and passing it to an LLM, instead of relying on static if/else responses.

## How it works

1. Foundation knowledge base (`nayepankh_knowledge.txt`) is chunked and embedded
2. Chunks stored in ChromaDB
3. On a question, relevant chunks are retrieved and passed to Groq's LLM as context
4. Answer is grounded in the retrieved chunks

## Tech stack

- Streamlit — UI
- ChromaDB — vector storage
- sentence-transformers — embeddings
- Groq API — LLM inference
- BeautifulSoup — content parsing

## Run locally

```bash
git clone https://github.com/Rahulgupta-63/nayepankh-chatbot.git
cd nayepankh-chatbot
pip install -r requirements.txt
```

Create `.env`:
```
GROQ_API_KEY=your_key_here
```

Run:
```bash
streamlit run app.py
```

## Note

Built as a hiring-process task, not a production deployment for the organization.

## Limitations

- Knowledge base is a single static text file — no update mechanism yet
- No conversation memory across turns
