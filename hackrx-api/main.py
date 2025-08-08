from fastapi import FastAPI, Request, Header, HTTPException
from pydantic import BaseModel
from typing import List
import requests
import openai
import pinecone
from io import BytesIO
from pdfminer.high_level import extract_text
import os

# Load environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENVIRONMENT")
BEARER_TOKEN = os.environ.get("BEARER_TOKEN", "hackrx-secret")

openai.api_key = OPENAI_API_KEY
app = FastAPI()

class QARequest(BaseModel):
    documents: str
    questions: List[str]

class QAResponse(BaseModel):
    answers: List[str]

@app.middleware("http")
async def verify_auth(request: Request, call_next):
    auth_header = request.headers.get("Authorization")
    if auth_header != f"Bearer {BEARER_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    return await call_next(request)

@app.post("/hackrx/run", response_model=QAResponse)
async def run_qa(request: QARequest):
    pdf_text = extract_text(BytesIO(requests.get(request.documents).content))
    chunks = chunk_text(pdf_text)

    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    index_name = "hackrx-index"
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=1536)
    index = pinecone.Index(index_name)

    embedded_chunks = [get_embedding(c) for c in chunks]
    vectors = [(f"id_{i}", vec, {"text": chunk}) for i, (vec, chunk) in enumerate(zip(embedded_chunks, chunks))]
    index.upsert(vectors=vectors)

    answers = []
    for q in request.questions:
        q_embed = get_embedding(q)
        context = retrieve_context(q_embed, index)
        answer = ask_gpt4(q, context)
        answers.append(answer)

    return {"answers": answers}

def chunk_text(text, max_tokens=300):
    lines = text.split('\n')
    chunks, chunk = [], ''
    for line in lines:
        if len(chunk.split()) + len(line.split()) > max_tokens:
            chunks.append(chunk)
            chunk = line
        else:
            chunk += ' ' + line
    if chunk:
        chunks.append(chunk)
    return chunks

def get_embedding(text):
    response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    return response['data'][0]['embedding']

def retrieve_context(q_embedding, index, top_k=3):
    res = index.query(vector=q_embedding, top_k=top_k, include_metadata=True)
    return "\n".join([match["metadata"]["text"] for match in res["matches"]])

def ask_gpt4(question, context):
    prompt = f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {question}
Answer:"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()
