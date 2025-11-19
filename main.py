import os
import glob
import asyncio
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

app = FastAPI()
load_dotenv()

# ----------- CONFIG -----------
DOC_DIR = "documents"
os.makedirs(DOC_DIR, exist_ok=True)

llm = OllamaLLM(model="mistral", base_url="http://127.0.0.1:11434/")
embeddings = OllamaEmbeddings(model="mistral", base_url="http://127.0.0.1:11434/")

# ----------- PINECONE -----------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "resume-rag"

existing = pc.list_indexes().names()
if INDEX_NAME not in existing:
    pc.create_index(
        name=INDEX_NAME,
        dimension=4096,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)


# ------- CLEAN RESUME TEXT -------
def clean_text(txt):
    txt = txt.replace("\n", " ").replace("\t", " ")
    while "  " in txt:
        txt = txt.replace("  ", " ")
    return txt.strip()


# ------- PROCESS RESUME -------
async def process_resume(filepath):
    filename = os.path.basename(filepath)

    try:
        # 1. Load
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(filepath)
        else:
            return f"Skipping unsupported: {filename}"

        docs = loader.load()
        # 2. Splitting to Chunk
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.split_documents(docs)

        # 3. Convert to embeddings
        vectors = []
        for idx, chunk in enumerate(chunks):
            text = clean_text(chunk.page_content)
            emb = embeddings.embed_query(text)

            vectors.append({
                "id": f"{filename}___chunk_{idx}",
                "values": emb,
                "metadata": {
                    "resume_name": filename,
                    "chunk": idx,
                    "text": text,
                    "source": filepath
                }
            })

        # 4. Upload Embeddings to Pincone
        index.upsert(vectors)
        return f"Indexed {filename}: {len(vectors)} chunks"

    except Exception as e:
        return f"Error in {filename}: {str(e)}"


# ===== TRIGGER (PROCESS ALL RESUMES) =====
@app.get("/trigger")
async def trigger():
    print("STARTED")
    try:
        files = glob.glob(f"{DOC_DIR}/*.pdf") + \
                glob.glob(f"{DOC_DIR}/*.txt") + \
                glob.glob(f"{DOC_DIR}/*.docx")

        if not files:
            return {"message": "No resumes found"}

        results = await asyncio.gather(*(process_resume(f) for f in files))
        print("END")
        return {"status": "done", "results": results}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ===== LIST ALL INDEXED RESUMES =====
@app.get("/list_resumes")
def list_resumes():
    try:
        stats = index.describe_index_stats()
        items = stats["namespaces"][""]["vector_count"]
        return {"total_vectors": items}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ===== ASK ANY QUESTION =====
@app.get("/ask")
def ask(query: str = Query(...)):
    try:
        q_emb = embeddings.embed_query(query)

        result = index.query(
            vector=q_emb,
            top_k=5,
            include_metadata=True
        )

        context = "\n\n".join([m["metadata"]["text"] for m in result["matches"]])

        prompt = f"""
        You are an expert HR recruiter. 
        Use the resume data below to answer the question accurately.

        Resume Data:
        {context}

        Question:
        {query}

        Answer:
        """

        answer = llm.invoke(prompt)

        return {"query": query, "answer": answer}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ===== SEARCH RESUMES FOR SKILL =====
@app.get("/search/skill")
def search_skill(skill: str):
    try:
        q_emb = embeddings.embed_query(f"Skill: {skill}")

        result = index.query(
            vector=q_emb,
            top_k=10,
            include_metadata=True
        )

        resumes = list(set([m["metadata"]["resume_name"] for m in result["matches"]]))

        return {"skill": skill, "matching_resumes": resumes}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
