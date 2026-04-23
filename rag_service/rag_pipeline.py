import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

INDEX_NAME = "medical-rag"
TOP_K = 3
MODEL_EMBED = "all-MiniLM-L6-v2"
MODEL_LLM = "allenai/Olmo-3-7B-Instruct:publicai"

# Load once at module startup
_embed_model = None
_pinecone_index = None
_llm_client = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(MODEL_EMBED)
    return _embed_model


def _get_index():
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        _pinecone_index = pc.Index(INDEX_NAME)
    return _pinecone_index


def _get_llm():
    global _llm_client
    if _llm_client is None:
        _llm_client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    return _llm_client


def get_rag_answer(query: str) -> str:
    """
    Full RAG pipeline:
    1. Embed the query
    2. Retrieve top-K chunks from Pinecone
    3. Build prompt with context
    4. Generate answer via HuggingFace LLM
    """

    # Step 1: Embed query
    embed_model = _get_embed_model()
    query_vector = embed_model.encode([query])[0].tolist()

    # Step 2: Retrieve from Pinecone
    index = _get_index()
    results = index.query(
        vector=query_vector,
        top_k=TOP_K,
        include_metadata=True,
    )

    matches = results.get("matches", [])
    if not matches:
        return (
            "I don't have specific information about that in my knowledge base. "
            "Please consult your doctor for medical advice."
        )

    # Step 3: Build context string
    context_parts = []
    for match in matches:
        text = match.get("metadata", {}).get("text", "")
        source = match.get("metadata", {}).get("source", "")
        if text:
            context_parts.append(f"[{source}]\n{text}")

    context = "\n\n---\n\n".join(context_parts)

    # Step 4: Generate grounded answer
    prompt = f"""You are a helpful medical information assistant.
Answer the user's question using ONLY the context provided below.
If the context doesn't contain enough information, say so clearly.
Do NOT make up medical facts. Keep the answer concise and clear.

Context:
{context}

Question: {query}

Answer:"""

    llm = _get_llm()
    response = llm.chat.completions.create(
        model=MODEL_LLM,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
    )

    return response.choices[0].message.content.strip()
