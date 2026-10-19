"""
RAG retrieval

- Chunkcing docs
- Embedding
- Storing in Chroma
- Searching
"""
import sys
# from functools import lru_cache TODO
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer



ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "docs"   # /data/docs
CHROMA_DIR = ROOT / "chroma"        # /chroma
COLLECTION = "tax_docs"
MODEL_NAME = "all-MiniLM-L6-v2"



# load the model once
model = SentenceTransformer(MODEL_NAME)
print(f"model {MODEL_NAME} is loaded!")


def embed(text):
    """ embed a list of strings to normalized vectors """
    return model.encode(text, normalize_embeddings=True).tolist()



def get_client():
    """ open the chroma database """
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def chunk_markdown(path):
    """split a md file to chunks (by '## ' sections)"""
    text = path.read_text(encoding="utf-8")
    title, _, rest = text.partition("\n")
    chunks = []
    for block in rest.split("\n## ")[1:]:
        heading, _, body = block.partition("\n")
        chunks.append({
            "id": f"{path.stem}-{len(chunks)}",
            "text": f"{title} - {heading.strip()}\n{body.strip()}",
            "source": path.name,
            "section": heading.strip(),
        })
    return chunks




def build_index():
    """Build the index from all the md files"""
    # get all the chunks
    chunks = []
    for file in sorted(DOCS_DIR.glob("*.md")):
        for chunk in chunk_markdown(file):
            chunks.append(chunk)

    client = get_client()
    try:
        client.delete_collection(COLLECTION)
    except Exception: 
        pass

    collection = client.create_collection(
        COLLECTION, configuration={"hnsw": {"space": "cosine"}}
    )

    collection.add(
        ids = [chunk["id"] for chunk in chunks],
        documents = [chunk["text"] for chunk in chunks],
        embeddings = embed([chunk["text"] for chunk in chunks]),
        metadatas = [{"source": chunk["source"], "section": chunk["section"]} for chunk in chunks]
    )

    print(f"Indexed {len(chunks)} chunks from {DOCS_DIR.name}")



def search_documents(query, k=3):
    """ returns k chunks which are closest to query """
    collection = get_client().get_collection(COLLECTION)
    res = collection.query(query_embeddings=embed([query]), n_results=k)
    return [
        {"text": doc, "source": meta["source"], "section": meta["section"], "distance": distance}
        for doc,meta,distance in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]



TEST_QUERIES = [
    "What is the standard VAT rate?",
    "When is the payment deadline for invoices?",
    "What does the reverse-charge rule mean?",
    "How long do I have to keep invoices?",
    "Can I charge extra if a client pays late?",
    "What is the current weather in Ljubljana?",  # off-topic query
]



if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build_index()
    else:
        # if not query input, fallback to example queries
        queries = sys.argv[1:]
        if not queries: queries = TEST_QUERIES

        for q in queries:
            print(f"\nQ: {q}")
            for hit in search_documents(q):
                print(f"\tdistance: {hit['distance']:.3f} | File: {hit['source']} | Section: {hit['section']}")


