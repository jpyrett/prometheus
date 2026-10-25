"""The smallest possible Chroma example, then the same idea through LangChain.

Part 1 is the class slide. It uses hand-made vectors, so there is no model to download.
Run from the repo root with the `cmu` conda env active:

    python labs/module3_rag/chroma_demo.py

Walk-through: labs/module3_rag/hpw_to_do_langchain_rag.md, section 0.4.
"""
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

TEXTS = ["Refunds: 30 days.", "Reset using email."]
IDS = ["refund", "login"]
VECTORS = [[1.0, 0.0], [0.0, 1.0]]  # hand-made "embeddings": one direction per topic

# The slide uses chromadb.Client(); we also switch Chroma's usage reporting off (design choice 10).
c = chromadb.Client(Settings(anonymized_telemetry=False))

# 1) The class slide. Store two texts with vectors we wrote ourselves, ask for the nearest to [1, 0].
print("1) class slide")
col = c.create_collection("demo", embedding_function=None)  # None = "I bring my own vectors"
col.add(ids=IDS, documents=TEXTS, embeddings=VECTORS)
print("  ", col.query(query_embeddings=[[1.0, 0.0]], n_results=1)["documents"])

# 2) How "near" is measured. Ask for both results and look at the distances (lower is closer).
print("2) distances")
res = col.query(query_embeddings=[[1.0, 0.0]], n_results=2)
print("   Chroma default metric:", list(zip(res["ids"][0], res["distances"][0])))
cosine = c.create_collection("demo_cosine", embedding_function=None,
                             configuration={"hnsw": {"space": "cosine"}})  # fixed when the collection is created
cosine.add(ids=IDS, documents=TEXTS, embeddings=VECTORS)
res = cosine.query(query_embeddings=[[1.0, 0.0]], n_results=2)
print("   cosine metric        :", list(zip(res["ids"][0], res["distances"][0])))


# 3) The same demo the way Prometheus does it. LangChain's Chroma wrapper needs an embeddings
#    object with two methods; a real one (HuggingFaceEmbeddings) turns text into 384 numbers.
#    This toy one just looks for the word "refund", so we still need no download.
class ToyEmbeddings(Embeddings):
    def _vector(self, text: str) -> list[float]:
        return [1.0, 0.0] if "refund" in text.lower() else [0.0, 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


print("3) the same through LangChain: relevance of each text to 'Can I get a refund?' (higher is closer)")
# Expect a LangChain UserWarning below for the default metric: one score falls under 0. That is the point.
for label, config in [("default metric", None), ("cosine metric ", {"hnsw": {"space": "cosine"}})]:
    store = Chroma(client=c, collection_name=f"demo_langchain_{label.split()[0]}",
                   embedding_function=ToyEmbeddings(), collection_configuration=config)
    store.add_documents([Document(page_content=t) for t in TEXTS], ids=IDS)  # LangChain embeds the texts
    hits = store.similarity_search_with_relevance_scores("Can I get a refund?", k=2)  # and the question
    print(f"   {label}:", "  ".join(f"{doc.id}={relevance:+.2f}" for doc, relevance in hits))
