import os
import chromadb

from dotenv import load_dotenv
from ollama import Client

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "medical_docs"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

OLLAMA_HOST = "https://ollama.com"
OLLAMA_MODEL = "gpt-oss:120b"

SIMILARITY_TOP_K = 3


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

if not OLLAMA_API_KEY:
    raise ValueError(
        "OLLAMA_API_KEY is missing. "
        "Please add OLLAMA_API_KEY to the .env file."
    )


# ============================================================
# OLLAMA CLOUD CLIENT
# ============================================================

print("Initializing Ollama Cloud client...")

ollama_client = Client(
    host=OLLAMA_HOST,
    headers={
        "Authorization": f"Bearer {OLLAMA_API_KEY}"
    }
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL
)

print("Embedding model loaded.")


# ============================================================
# CHROMADB
# ============================================================

print("Connecting to ChromaDB...")

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

try:
    chroma_collection = chroma_client.get_collection(
        name=COLLECTION_NAME
    )

except Exception as error:
    raise RuntimeError(
        f"Unable to find ChromaDB collection "
        f"'{COLLECTION_NAME}'. "
        f"Run ingest.py first."
    ) from error


vector_count = chroma_collection.count()

print(f"Vectors available: {vector_count}")

if vector_count == 0:
    raise RuntimeError(
        "The medical_docs collection contains no vectors. "
        "Run ingest.py first."
    )


# ============================================================
# LLAMAINDEX VECTOR STORE
# ============================================================

vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection
)


# ============================================================
# RECONSTRUCT LLAMAINDEX INDEX
# ============================================================

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model
)


# ============================================================
# RETRIEVER
# ============================================================

retriever = index.as_retriever(
    similarity_top_k=SIMILARITY_TOP_K
)

print("Medical RAG service initialized successfully.")


# ============================================================
# FUNCTION 1
# RETRIEVE MEDICAL DOCUMENTS
# ============================================================

def retrieve_medical_context(question: str):
    """
    Retrieve relevant chunks from the indexed medical PDFs.

    This function performs retrieval only.
    It does NOT call the LLM.

    This is useful because the ReAct agent can use retrieval
    as a tool independently of answer generation.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    question = question.strip()

    print()
    print("=" * 60)
    print("MEDICAL DOCUMENT RETRIEVAL")
    print("=" * 60)
    print(f"Question: {question}")

    nodes = retriever.retrieve(question)

    contexts = []
    sources = []

    for node in nodes:

        text = node.text.strip()

        if not text:
            continue

        metadata = node.metadata or {}

        file_name = metadata.get(
            "file_name",
            "Unknown document"
        )

        score = None

        if node.score is not None:
            score = round(
                float(node.score),
                4
            )

        contexts.append(text)

        sources.append(
            {
                "file_name": file_name,
                "similarity": score,
                "text": text
            }
        )

    print(f"Retrieved chunks: {len(contexts)}")

    return {
        "contexts": contexts,
        "sources": sources
    }


# ============================================================
# FUNCTION 2
# BUILD RAG CONTEXT
# ============================================================

def build_context(contexts):
    """
    Combine retrieved chunks into a context block
    that can be supplied to the LLM.
    """

    context_parts = []

    for i, context in enumerate(
        contexts,
        start=1
    ):

        context_parts.append(
            f"""
[Medical Document Source {i}]

{context}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# FUNCTION 3
# GENERATE DOCUMENT-GROUNDED ANSWER
# ============================================================

def generate_medical_answer(
    question: str,
    contexts: list[str]
):
    """
    Generate an answer using ONLY retrieved medical
    document context.
    """

    if not contexts:

        return (
            "I could not find relevant information "
            "in the provided medical documents."
        )

    context = build_context(contexts)

    system_prompt = f"""
You are a Medical Document Assistant.

Your job is to answer questions using ONLY the medical
document context supplied below.

IMPORTANT RULES:

1. Use only information contained in the provided
   medical document context.

2. Do not invent symptoms, diagnoses, medicines,
   treatments, dosages, contraindications, or other
   medical information.

3. If the documents do not contain enough information
   to answer the question, respond:

   "I could not find this information in the provided
   medical documents."

4. Clearly distinguish information contained in the
   documents from general advice.

5. Do not diagnose a patient.

6. Do not provide individualized medication or dosage
   recommendations.

7. If the user describes potentially serious symptoms,
   advise them to seek appropriate professional medical
   assessment rather than attempting to diagnose them.

8. Keep the answer clear and concise.

9. When appropriate, state that the information is
   educational and is not a substitute for professional
   medical advice.


MEDICAL DOCUMENT CONTEXT:

{context}
"""

    response = ollama_client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response["message"]["content"]


# ============================================================
# FUNCTION 4
# COMPLETE RAG PIPELINE
# ============================================================

def ask_medical_assistant(question: str):
    """
    Complete RAG pipeline:

    Question
       ↓
    LlamaIndex Retriever
       ↓
    ChromaDB
       ↓
    Retrieved Context
       ↓
    Ollama Cloud
       ↓
    Grounded Answer
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    question = question.strip()

    # --------------------------------------------------------
    # Step 1 - Retrieval
    # --------------------------------------------------------

    retrieval_result = retrieve_medical_context(
        question
    )

    contexts = retrieval_result["contexts"]
    retrieved_sources = retrieval_result["sources"]

    # --------------------------------------------------------
    # Step 2 - Generation
    # --------------------------------------------------------

    answer = generate_medical_answer(
        question=question,
        contexts=contexts
    )

    # --------------------------------------------------------
    # Step 3 - Remove duplicate source filenames
    # --------------------------------------------------------

    sources = []

    seen_files = set()

    for source in retrieved_sources:

        file_name = source["file_name"]

        if file_name in seen_files:
            continue

        sources.append(
            {
                "file_name": file_name,
                "similarity": source["similarity"]
            }
        )

        seen_files.add(file_name)

    # --------------------------------------------------------
    # Step 4 - Return structured response
    # --------------------------------------------------------

    return {
        "answer": answer,
        "sources": sources
    }


# ============================================================
# FUNCTION 5
# TOOL-FRIENDLY MEDICAL DOCUMENT SEARCH
# ============================================================

def search_medical_documents(query: str):
    """
    Tool-friendly version of the medical document search.

    The ReAct agent can call this function when it needs
    information from the indexed medical PDFs.
    """

    result = retrieve_medical_context(query)

    contexts = result["contexts"]
    sources = result["sources"]

    if not contexts:

        return {
            "found": False,
            "context": "",
            "sources": []
        }

    clean_sources = []

    for source in sources:

        clean_sources.append(
            {
                "file_name": source["file_name"],
                "similarity": source["similarity"]
            }
        )

    return {
        "found": True,
        "context": build_context(contexts),
        "sources": clean_sources
    }


def search_medical_documents(query: str):

    result = retrieve_medical_context(query)

    contexts = result["contexts"]
    sources = result["sources"]

    if not contexts:

        return {
            "found": False,
            "context": "",
            "sources": []
        }

    clean_sources = []

    for source in sources:

        clean_sources.append(
            {
                "file_name":
                    source["file_name"],

                "similarity":
                    source["similarity"]
            }
        )

    return {
        "found": True,
        "context":
            build_context(contexts),
        "sources":
            clean_sources
    }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("MEDICAL RAG SERVICE TEST")
    print("=" * 60)

    while True:

        question = input(
            "\nAsk: "
        ).strip()

        if question.lower() in {
            "exit",
            "quit"
        }:
            break

        try:

            result = ask_medical_assistant(
                question
            )

            print()
            print("Answer:")
            print()
            print(result["answer"])

            print()
            print("Sources:")

            if not result["sources"]:
                print("No sources found.")

            for i, source in enumerate(
                result["sources"],
                start=1
            ):

                print(
                    f"{i}. {source['file_name']}"
                )

                if source["similarity"] is not None:

                    print(
                        f"   Similarity: "
                        f"{source['similarity']}"
                    )

        except Exception as error:

            print()
            print(
                f"ERROR: {error}"
            )