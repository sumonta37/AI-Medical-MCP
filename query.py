import os
import chromadb

from dotenv import load_dotenv

from ollama import Client

from llama_index.core import (
    VectorStoreIndex
)

from llama_index.vector_stores.chroma import (
    ChromaVectorStore
)

from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding
)


# -------------------------------------------------
# STEP 1
# Load environment variables
# -------------------------------------------------

load_dotenv()

api_key = os.getenv(
    "OLLAMA_API_KEY"
)


if not api_key:

    raise ValueError(
        "OLLAMA_API_KEY is missing. "
        "Check your .env file."
    )


# -------------------------------------------------
# STEP 2
# Create Ollama Cloud client
# -------------------------------------------------

ollama_client = Client(

    host="https://ollama.com",

    headers={
        "Authorization":
        f"Bearer {api_key}"
    }
)


# -------------------------------------------------
# STEP 3
# Load embedding model
# -------------------------------------------------

print("Loading embedding model...")


embed_model = HuggingFaceEmbedding(

    model_name=(
        "BAAI/bge-small-en-v1.5"
    )
)


# -------------------------------------------------
# STEP 4
# Connect to existing ChromaDB
# -------------------------------------------------

chroma_client = (
    chromadb.PersistentClient(
        path="./chroma_db"
    )
)


chroma_collection = (
    chroma_client.get_collection(
        name="medical_docs"
    )
)


print(
    "Vectors available:",
    chroma_collection.count()
)


# -------------------------------------------------
# STEP 5
# Connect Chroma with LlamaIndex
# -------------------------------------------------

vector_store = ChromaVectorStore(

    chroma_collection=(
        chroma_collection
    )
)


# -------------------------------------------------
# STEP 6
# Reconstruct index
# -------------------------------------------------

index = VectorStoreIndex.from_vector_store(

    vector_store=vector_store,

    embed_model=embed_model
)


# -------------------------------------------------
# STEP 7
# Create Retriever
# -------------------------------------------------

retriever = index.as_retriever(

    similarity_top_k=3
)


# -------------------------------------------------
# STEP 8
# RAG function
# -------------------------------------------------

def ask_medical_assistant(question):

    # ---------------------------------------------
    # Retrieve relevant chunks
    # ---------------------------------------------

    nodes = retriever.retrieve(
        question
    )


    if not nodes:

        return (
            "I could not find relevant "
            "information in the medical "
            "documents."
        )


    # ---------------------------------------------
    # Build context
    # ---------------------------------------------

    context_parts = []


    for i, node in enumerate(
        nodes,
        start=1
    ):

        text = node.text

        context_parts.append(
            f"[Source {i}]\n{text}"
        )


    context = "\n\n".join(
        context_parts
    )


    # ---------------------------------------------
    # Create RAG system prompt
    # ---------------------------------------------

    system_prompt = f"""
You are a Medical Document Assistant.

Your job is to answer questions using ONLY the
medical document context provided below.

Rules:

1. Use only the provided context.
2. Do not invent medical information.
3. If the answer cannot be found in the context,
   say:
   "I could not find this information in the
   provided medical documents."
4. Keep the answer clear and concise.
5. Mention that the information is educational
   and is not a substitute for professional
   medical advice when appropriate.

MEDICAL DOCUMENT CONTEXT:

{context}
"""


    # ---------------------------------------------
    # Call Ollama Cloud
    # ---------------------------------------------

    response = ollama_client.chat(

        model="gpt-oss:120b",

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


    answer = (
        response["message"]["content"]
    )


    return answer, nodes


# -------------------------------------------------
# STEP 9
# Console Chat Application
# -------------------------------------------------

print("\n" + "=" * 60)

print("MEDICAL AI ASSISTANT")

print("=" * 60)

print(
    "Ask questions about your medical PDFs."
)

print(
    "Type 'exit' to quit.\n"
)


while True:

    question = input(
        "Ask: "
    ).strip()


    # Prevent empty question
    if not question:

        print(
            "\nPlease enter a question.\n"
        )

        continue


    # Exit application
    if question.lower() in [
        "exit",
        "quit"
    ]:

        print(
            "\nGoodbye!"
        )

        break


    try:

        result = (
            ask_medical_assistant(
                question
            )
        )


        if isinstance(
            result,
            tuple
        ):

            answer, nodes = result

        else:

            print(
                "\nAnswer:\n"
            )

            print(result)

            continue


        print(
            "\nAnswer:\n"
        )

        print(answer)


        # -----------------------------------------
        # Display retrieved sources
        # -----------------------------------------

        print(
            "\nRetrieved Sources:"
        )


        for i, node in enumerate(
            nodes,
            start=1
        ):

            metadata = (
                node.metadata
                if node.metadata
                else {}
            )


            file_name = (
                metadata.get(
                    "file_name",
                    "Unknown"
                )
            )


            print(
                f"\n{i}. {file_name}"
            )


            if node.score is not None:

                print(
                    "   Similarity:",
                    round(
                        node.score,
                        4
                    )
                )


        print(
            "\n" + "-" * 60
        )


    except Exception as error:

        print(
            "\nERROR:"
        )

        print(error)

        print()