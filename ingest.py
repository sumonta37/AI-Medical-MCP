import chromadb

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex
)


from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PDFReader



print("=" * 60)
print("MEDICAL DOCUMENT INGESTION")
print("=" * 60)


# -------------------------------------------------
# STEP 1
# Load medical PDFs
# -------------------------------------------------

print("\n1. Loading medical PDFs...")
pdf_reader = PDFReader()
documents = SimpleDirectoryReader(
    input_dir="./data",
    file_extractor={
        ".pdf": pdf_reader
    }
).load_data()

print(f"Loaded {len(documents)} document/page objects.")

# -------------------------------------------------
# STEP 2
# Verify extracted text
# -------------------------------------------------

print("\n2. Checking extracted text...")

# -------------------------------------------------
# DEBUG: Check what was extracted from PDFs
# -------------------------------------------------

print("\n========== EXTRACTED PDF TEXT ==========")

for i, document in enumerate(documents, start=1):

    print(f"\n--- Document/Page {i} ---")

    print("Metadata:")
    print(document.metadata)

    print("\nExtracted Text:")
    print(document.text[:2000])

print("\n========================================")


# -------------------------------------------------
# STEP 3
# Configure chunking
# -------------------------------------------------

print("\n3. Configuring text chunking...")

text_splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50
)


# -------------------------------------------------
# STEP 4
# Configure embedding model
# -------------------------------------------------

print("\n4. Loading embedding model...")

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

print("   Embedding model loaded.")


# -------------------------------------------------
# STEP 5
# Create persistent ChromaDB
# -------------------------------------------------

print("\n5. Connecting to ChromaDB...")

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)


# -------------------------------------------------
# STEP 6
# Re-create collection
# -------------------------------------------------

# For this learning project, deleting the old
# collection prevents duplicate vectors when
# ingest.py is run repeatedly.

try:
    chroma_client.delete_collection(
        name="medical_docs"
    )

    print("   Old collection removed.")

except Exception:
    pass


chroma_collection = (
    chroma_client.get_or_create_collection(
        name="medical_docs"
    )
)


# -------------------------------------------------
# STEP 7
# Connect ChromaDB with LlamaIndex
# -------------------------------------------------

vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection
)


storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)


# -------------------------------------------------
# STEP 8
# Build index
# -------------------------------------------------

print("\n6. Creating embeddings and index...")

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model,
    transformations=[
        text_splitter
    ],
    show_progress=True
)


# -------------------------------------------------
# STEP 9
# Show results
# -------------------------------------------------

count = chroma_collection.count()

print("\n" + "=" * 60)

print("INGESTION COMPLETED")

print(f"Vectors stored: {count}")

print("=" * 60)