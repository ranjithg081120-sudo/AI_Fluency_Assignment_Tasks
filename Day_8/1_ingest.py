
"""Day 8: Load, split, embed, and store handbook documents."""

import shutil

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from lc_config import (
    BASE_DIR,
    DATA_DIR,
    DB_DIR,
    EMBED_PROVIDER,
    get_vectorstore,
)


def main():
    paths = sorted(DATA_DIR.glob("*.md"))

    if not paths:
        raise SystemExit(f"No Markdown files found in {DATA_DIR}")

    documents = [
        Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"source": path.name},
        )
        for path in paths
    ]

    print(f"Loaded {len(documents)} documents")
    for document in documents:
        print(f"  {document.metadata['source']}")

    if len(documents) != 6:
        raise SystemExit(
            "Expected 6 documents. Check the data folder before ingesting."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)

    print(f"Split into {len(chunks)} chunks")
    if not chunks:
        raise SystemExit("No chunks were produced.")

    # Clear the old collection so repeated runs do not duplicate vectors.
    shutil.rmtree(DB_DIR, ignore_errors=True)

    store = get_vectorstore()
    ids = store.add_documents(chunks)

    print(f"Stored {len(ids)} vectors in {DB_DIR}")
    print(f"Embedding provider: {EMBED_PROVIDER}")
    print(
        "Vector length:",
        len(store.embeddings.embed_query("placement eligibility")),
    )


if __name__ == "__main__":
    main()
