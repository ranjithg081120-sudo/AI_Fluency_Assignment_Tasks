
"""Day 8: Search Chroma and inspect relevance scores."""

from lc_config import get_vectorstore

QUESTIONS = [
    "How much is the late fee for paying fees after the due date?",
    "What time should hostel students return?",
    "Can I write the exam with 70% attendance?",
    "What CGPA do I need to be eligible for placements?",
    "What is the capital of France?",
]


def main():
    store = get_vectorstore()

    for question in QUESTIONS:
        print(f"\nQ: {question}")

        results = store.similarity_search_with_score(question, k=3)

        if not results:
            print("No results returned.")
            continue

        for document, score in results:
            source = document.metadata.get("source", "unknown")
            text = " ".join(document.page_content.split())
            print(f"  {score:.3f} [{source}] {text[:160]!r}")

    print("\nFiltered search: library.md")
    filtered = store.similarity_search(
        "fine for late return",
        k=2,
        filter={"source": "library.md"},
    )

    for document in filtered:
        print(
            f"  [{document.metadata.get('source')}] "
            f"{' '.join(document.page_content.split())[:140]}"
        )


if __name__ == "__main__":
    main()
