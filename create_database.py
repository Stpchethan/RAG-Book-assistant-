from document_loaders.bulk_ingest import ingest_all_pdfs


def main():
    results = ingest_all_pdfs("document_loaders")

    print("\n All PDFs indexed successfully!")

    for item in results:
        print(f"\nFile: {item['file']}")
        print(f"Pages: {item['pages']}")
        print(f"Chunks: {item['chunks']}")


if __name__ == "__main__":
    main()