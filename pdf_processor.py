
import pymupdf


def extract_pages_from_pdf(pdf_path):
    pages = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text()

            pages.append({
                "page_number": page_number,
                "text": page_text
            })

    return pages


def split_pages_into_chunks(pages, chunk_size=1000, overlap=200):
    chunks = []
    step_size = chunk_size - overlap

    for page in pages:
        text = page["text"]
        page_number = page["page_number"]

        start = 0

        while start < len(text):
            end = start + chunk_size

            chunks.append({
                "page_number": page_number,
                "text": text[start:end]
            })

            start += step_size

    return chunks


if __name__ == "__main__":
    pages = extract_pages_from_pdf("sample.pdf")
    chunks = split_pages_into_chunks(pages)

    print("Total pages:", len(pages))
    print("Total chunks:", len(chunks))

    if chunks:
        print("\nFirst chunk:")
        print(chunks[0]["text"])
        print("Page number:", chunks[0]["page_number"])
