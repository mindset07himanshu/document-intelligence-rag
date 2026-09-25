from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from .config import get_settings


def build_index(data_dir: str | None = None) -> int:
    """Build or rebuild the local Chroma index from PDFs in data_dir."""
    settings = get_settings()
    data_path = Path(data_dir or settings.data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(data_path.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDF files found in {data_path.resolve()}. Add at least one PDF first."
        )

    documents = PyPDFDirectoryLoader(str(data_path)).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        collection_name="document_knowledge",
        embedding_function=embeddings,
        persist_directory=settings.persist_dir,
    )

    existing = vectorstore.get()
    if existing.get("ids"):
        vectorstore.delete(ids=existing["ids"])

    vectorstore.add_documents(chunks)
    print(f"Indexed {len(documents)} pages into {len(chunks)} chunks.")
    return len(chunks)


if __name__ == "__main__":
    build_index()
