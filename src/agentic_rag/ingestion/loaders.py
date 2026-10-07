from pathlib import Path

from agentic_rag.storage.models import Document

SUPPORTED = {
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".html": "text/html",
    ".pdf": "application/pdf",
}


def load_file(path: str | Path, source: str | None = None) -> Document:
    file = Path(path).resolve()
    if file.suffix.lower() not in SUPPORTED:
        raise ValueError(f"unsupported file type: {file.suffix}")
    if not file.is_file():
        raise FileNotFoundError(file)
    content = file.read_text(encoding="utf-8")
    return Document.from_content(
        source or str(file), file.name, SUPPORTED[file.suffix.lower()], content
    )
