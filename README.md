# RAG Practice

A local command-line application that indexes PDF and text files, retrieves
relevant passages from ChromaDB, and can answer questions using OpenAI.

## Requirements

- Python 3.11 or newer
- An OpenAI API key for the `ask` command (ingestion does not require one)

## Install

From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` before asking questions.
Do not commit `.env`; it is ignored by Git.

## Use

Put PDF or UTF-8 text files under `Documents/` (subdirectories are scanned).
Then, from the repository root:

```powershell
rag-practice ingest
rag-practice ask "How many paid vacation days do full-time employees receive?"
```

The equivalent module invocation works without relying on the console-script
PATH:

```powershell
python -m rag_practice ingest
python -m rag_practice ask "Your question here" --top-k 5 --min-score 0
```

The first ingestion run downloads the configured sentence-transformer model.
The ChromaDB index is stored under `.data/chroma/`; it is local generated data
and is not tracked by Git. Run `ingest` again after adding or changing source
documents. Ingestion is idempotent for identical chunks.

## Configuration

Settings may be supplied as environment variables or in a root `.env` file:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | unset | Required only for `ask` |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat completion model |
| `RAG_DOCUMENTS_DIR` | `Documents` | Input PDF and text files |
| `RAG_VECTOR_STORE_DIR` | `.data/chroma` | Persistent vector index |
| `RAG_COLLECTION_NAME` | `pdf_collection` | ChromaDB collection |
| `RAG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `RAG_CHUNK_SIZE` | `1000` | Chunk size in characters |
| `RAG_CHUNK_OVERLAP` | `200` | Overlap between chunks |

For backwards compatibility, the app also reads `API_KEY` and a local
`Document_Ingestion/.env` if no root `.env` exists. Prefer moving your key to
the root `.env` as `OPENAI_API_KEY`.

## Development and tests

```powershell
python -m unittest discover -s tests -v
```

The tests use a temporary ChromaDB directory and do not modify the local index.
