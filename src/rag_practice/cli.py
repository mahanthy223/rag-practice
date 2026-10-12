from __future__ import annotations

import argparse
import logging
from rag_practice.ingestion import ingest
from rag_practice.rag import RAGService
from rag_practice.settings import Settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-practice",
        description="Index local PDF/TXT files and ask questions about them.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ingest", help="Load, split, embed, and index source files.")
    ask_parser = commands.add_parser("ask", help="Ask a question about indexed files.")
    ask_parser.add_argument("question", help="Question to answer from the documents.")
    ask_parser.add_argument("--top-k", type=int, default=5)
    ask_parser.add_argument("--min-score", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    try:
        settings = Settings.from_environment()
        if args.command == "ingest":
            ingest(settings)
        else:
            if args.top_k <= 0:
                raise ValueError("--top-k must be greater than zero.")
            if not -1.0 <= args.min_score <= 1.0:
                raise ValueError("--min-score must be between -1 and 1.")
            print(
                RAGService(settings).answer(
                    args.question,
                    top_k=args.top_k,
                    min_score=args.min_score,
                )
            )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        logging.getLogger(__name__).error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
