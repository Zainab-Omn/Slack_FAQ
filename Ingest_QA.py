#!/usr/bin/env python3
"""
Index Slack Q&A JSON into a Qdrant collection with text embeddings.

Example:
    python index_slack_qas.py \
        --file Data/slack_QA.json \
        --collection slack_QA \
        --qdrant-url http://localhost:6333 \
        --model jinaai/jina-embeddings-v2-base-en 

Requirements:
    pip install qdrant-client fastembed

JSON format (per item):
[
  {
    "channel": "some-channel",
    "thread_ts": "1712345678.123456",
    "qas": [
      {
        "asked_by": "alice",
        "answered_by": "bob",
        "question": "Q?",
        "answer": "A."
      }
    ]
  },
  ...
]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding


DEFAULT_EMBED_DIM = 768
DEFAULT_MODEL = "jinaai/jina-embeddings-v2-base-en"


def qdrant_point_id(name: str) -> str:
    """Deterministic UUIDv5 for a given name string."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))

def create_qdrant_collection(
    client: QdrantClient,
    collection_name: str,
    mode: str,  # "dense", "sparse", or "hybrid"
    embedding_dimensionality: int | None = None,  # required for dense/hybrid
    dense_name: str = "dense",
    sparse_name: str = "sparse",
):


    m = mode.lower().strip()
    if m not in {"dense", "sparse", "hybrid"}:
        raise ValueError("mode must be one of: 'dense', 'sparse', 'hybrid'")

    kwargs = {}

    if m in {"dense", "hybrid"}:
        if not embedding_dimensionality:
            raise ValueError("embedding_dimensionality is required for dense or hybrid mode")
        kwargs["vectors_config"] = {
            dense_name: models.VectorParams(
                size=embedding_dimensionality,
                distance=models.Distance.COSINE, 
            )
        }

    if m in {"sparse", "hybrid"}:
        kwargs["sparse_vectors_config"] = {
            sparse_name: models.SparseVectorParams(
                modifier=models.Modifier.IDF  
            )
        }

    return client.create_collection(
        collection_name=collection_name,
        **kwargs
    )

def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    mode : str ,
    vector_size: int | None = None
) -> None:

    if client.collection_exists(collection_name):
        logging.info("Collection '%s' already exists.", collection_name)
        return

    logging.info("Creating collection '%s' (size=%d, distance=COSINE)...", collection_name,vector_size,)

    create_qdrant_collection(client, collection_name, mode , vector_size )
  


    logging.info("Creating payload index on 'channel'...")
    client.create_payload_index(
        collection_name=collection_name,
        field_name="channel",
        field_schema="keyword",
    )


def load_data(json_path: Path) -> List[Dict]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Top-level JSON must be a list of thread objects.")
    return data


def iter_records(data: List[Dict]) -> Iterable[Tuple[str, Dict]]:
    """
    Yield (deterministic_id, record) for each QA entry in the dataset.

    The deterministic id is based on <index_in_thread><thread_ts_nodot>.
    """
    for thread in data:
        thread_ts = str(thread.get("thread_ts", ""))
        thread_ts_nodot = thread_ts.replace(".", "")
        qas = thread.get("qas", [])
        if not isinstance(qas, list):
            continue
        for i, qa in enumerate(qas):
            name = f"{i}{thread_ts_nodot}"
            point_id = qdrant_point_id(name)
            record = {
                "channel": thread.get("channel"),
                "thread_ts": thread_ts,
                "asked_by": qa.get("asked_by"),
                "answered_by": qa.get("answered_by"),
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
            }
            yield point_id, record


def concatenate_text(record: Dict) -> str:
    """Combine question + answer for embedding."""
    return f"{record.get('question','')}\n{record.get('answer','')}".strip()


def fetch_existing_ids(client: QdrantClient, collection_name: str) -> set:
    """Scroll through the collection and collect all existing point IDs."""
    logging.info("Fetching existing IDs from collection '%s' (this may take a while)...", collection_name)
    existing_ids = set()
    next_page = None
    while True:
        result, next_page = client.scroll(
            collection_name=collection_name,
            with_payload=False,
            with_vectors=False,
            limit=10_000,
            offset=next_page,
        )
        for pt in result:
            existing_ids.add(str(pt.id))
        if not next_page or not result:
            break
    logging.info("Found %d existing IDs.", len(existing_ids))
    return existing_ids


# def embed_texts(
#     embedder: TextEmbedding,
#     texts: List[str],
# ) -> List[List[float]]:
#     """Compute embeddings for a batch of texts."""
#     # fastembed returns an iterator; convert to list of vectors
#     return [vec for vec in embedder.embed(texts)]


def upsert_points(
    client: QdrantClient,
    collection_name: str,
    mode : str,
    batch: List[Tuple[str, Dict]],
    model_handle :str
) -> None:
    """Upsert a batch of points into Qdrant."""
    m = mode.lower().strip()

    if not batch:
        return
    points = []
    for pid, record in batch:
        vector = {}
        text=concatenate_text(record)
        if m in {"dense", "hybrid"}:
            vector["dense"] = models.Document(
                text=text,
                model=model_handle,  
            )
        if m in {"sparse", "hybrid"}:
            vector["sparse"] = models.Document(
                text=text,
                model="Qdrant/bm25",  
            )
        points.append(
            models.PointStruct(
                id=pid,
                vector=vector, 
                payload={
                    "channel": record.get("channel"),
                    "thread_ts": record.get("thread_ts"),
                    "asked_by": record.get("asked_by"),
                    "answered_by": record.get("answered_by"),
                    "question": record.get("question"),
                    "answer": record.get("answer"),
                },
            )
        )

    client.upsert(collection_name=collection_name, points=points)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index Slack Q&A JSON into Qdrant.")
    parser.add_argument(
        "--file",
        required=True,
        type=Path,
        help="Path to the JSON file (e.g., Data/slack_QA.json)",
    )
    parser.add_argument(
        "--collection",  
        default="slack_QA",
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
        help="Qdrant URL (e.g., http://localhost:6333)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Embedding model handle for fastembed (e.g., jinaai/jina-embeddings-v2-base-en)",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=DEFAULT_EMBED_DIM,
        help=f"Embedding dimensionality (default: {DEFAULT_EMBED_DIM})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for embeddings/upserts",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip records whose IDs already exist in the collection to avoid re-embedding",
    )
    parser.add_argument(
        "--mode",
        default="dense",
        help="dense, sparse, or hybrid",
     )


    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, 'INFO'),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Initialize clients
    client = QdrantClient(args.qdrant_url)
    if args.mode=="sparse":
       embed_dim = None 
    embed_dim=args.embed_dim
    ensure_collection(client, args.collection, args.mode, embed_dim)

    # Load and flatten records
    data = load_data(args.file)
    pairs = list(iter_records(data))
    if not pairs:
        logging.warning("No records found in input JSON.")
        return 0

    # Optionally get existing IDs to skip
    existing_ids = set()
    if args.skip_existing:
        existing_ids = fetch_existing_ids(client, args.collection)

    # Prepare embedder
    # logging.info("Loading embedding model: %s", args.model)
    # embedder = TextEmbedding(model_name=args.model)

    # Process in batches
    total = 0
    to_process: List[Tuple[str, Dict]] = [
        (pid, rec) for (pid, rec) in pairs if (not args.skip_existing or pid not in existing_ids)
    ]
    skipped = len(pairs) - len(to_process)
    if skipped:
        logging.info("Skipping %d existing records.", skipped)

    logging.info("Indexing %d records (batch size=%d)...", len(to_process), args.batch_size)

    batch_ids: List[str] = []
    batch_records: List[Dict] = []
    for pid, rec in to_process:
        batch_ids.append(pid)
        batch_records.append(rec)

        if len(batch_ids) >= args.batch_size:
            ##texts = [concatenate_text(r) for r in batch_records]
            ##vectors = embed_texts(embedder, texts)

            batch = list(zip(batch_ids, batch_records))
            upsert_points(client,args.collection, args.mode,batch,args.model)

            total += len(batch)
            batch_ids.clear()
            batch_records.clear()

    # Remainer
    if batch_ids:
        #texts = [concatenate_text(r) for r in batch_records]
        #vectors = embed_texts(embedder, texts)
        batch = list(zip(batch_ids, batch_records))
        upsert_points(client,args.collection, args.mode,batch,args.model)
        total += len(batch)

    logging.info("Done. Upserted %d points into collection '%s'.", total, args.collection)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise
