from qdrant_client import QdrantClient, models

# Defaults (can be overridden from CLI)
DEFAULT_HOST = "http://localhost"
DEFAULT_PORT = 6333
DEFAULT_MODEL = "jinaai/jina-embeddings-v2-base-en"


def make_client(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
    """Create a Qdrant client."""
    return QdrantClient(f"{host}:{port}")


def search_sparse(client, query: str, channel: str = "#course-llm-zoomcamp", limit: int = 1):
    results = client.query_points(
        collection_name="salck_sparse",
        query=models.Document(
            text=query,
            model="Qdrant/bm25",
        ),
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="channel",
                    match=models.MatchValue(value=channel),
                )
            ]
        ),
        using="sparse",
        limit=limit,
        with_payload=True,
    )
    return results.points


def search_dense(client, query: str, model_handle: str, channel: str = "#course-llm-zoomcamp", limit: int = 1):
    results = client.query_points(
        collection_name="salck_dense",
        query=models.Document(
            text=query,
            model=model_handle,
        ),
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="channel",
                    match=models.MatchValue(value=channel),
                )
            ]
        ),
        using="dense",
        limit=limit,
        with_payload=True,
    )
    return results.points


def search_hyprid(client, query: str, model_handle: str, channel: str = "#course-llm-zoomcamp", limit: int = 1):
    results = client.query_points(
        collection_name="salck_hyprid",
        prefetch=[
            models.Prefetch(
                query=models.Document(
                    text=query,
                    model=model_handle,
                ),
                using="dense",
                limit=(5 * limit),
            ),
            models.Prefetch(
                query=models.Document(
                    text=query,
                    model="Qdrant/bm25",
                ),
                using="sparse",
                limit=(5 * limit),
            ),
        ],
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="channel",
                    match=models.MatchValue(value=channel),
                )
            ]
        ),
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        with_payload=True,
    )
    return results.points


# Function registry (needs client + model when called)
search_functions = {
    "sparse": lambda client, query, model_handle, channel, limit: search_sparse(client, query, channel, limit),
    "dense": lambda client, query, model_handle, channel, limit: search_dense(client, query, model_handle, channel, limit),
    "hyprid": lambda client, query, model_handle, channel, limit: search_hyprid(client, query, model_handle, channel, limit),
}


def run_search(method: str, query: str, client=None, model_handle: str = DEFAULT_MODEL, channel: str = "#course-llm-zoomcamp", limit: int = 1):
    """Dynamically run a search by method name."""
    if method not in search_functions:
        raise ValueError(f"Unknown search method: {method}. Choose from {list(search_functions.keys())}.")
    if client is None:
        client = make_client()
    return search_functions[method](client, query, model_handle, channel, limit)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Qdrant search")
    parser.add_argument("method", choices=search_functions.keys(), help="Search method")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--channel", default="#course-llm-zoomcamp", help="Slack channel")
    parser.add_argument("--limit", type=int, default=1, help="Number of results")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Qdrant host (default: localhost)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Qdrant port (default: 6333)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model (default: jinaai/jina-embeddings-v2-base-en)")

    args = parser.parse_args()

    client = make_client(host=args.host, port=args.port)

    results = run_search(args.method, args.query, client=client, model_handle=args.model, channel=args.channel, limit=args.limit)
    for r in results:
        print(r.payload['answer'])
