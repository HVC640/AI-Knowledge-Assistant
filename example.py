from docmind.app.ingestion.parser import Parser
from docmind.app.ingestion.chunker import Chunker
from docmind.app.vectorstore.qdrant_client import QdrantVectorStore

if __name__ == "__main__":
    # source_path = "C:/Workspace/projects/AI-Knowledge-Assistant/docs/Stryker Corporation.pdf"
    # pages = Parser(source_path).parse()
    # chunker = Chunker()
    # chunks = chunker.chunk_pages(pages, source_path)
    # print(f"Total chunks created: {len(chunks)}")

    vector_store = QdrantVectorStore()
    # vector_store.upsert_chunks(chunks)
    query = "What are the CONDITIONS PRECEDENT TO CREDIT EXTENSIONS ?"
    results = vector_store.search(query)
    print(f"Top {len(results)} results for query: '{query}'")
    for idx, result in enumerate(results):
        print(f"\nResult {idx + 1}:")
        print(f"ID: {result.id}")
        print(f"Chunk ID: {result.chunk_id}")
        print(f"Score: {result.score}")
        print(f"Text: {result.text}")
        print(f"Metadata: {result.metadata}")




