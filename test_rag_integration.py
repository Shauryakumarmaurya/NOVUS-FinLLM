import asyncio
from core.tools import build_shared_tools
from rag_engine import query as rag_query

def test_rag():
    print("Testing RAG Query...")
    res = rag_query("HINDUNILVR", "exceptional other income", top_k=2)
    for r in res:
         print(f"[{r['relevance']}] {r['text'][:100]}")
         
    print("\nTesting build_shared_tools doc search...")
    tools = build_shared_tools("Dummy text", {}, ticker="HINDUNILVR")
    search_tool_json = tools.execute("search_document", {"query": "exceptional other income", "max_results": 2})
    print(search_tool_json[:500])

if __name__ == "__main__":
    test_rag()
