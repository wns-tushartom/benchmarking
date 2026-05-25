import asyncio
import httpx

async def call_chat_service(message: str, use_rag: bool = True):
    url = "http://0.0.0.0:5010/llm/"  # <-- note trailing slash here
    headers = {"Content-Type": "application/json"}
    payload = {
        "message": message,
        "use_rag": use_rag,
        "model": "llama3.3-70b-wns-airesearch-domain",
        "max_tokens": 1000,
        "temperature": 0.7,
        "context_k": 5,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

async def main():
    #question = "What is Businessclass.com and how does it operate?"
    question = "how to refund involuntary reissued ticket?"
    result = await call_chat_service(question)
    print("Chat response:", result.get("response"))
    if "sources" in result and result["sources"]:
        print("\nSources:")
        for i, source in enumerate(result["sources"], 1):
            print(f"[{i}] Document: {source.get('document_id')}, Page: {source.get('page_number')}")
            print(source.get("text", "")[:200])

if __name__ == "__main__":
    asyncio.run(main())
