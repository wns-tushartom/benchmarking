import requests
import asyncio
import httpx
import asyncio
import logging

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomQARAGPipeline:
    def __init__(self):
        return

    #load pdfs into vector database
    def upload_pdf_file(self, file_path: str, base_url: str = "http://localhost:8000"):
        """
        Upload a PDF file to the server using the same form field as the HTML form.
        """

        server_url = f"{base_url}/documents/upload"
        with open(file_path, "rb") as file_data:
            files = {'file': (file_path, file_data, 'application/pdf')}
            response = requests.post(server_url, files=files)
        return response

    #search for embeddings
    def search_documents(self, query: str, top_k: int = 3, collection_name: str = "documents", base_url: str = "http://localhost:8000"):
        """
        submit a search query to the FastAPI search endpoint.
        """
        url = f"{base_url}/search"
        payload = {
            "query": query,
            "top_k": top_k,
            "collection_name": collection_name,
        }
        response = requests.post(url, json=payload)
        try:
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            print(f"Search failed: {e}")
            print(f"Response: {response.text}")
            return None
    #query llm
    async def call_llm_service(self, message: str, use_rag: bool = True):
        url = "http://localhost:8000/llm/"  # <-- note trailing slash here
        headers = {"Content-Type": "application/json"}
        payload = {
            "message": message,
            "use_rag": use_rag,
            "model": "gpt-4o",
            "max_tokens": 1000,
            "temperature": 0.7,
            "context_k": 5,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

    async def process_query(self, query, pdf_file_path):
        # Upload PDF
        print("Uploading PDF file to Vector DB...")
        #pdf_file_path = "tests/test.pdf"
        upload_resp = self.upload_pdf_file(pdf_file_path)
        print(f"Upload status code: {upload_resp.status_code}")
        print(f"Upload response: {upload_resp.text}")

        # Perform embedding search
        print("\n--------------------------------------------------------------")
        print("\nPerforming Search result on Vector DB....")
        search_resp = self.search_documents(query, top_k=2)
        print("Embeddings Chunks Search results:")
        print(search_resp['results'][0]['content'])

        #perform prompt search on llm
        result = await self.call_llm_service(query)
        print("\n--------------------------------------------------------------")
        print("\nLLM response:", result.get("response"))
        if "sources" in result and result["sources"]:
            print("\nSources:")
            for i, source in enumerate(result["sources"], 1):
                print(f"[{i}] Document: {source.get('document_id')}, Page: {source.get('page_number')}")
                print(source.get("text", "")[:200])


if __name__ == "__main__":
    async def main():
        query = "What is Businessclass.com and how it operates?"
        pipeline = CustomQARAGPipeline()
        await pipeline.process_query(query, "tests/test.pdf")
    asyncio.run(main())
