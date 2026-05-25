import requests


def upload_pdf_file(file_path: str, server_url: str):
    """
    Upload a PDF file to the server using the same form field as the HTML form.
    """
    file_path_obj = Path(file_path)
    with file_path_obj.open("rb") as file_data:
        files = {'file': (file_path_obj.name, file_data, 'application/pdf')}
        response = requests.post(server_url, files=files)
    return response

    #with open(file_path, "rb") as file_data:
    #    files = {'file': (file_path, file_data, 'application/pdf')}
    #    response = requests.post(server_url, files=files)
    #return response


def search_documents(query: str, top_k: int = 5, collection_name: str = None, base_url: str = "http://localhost:5010"):
    """
    Submit a search query to the FastAPI search endpoint.
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
if __name__ == "__main__":
    # Upload PDF
    print("Uploading PDF file to Vector DB...")
    from pathlib import Path
    folder_path = Path('data/pdfs_bkp/')
    for pdf_file in folder_path.glob('*.pdf'):
        pdf_file_path = pdf_file.resolve()
        upload_url = "http://localhost:5010/documents/upload"
        upload_resp = upload_pdf_file(pdf_file_path, upload_url)
        print(f"Upload status code: {upload_resp.status_code}")
        print(f"Upload response: {upload_resp.text}")


    # Perform search
    print("\n\n\nPerforming Search result on Vector DB....")
    search_resp = search_documents("What is Businessclass.com and how it operates?", top_k=3, collection_name="documents")
    print("Search results:")
    print(search_resp['results'][0]['content'])
