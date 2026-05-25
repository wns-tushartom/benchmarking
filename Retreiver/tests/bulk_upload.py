import requests
from pathlib import Path

def upload_pdf_file(file_path: str, server_url: str):
    """
    Upload a PDF file to the server using the same form field as the HTML form.
    """
    file_path_obj = Path(file_path)
    with file_path_obj.open("rb") as file_data:
        files = {'file': (file_path_obj.name, file_data, 'application/pdf')}
        response = requests.post(server_url, files=files)
    return response

if __name__ == "__main__":
    # Upload PDF files from /home/pdfs directory
    print("Uploading PDF files from /home/pdfs to Vector DB...")
    folder_path = Path('/home/pdfs')

    # Check if directory exists
    if not folder_path.exists():
        print(f"Error: Directory {folder_path} does not exist!")
        exit(1)

    upload_url = "http://0.0.0.0:5010/documents/upload"

    for pdf_file in folder_path.glob('*.pdf'):
        pdf_file_path = pdf_file.resolve()
        print(f"Uploading: {pdf_file_path}")

        upload_resp = upload_pdf_file(pdf_file_path, upload_url)
        print(f"Upload status code: {upload_resp.status_code}")
        print(f"Upload response: {upload_resp.text}")
        print("-" * 50)

    print("All PDF uploads completed!")
