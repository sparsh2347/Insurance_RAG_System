import os
from google.cloud import documentai_v1 as documentai
from mimetypes import guess_type
import os
import requests
import tempfile
from dotenv import load_dotenv # type: ignore
# from urllib.parse import urlparse
load_dotenv()

# PROCESSOR_ID="e8ff79a109fb9472"
# PROJECT_ID="bajaj-hackathon-468218"
PROCESSOR_ID=os.getenv("PROCESSOR_ID")
PROJECT_ID=os.getenv("PROJECT_ID")

def extract_text_from_file(file_path: str, processor_id: str, project_id: str, location: str = "us") -> str:
    """
    Extracts text from a PDF or DOCX file using Google Document AI.
    
    Args:
        file_path (str): Path to the input file (.pdf or .docx).
        processor_id (str): The ID of the Document AI processor.
        project_id (str): Your Google Cloud project ID.
        location (str): Processor location (default is 'us').

    Returns:
        str: Extracted plain text content from the document.
    """
    print("-------Starting Extraction-------")
    if file_path.startswith("http://") or file_path.startswith("https://"):
        print("Downloading file from URL...")
        response = requests.get(file_path,stream=True)
        if response.status_code != 200:
            raise Exception("Failed to download file from URL")
        else:
            
    #         file_name = "downloaded.pdf"
    #         file_path = os.path.join("sample_files", file_name) 
    #         with open(file_path, "wb") as f:
    #             f.write(response.content)
    #         print(f"PDF saved permanently at: {file_path}") 

        # Save to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(response.content)
                file_path = tmp.name
                print(f"Saved URL content to temporary file: {file_path}")  

    # Initialize client
    client = documentai.DocumentProcessorServiceClient()
    # print(file_path)
    # Load file
    with open(file_path, "rb") as file:
        file_content = file.read()

    # print(file_content)

    # Detect MIME type
    mime_type, _ = guess_type(file_path)
    # print(mime_type)

    if mime_type not in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        raise ValueError(f"Unsupported file type: {mime_type}")

    # Construct processor resource name
    name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"

    # Prepare request
    raw_document = documentai.RawDocument(content=file_content, mime_type=mime_type)

    #Imageless mode
    request = documentai.ProcessRequest(name=name, raw_document=raw_document,imageless_mode=True)

    # Call API
    result = client.process_document(request=request)
    document = result.document
    print("--------Extraction Completed-------")
    return document.text

# if __name__ == "__main__":
#     file_path = input("Enter the file path or URL of the PDF: ").strip()
#     extracted_text = extract_text_from_file(file_path, PROCESSOR_ID, PROJECT_ID)
#     print(extracted_text[:10000])  # Print first 1000 characters
