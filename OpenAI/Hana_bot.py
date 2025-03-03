import os
import pytesseract
import cv2
from PIL import Image
from typing import List
from docx import Document
from langchain.schema import Document as LangChainDocument
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = openai_api_key


class HanaBOT:
    def __init__(self, index_path="faiss_index"):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        self.index_path = index_path
        self.vectorstore = None
        self.load_existing_index()

    def preprocess_image(self, image_path: str) -> Image:
        """Preprocess the image for better OCR accuracy (for both printed and handwritten text)."""
        image = cv2.imread(image_path)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding to enhance text visibility
        processed_image = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Apply median blur to reduce noise
        processed_image = cv2.medianBlur(processed_image, 3)

        # Save temp processed image
        temp_processed_path = "processed_temp.png"
        print(os.path.abspath(temp_processed_path))
        cv2.imwrite(temp_processed_path, processed_image)

        return temp_processed_path

    def extract_text_from_image(self, image_path: str) -> str:
        """Extracts text from both printed and handwritten images using OCR."""
        processed_image_path = self.preprocess_image(image_path)
        print(processed_image_path)
        extracted_text = pytesseract.image_to_string(
            Image.open(processed_image_path),
            config="--oem 1 --psm 6"  # LSTM OCR with automatic segmentation
        )
        os.remove(processed_image_path)  # Cleanup temp processed file
        return extracted_text.strip()

    def load_file(self, file_path: str, file_type: str) -> List[LangChainDocument]:
        """Loads and splits file content into chunks based on file type."""
        if file_type == "pdf":
            loader = PDFPlumberLoader(file_path)
            pages = loader.load()
        elif file_type == "docx":
            doc = Document(file_path)
            pages = [LangChainDocument(page_content=para.text) for para in doc.paragraphs if para.text.strip()]
        elif file_type in ["png", "jpg", "jpeg"]:
            extracted_text = self.extract_text_from_image(file_path)
            pages = [LangChainDocument(page_content=extracted_text)]
        else:
            raise ValueError("Unsupported file type")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(pages)
        return texts

    def process_and_store(self, texts: List[LangChainDocument]):
        """Creates FAISS index only once and stores embeddings."""
        if not os.path.exists(self.index_path):
            print("Creating new index...............")
            # Create new embeddings if they don't exist
            self.vectorstore = FAISS.from_documents(texts, self.embeddings)
            self.vectorstore.save_local(self.index_path)
        else:
            # Load existing embeddings
            print("Load existing indexes..............")
            self.load_existing_index()

    def load_existing_index(self):
        """Loads existing FAISS index if available."""
        if os.path.exists(self.index_path):
            self.vectorstore = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )

    def retrieve_relevant_docs(self, query: str, k: int = 5) -> List[str]:
        """Retrieves the most relevant documents based on the query."""
        if not self.vectorstore:
            raise ValueError("No document has been processed. Please upload and process a file first.")

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(query)
        return [doc.page_content for doc in docs]

    def generate_answer(self, query: str, context: List[str]) -> str:
        """Uses retrieved context to generate an answer."""
        context_text = "\n".join(context)
        prompt = f"""
          Use the provided context to answer the user questions. The entire context may not be related to user question, so answer wisely from the context.
          If the answer is not available in the context, please respond with "I couldn't find relevant information about that in the provided documents."

          You have to give the information whatever present in the document,pdf and image without any additional information or summarising the information.
          If the user's query is asking for 'Lupin',Ignore the word Lupin from the query and give the result for the remaining query.
          For Example:Scope3 emissions at Lupin == Scope3 emissions.

        ### Context:
        {context_text}

        ### Question:
        {query}
        """
        return self.llm.invoke(prompt).content
