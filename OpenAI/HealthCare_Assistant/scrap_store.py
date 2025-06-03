import os
import requests
import pickle
import time
import concurrent.futures
from typing import List, Union
from tqdm import tqdm
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


CONFIG = {
    "output_dir": "scraping_output",
    "batch_size": 50,
    "max_workers": 8,
    "request_delay": 0.5,
    "faiss_index_name": "combined_faiss_index",
    "processed_log": "processed_urls.log",
    "processed_pdfs_log": "processed_pdfs.log",
    "scraped_data_backup": "scraped_data.pkl",
    "embedding_model": "text-embedding-3-small"
}

def setup_environment():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    for file in [CONFIG["processed_log"], CONFIG["processed_pdfs_log"], CONFIG["scraped_data_backup"]]:
        file_path = os.path.join(CONFIG["output_dir"], file)
        if not os.path.exists(file_path):
            open(file_path, 'w').close()

def get_processed_items(log_file: str) -> set:
    try:
        with open(os.path.join(CONFIG["output_dir"], log_file), 'r') as f:
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        return set()

def process_pdf(pdf_path: str) -> dict:
    try:
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"File path {pdf_path} is not valid.")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        full_text = "\n".join([page.page_content for page in pages])
        return {
            "source": pdf_path,
            "content": full_text,
            "type": "pdf",
            "timestamp": time.time()
        }
    except Exception as e:
        print(f"\nFailed to process {pdf_path}: {str(e)}")
        return {"source": pdf_path, "content": "", "error": str(e), "type": "pdf"}

def scrape_single_url(url: str) -> dict:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(["script", "style", "nav", "footer", "iframe", "header"]):
            element.decompose()
        text = soup.get_text(separator=' ', strip=True)
        time.sleep(CONFIG["request_delay"])
        return {
            "source": url,
            "content": text,
            "type": "url",
            "timestamp": time.time()
        }
    except Exception as e:
        print(f"\nFailed to scrape {url}: {str(e)}")
        return {"source": url, "content": "", "error": str(e), "type": "url"}

def process_batch(items: List[Union[str, dict]], is_pdf: bool = False) -> list:
    results = []
    log_file = CONFIG["processed_pdfs_log"] if is_pdf else CONFIG["processed_log"]
    processed_items = get_processed_items(log_file)
    items_to_process = [item for item in items if item not in processed_items]

    if not items_to_process:
        return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        futures = {executor.submit(process_pdf if is_pdf else scrape_single_url, item): item for item in items_to_process}
        with tqdm(total=len(items_to_process), desc=f"Processing {'PDF' if is_pdf else 'URL'} batch") as pbar:
            for future in concurrent.futures.as_completed(futures):
                item = futures[future]
                try:
                    result = future.result()
                    if result and result.get("content"):
                        results.append(result)
                        with open(os.path.join(CONFIG["output_dir"], log_file), 'a') as f:
                            f.write(item + '\n')
                except Exception as e:
                    print(f"\nError processing {item}: {str(e)}")
                finally:
                    pbar.update(1)
    return results

def create_documents(scraped_data: list) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    documents = []
    for item in scraped_data:
        if not item["content"]:
            continue
        chunks = text_splitter.split_text(item["content"])
        for chunk in chunks:
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "source": item["source"],
                    "type": item["type"],
                    "timestamp": item["timestamp"]
                }
            ))
    return documents

def update_faiss_index(scraped_data: list, existing_index: FAISS = None) -> FAISS:
    embeddings = OpenAIEmbeddings(
        model=CONFIG["embedding_model"]
    )
    documents = create_documents(scraped_data)
    if existing_index:
        print("Updating existing FAISS index...")
        existing_index.add_documents(documents)
        return existing_index
    else:
        print("Creating new FAISS index...")
        return FAISS.from_documents(documents, embeddings)

def save_backup(data: list):
    backup_path = os.path.join(CONFIG["output_dir"], CONFIG["scraped_data_backup"])
    try:
        existing_data = []
        if os.path.exists(backup_path):
            with open(backup_path, 'rb') as f:
                existing_data = pickle.load(f)
        combined_data = existing_data + data
        with open(backup_path, 'wb') as f:
            pickle.dump(combined_data, f)
    except Exception as e:
        print(f"Error saving backup: {str(e)}")

def main(url_list: List[str], pdf_list: List[str] = None):
    setup_environment()

    index_path = os.path.join(CONFIG["output_dir"], CONFIG["faiss_index_name"])
    faiss_index = None
    if os.path.exists(index_path):
        try:
            embeddings = OpenAIEmbeddings(model=CONFIG["embedding_model"])
            faiss_index = FAISS.load_local(index_path, embeddings)
            print("Loaded existing FAISS index")
        except Exception as e:
            print(f"Error loading existing index: {str(e)}")
            faiss_index = None

    if url_list:
        print(f"\nProcessing {len(url_list)} URLs...")
        url_batches = [url_list[i:i + CONFIG["batch_size"]] for i in range(0, len(url_list), CONFIG["batch_size"])]
        for batch in url_batches:
            scraped_batch = process_batch(batch, is_pdf=False)
            if scraped_batch:
                faiss_index = update_faiss_index(scraped_batch, faiss_index)
                save_backup(scraped_batch)

    if pdf_list:
        print(f"\nProcessing {len(pdf_list)} PDFs...")
        pdf_batches = [pdf_list[i:i + CONFIG["batch_size"]] for i in range(0, len(pdf_list), CONFIG["batch_size"])]
        for batch in pdf_batches:
            scraped_batch = process_batch(batch, is_pdf=True)
            if scraped_batch:
                faiss_index = update_faiss_index(scraped_batch, faiss_index)
                save_backup(scraped_batch)

    if faiss_index:
        faiss_index.save_local(index_path)
        print(f"\nFinal combined FAISS index saved to {index_path}")

        query = "What are the symptoms of abdominal aortic aneurysm?"
        print(f"\nRunning similarity search for: '{query}'")
        results = faiss_index.similarity_search(query, k=3)
        for i, doc in enumerate(results, 1):
            print(f"\nResult {i} (Source: {doc.metadata['source']}, Type: {doc.metadata['type']}):")
            print(doc.page_content[:500] + "...")

if __name__ == "__main__":
    # links to scrape data from
    links = ['https://www.nhs.uk/conditions/abdominal-aortic-aneurysm-screening/',
             'https://www.nhs.uk/conditions/abdominal-aortic-aneurysm/',
             'https://www.nhs.uk/conditions/abdominal-aortic-aneurysm/',
             'https://www.nhs.uk/conditions/abdominal-aortic-aneurysm-screening/',
             'https://www.nhs.uk/conditions/abortion/', 'https://www.nhs.uk/conditions/acanthosis-nigricans/',
             'https://www.nhs.uk/conditions/achalasia/', 'https://www.nhs.uk/conditions/acid-and-chemical-burns/',
             'https://www.nhs.uk/conditions/reflux-in-babies/', 'https://www.nhs.uk/conditions/acne/',
             'https://www.nhs.uk/conditions/acoustic-neuroma/', 'https://www.nhs.uk/conditions/acromegaly/',
             'https://www.nhs.uk/conditions/actinic-keratoses/', 'https://www.nhs.uk/conditions/actinomycosis/',
             'https://www.nhs.uk/conditions/acupuncture/', 'https://www.nhs.uk/conditions/acute-cholecystitis/',
             'https://www.nhs.uk/conditions/acute-kidney-injury/',
             'https://www.nhs.uk/conditions/acute-lymphoblastic-leukaemia/',
             'https://www.nhs.uk/conditions/acute-myeloid-leukaemia/',
             'https://www.nhs.uk/conditions/acute-pancreatitis/',
             'https://www.nhs.uk/conditions/acute-respiratory-distress-syndrome/',
             'https://www.nhs.uk/conditions/addisons-disease/', 'https://www.nhs.uk/conditions/adenoidectomy/',
             'https://www.nhs.uk/conditions/adenomyosis/', 'https://www.nhs.uk/conditions/cataracts/',
             'https://www.nhs.uk/conditions/age-related-macular-degeneration-amd/',
             'https://www.nhs.uk/mental-health/conditions/agoraphobia/', 'https://www.nhs.uk/conditions/albinism/',
             'https://www.nhs.uk/conditions/alcohol-misuse/', 'https://www.nhs.uk/conditions/alcohol-poisoning/',
             'https://www.nhs.uk/conditions/alcohol-related-liver-disease-arld/',
             'https://www.nhs.uk/conditions/alexander-technique/', 'https://www.nhs.uk/conditions/alkaptonuria/',
             'https://www.nhs.uk/conditions/allergic-rhinitis/', 'https://www.nhs.uk/conditions/allergies/',
             'https://www.nhs.uk/conditions/altitude-sickness/', 'https://www.nhs.uk/conditions/alzheimers-disease/',
             'https://www.nhs.uk/conditions/lazy-eye/', 'https://www.nhs.uk/conditions/memory-loss-amnesia/',
             'https://www.nhs.uk/conditions/amniocentesis/', 'https://www.nhs.uk/conditions/amputation/',
             'https://www.nhs.uk/conditions/amyloidosis/', 'https://www.nhs.uk/conditions/anabolic-steroid-misuse/',
             'https://www.nhs.uk/conditions/iron-deficiency-anaemia/',
             'https://www.nhs.uk/conditions/vitamin-b12-or-folate-deficiency-anaemia/',
             'https://www.nhs.uk/conditions/anal-cancer/', 'https://www.nhs.uk/conditions/anal-fissure/',
             'https://www.nhs.uk/conditions/anal-fistula/', 'https://www.nhs.uk/conditions/anal-pain/',
             'https://www.nhs.uk/conditions/anaphylaxis/',
             'https://www.nhs.uk/conditions/androgen-insensitivity-syndrome/',
             'https://www.nhs.uk/conditions/abdominal-aortic-aneurysm/',
             'https://www.nhs.uk/conditions/brain-aneurysm/', 'https://www.nhs.uk/conditions/angelman-syndrome/',
             'https://www.nhs.uk/conditions/angina/', 'https://www.nhs.uk/conditions/angioedema/',
             'https://www.nhs.uk/conditions/angiography/', 'https://www.nhs.uk/conditions/coronary-angioplasty/',
             'https://www.nhs.uk/conditions/animal-and-human-bites/',
             'https://www.nhs.uk/conditions/foot-pain/ankle-pain/',
             'https://www.nhs.uk/conditions/ankylosing-spondylitis/',
             'https://www.nhs.uk/mental-health/conditions/anorexia/',
             'https://www.nhs.uk/conditions/lost-or-changed-sense-smell/', 'https://www.nhs.uk/conditions/antacids/',
             'https://www.nhs.uk/conditions/antibiotics/', 'https://www.nhs.uk/conditions/anticoagulants/',
             'https://www.nhs.uk/mental-health/talking-therapies-medicine-treatments/medicines-and-psychiatry/antidepressants/',
             'https://www.nhs.uk/conditions/antifungal-medicines/', 'https://www.nhs.uk/conditions/antihistamines/',
             'https://www.nhs.uk/conditions/antiphospholipid-syndrome/',
             'https://www.nhs.uk/mental-health/conditions/antisocial-personality-disorder/',
             'https://www.nhs.uk/conditions/itchy-anus/',
             'https://www.nhs.uk/mental-health/children-and-young-adults/advice-for-parents/anxiety-disorders-in-children/',
             'https://www.nhs.uk/conditions/aortic-valve-replacement/', 'https://www.nhs.uk/conditions/aphasia/',
             'https://www.nhs.uk/conditions/appendicitis/', 'https://www.nhs.uk/conditions/arrhythmia/',
             'https://www.nhs.uk/conditions/arthritis/', 'https://www.nhs.uk/conditions/arthroscopy/',
             'https://www.nhs.uk/conditions/asbestosis/', 'https://www.nhs.uk/conditions/autism/',
             'https://www.nhs.uk/conditions/aspergillosis/', 'https://www.nhs.uk/conditions/asthma/',
             'https://www.nhs.uk/conditions/astigmatism/', 'https://www.nhs.uk/conditions/ataxia/',
             'https://www.nhs.uk/conditions/atherosclerosis/', 'https://www.nhs.uk/conditions/athletes-foot/']

    pdf_urls = ["pdfs/71763-gale-encyclopedia-of-medicine.-vol.-1.-2nd-ed.pdf",
                "pdfs/Internal Medicine, Getachew Tizazu, Tadesse Anteneh.pdf"]
    main(url_list=links, pdf_list=pdf_urls)
