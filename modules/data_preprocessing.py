import os  # For file and directory operations
import json  # For loading Q/A pairs from JSON files
import logging  # For logging information and errors
import pytesseract  # For OCR on images
from pdf2image import convert_from_path  # For converting PDF pages to images
import PyPDF2  # For extracting text from PDF files
import re  # For regular expressions in text cleaning
from typing import List, Tuple, Dict  # For type annotations

logging.basicConfig(level=logging.INFO)  # Set logging level to INFO
logging.info('data_preprocessing module loaded.')  # Log module load

# Utility function to get all PDF files in the data folder
# Returns a list of all PDF file paths in the specified data folder.
def get_pdf_file_paths(data_folder: str) -> List[str]:
    """
    Returns a list of all PDF file paths in the specified data folder.
    Args:
        data_folder (str): Path to the folder containing PDF files.
    Returns:
        List[str]: List of PDF file paths.
    """
    logging.info(f'get_pdf_file_paths called with data_folder: {data_folder}')  # Log function call
    pdf_files = []  # List to store PDF file paths
    for file in os.listdir(data_folder):  # Iterate over files in the folder
        if file.lower().endswith('.pdf'):  # Check if file is a PDF
            pdf_files.append(os.path.join(data_folder, file))  # Add full path to list
    logging.info(f'PDF files found: {pdf_files}')  # Log found PDF files
    return pdf_files

# Example usage (uncomment to use)
# data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')
# pdf_paths = get_pdf_file_paths(data_folder)
# docs, sections = load_and_preprocess_documents(pdf_paths)

"""
Data Collection & Preprocessing Module
Loads financial PDFs, converts to text, cleans, and segments into logical sections.
Also supports loading Q/A pairs from a JSON file in the Q_and_A folder.
"""

def load_and_preprocess_documents(file_paths: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """
    Loads financial statement files, converts to plain text, cleans, and segments into sections.
    Args:
        file_paths (List[str]): List of file paths to financial documents (PDF).
    Returns:
        Tuple[List[str], Dict[str, str]]: List of cleaned document texts, and dict of sectioned texts.
    Processing Steps:
        - Load PDFs and convert to images for OCR.
        - Extract text using OCR and PDF parser.
        - Clean text (remove headers, footers, page numbers).
        - Segment into logical sections (income statement, balance sheet, etc.).
    """
    logging.info(f'load_and_preprocess_documents called with file_paths: {file_paths}')  # Log function call
    docs = []  # List to store cleaned document texts
    sections = {}  # Dictionary to store sectioned texts
    for file_path in file_paths:  # Iterate over each file path
        text = extract_text_from_pdf(file_path)  # Extract text from PDF
        logging.info(f'Extracted text from {file_path}: {text[:100]}...')  # Log extracted text (preview)
        cleaned = clean_text(text)  # Clean the extracted text
        logging.info(f'Cleaned text from {file_path}: {cleaned[:100]}...')  # Log cleaned text (preview)
        docs.append(cleaned)  # Add cleaned text to docs list
        segs = segment_sections(cleaned)  # Segment cleaned text into sections
        logging.info(f'Segmented sections from {file_path}: {list(segs.keys())}')  # Log segmented sections
        sections.update(segs)  # Add sections to sections dict
    logging.info(f'Loaded and preprocessed {len(docs)} documents.')  # Log number of documents processed
    return docs, sections  # Return cleaned docs and sections

# Extracts text from PDF using OCR and PyPDF2
# Returns extracted text as a string

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from PDF using OCR and PyPDF2.
    Args:
        file_path (str): Path to PDF file.
    Returns:
        str: Extracted text.
    """
    logging.info(f'extract_text_from_pdf called with file_path: {file_path}')  # Log function call
    text = ""  # Initialize text variable
    try:
        with open(file_path, 'rb') as f:  # Open PDF file in binary mode
            reader = PyPDF2.PdfReader(f)  # Create PDF reader object
            num_pages = len(reader.pages)  # Get number of pages
            for i, page in enumerate(reader.pages):  # Iterate over each page
                page_text = page.extract_text()  # Try to extract text from page
                if page_text and page_text.strip():  # If text is found
                    text += page_text  # Add to text
                else:
                    # OCR only if no text extracted
                    images = convert_from_path(file_path, first_page=i+1, last_page=i+1)  # Convert page to image
                    for img in images:  # Iterate over images (should be one per page)
                        ocr_text = pytesseract.image_to_string(img)  # Extract text using OCR
                        text += ocr_text  # Add OCR text to text
    except Exception as e:
        logging.error(f"Error extracting text: {e}")  # Log error if extraction fails
    logging.info(f"Extracted {len(text)} characters from {file_path}")  # Log number of characters extracted
    return text  # Return extracted text

# Cleans text by removing headers, footers, and page numbers
# Returns cleaned text as a string

def clean_text(text: str) -> str:
    """
    Cleans text by removing headers, footers, and page numbers.
    Args:
        text (str): Raw extracted text.
    Returns:
        str: Cleaned text.
    """
    logging.info(f'clean_text called with text: {text[:100]}...')  # Log function call
    # Remove page numbers (various formats)
    text = re.sub(r'\n\s*Page[:]?\s*\d+\s*\n', '\n', text, flags=re.IGNORECASE)  # Remove 'Page X' lines
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Remove lines with only numbers
    # Remove headers/footers (case-insensitive, allow extra text)
    text = re.sub(r'NASDAQ[\w\s,.:-]*\n', '', text, flags=re.IGNORECASE)  # Remove NASDAQ headers
    text = re.sub(r'AMZN[\w\s,.:-]*\n', '', text, flags=re.IGNORECASE)  # Remove AMZN headers
    text = re.sub(r'Amazon[\w\s,.:-]*\n', '', text, flags=re.IGNORECASE)  # Remove Amazon headers
    # Remove lines with copyright/date patterns (optional, common in footers)
    text = re.sub(r'Copyright.*\n', '', text, flags=re.IGNORECASE)  # Remove copyright lines
    text = re.sub(r'\d{4} NASDAQ.*\n', '', text, flags=re.IGNORECASE)  # Remove year NASDAQ lines
    # Remove multiple newlines
    text = re.sub(r'\n{2,}', '\n', text)  # Replace multiple newlines with single newline
    logging.info("Text cleaned.")  # Log cleaning complete
    return text  # Return cleaned text

# Segments cleaned text into logical financial sections
# Returns a dictionary mapping section names to text

def segment_sections(text: str) -> Dict[str, str]:
    """
    Segments cleaned text into logical financial sections.
    Args:
        text (str): Cleaned document text.
    Returns:
        Dict[str, str]: Section name to text mapping.
    """
    logging.info(f'segment_sections called with text: {text[:100]}...')  # Log function call
    sections = {}  # Dictionary to store sections
    patterns = {
        'income_statement': r'(?i)(income statement|statement of income|consolidated statements of operations|statements of earnings)[\s\S]*?(?=balance sheet|statement of financial position|cash flow|statement of cash flows|shareholders|comprehensive|notes|$)',
        'balance_sheet': r'(?i)(balance sheet|statement of financial position|consolidated balance sheets)[\s\S]*?(?=income statement|statement of income|cash flow|statement of cash flows|shareholders|comprehensive|notes|$)',
        'cash_flow': r'(?i)(cash flow|statement of cash flows|consolidated statements of cash flows)[\s\S]*?(?=income statement|balance sheet|statement of income|statement of financial position|shareholders|comprehensive|notes|$)',
        'shareholders_equity': r'(?i)(statement of shareholders\' equity|statement of stockholders\' equity|consolidated statements of shareholders\' equity)[\s\S]*?(?=income statement|balance sheet|cash flow|comprehensive|notes|$)',
        'comprehensive_income': r'(?i)(statement of comprehensive income|consolidated statements of comprehensive income)[\s\S]*?(?=income statement|balance sheet|cash flow|shareholders|notes|$)',
        'notes': r'(?i)(notes to (the )?financial statements|notes)[\s\S]*?$'
    }
    for name, pat in patterns.items():  # Iterate over each section pattern
        match = re.search(pat, text)  # Search for section in text
        if match:
            sections[name] = match.group(0)  # Add matched section text to dictionary
            logging.info(f'Section found: {name}')  # Log found section
    logging.info(f"Segmented sections: {list(sections.keys())}")  # Log found sections
    return sections  # Return sections dictionary

# Loads Q/A pairs from a JSON file
# Returns a list of Q/A dictionaries with 'question' and 'answer' keys

def load_qa_pairs_from_json(json_path: str) -> list:
    """
    Loads Q/A pairs from a JSON file.
    Args:
        json_path (str): Path to the JSON file containing Q/A pairs.
    Returns:
        List[Dict[str, str]]: List of Q/A dicts with 'question' and 'answer' keys.
    """
    logging.info(f'load_qa_pairs_from_json called with json_path: {json_path}')  # Log function call
    try:
        with open(json_path, 'r', encoding='utf-8') as f:  # Open JSON file
            qa_pairs = json.load(f)  # Load Q/A pairs from file
        logging.info(f"Loaded {len(qa_pairs)} Q/A pairs from JSON.")  # Log number of pairs loaded
        return qa_pairs  # Return loaded Q/A pairs
    except Exception as e:
        logging.error(f"Error loading Q/A pairs from JSON: {e}")  # Log error if loading fails
        return []  # Return empty list on error
