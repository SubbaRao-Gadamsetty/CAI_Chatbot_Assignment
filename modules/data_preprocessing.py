import os  # For file and directory operations
import json  # For loading Q/A pairs from JSON files
from logger_setup import logger
import pytesseract  # For OCR on images
from pdf2image import convert_from_path  # For converting PDF pages to images
import PyPDF2  # For extracting text from PDF files
import re  # For regular expressions in text cleaning
from typing import List, Tuple, Dict  # For type annotations


logger.info('data_preprocessing module loaded.')  # Log module load

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
    logger.info(f'load_and_preprocess_documents called with file_paths: {file_paths}')  # Log function call
    docs = []  # List to store cleaned document texts
    sections = {}  # Dictionary to store sectioned texts
    for file_path in file_paths:  # Iterate over each file path
        text = extract_text_from_pdf(file_path)  # Extract text from PDF
        logger.info(f'Extracted text from {file_path}: {text[:100]}...')  # Log extracted text (preview)
        cleaned = clean_text(text)  # Clean the extracted text
        logger.info(f'Cleaned text from {file_path}: {cleaned[:100]}...')  # Log cleaned text (preview)
        docs.append(cleaned)  # Add cleaned text to docs list
        segs = segment_sections(cleaned)  # Segment cleaned text into sections
        logger.info(f'Segmented sections from {file_path}: {list(segs.keys())}')  # Log segmented sections
        sections.update(segs)  # Add sections to sections dict
    logger.info(f'Loaded and preprocessed {len(docs)} documents.')  # Log number of documents processed
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
    logger.info(f'extract_text_from_pdf called with file_path: {file_path}')  # Log function call
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
        logger.error(f"Error extracting text: {e}")  # Log error if extraction fails
    logger.info(f"Extracted {len(text)} characters from {file_path}")  # Log number of characters extracted
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
    logger.info(f'clean_text called with text: {text[:100]}...')  # Log function call
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
    # Log first 10 lines of cleaned text for inspection
    cleaned_lines = text.splitlines()
    preview = '\n'.join(cleaned_lines[:10])
    logger.info(f"First 10 lines of cleaned text:\n{preview}")
    logger.info("Text cleaned.")  # Log cleaning complete
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
    logger.info(f'segment_sections called with text: {text[:100]}...')  # Log function call
    sections = {}  # Dictionary to store sections
    patterns = {
        'income_statement': r'(?i)(income statement|statement of income|consolidated statements of operations|statements of earnings)[\s\S]*?(?=balance sheet|statement of financial position|cash flow|statement of cash flows|shareholders|comprehensive|notes|management discussion|risk factors|auditor|summary|$)',
        'balance_sheet': r'(?i)(balance sheet|statement of financial position|consolidated balance sheets)[\s\S]*?(?=income statement|statement of income|cash flow|statement of cash flows|shareholders|comprehensive|notes|management discussion|risk factors|auditor|summary|$)',
        'cash_flow': r'(?i)(cash flow|statement of cash flows|consolidated statements of cash flows)[\s\S]*?(?=income statement|balance sheet|statement of income|statement of financial position|shareholders|comprehensive|notes|management discussion|risk factors|auditor|summary|$)',
        'shareholders_equity': r'(?i)(statement of shareholders\' equity|statement of stockholders\' equity|consolidated statements of shareholders\' equity)[\s\S]*?(?=income statement|balance sheet|cash flow|comprehensive|notes|management discussion|risk factors|auditor|summary|$)',
        'comprehensive_income': r'(?i)(statement of comprehensive income|consolidated statements of comprehensive income)[\s\S]*?(?=income statement|balance sheet|cash flow|shareholders|notes|management discussion|risk factors|auditor|summary|$)',
        'notes': r'(?i)(notes to (the )?financial statements|notes)[\s\S]*?(?=management discussion|risk factors|auditor|summary|$)',
        'management_discussion': r'(?i)(management\'s discussion and analysis|md&a|management discussion)[\s\S]*?(?=income statement|balance sheet|cash flow|shareholders|comprehensive|notes|risk factors|auditor|summary|$)',
        'risk_factors': r'(?i)(risk factors|risks related to)[\s\S]*?(?=income statement|balance sheet|cash flow|shareholders|comprehensive|notes|management discussion|auditor|summary|$)',
        'auditor_report': r'(?i)(independent auditor\'s report|auditor\'s report|report of independent registered public accounting firm)[\s\S]*?(?=income statement|balance sheet|cash flow|shareholders|comprehensive|notes|management discussion|risk factors|summary|$)',
        'summary_of_operations': r'(?i)(summary of operations|highlights|financial highlights)[\s\S]*?(?=income statement|balance sheet|cash flow|shareholders|comprehensive|notes|management discussion|risk factors|auditor|$)',
        'liquidity_and_capital_resources': r'(?i)(liquidity and capital resources)[\s\S]*?(?=legal proceedings|critical accounting estimates|overview|technology and infrastructure|executive officers and directors|seasonality|business and industry risks|human capital|forward-looking statements|$)',
        'legal_proceedings': r'(?i)(legal proceedings)[\s\S]*?(?=liquidity and capital resources|critical accounting estimates|overview|technology and infrastructure|executive officers and directors|seasonality|business and industry risks|human capital|forward-looking statements|$)',
        'critical_accounting_estimates': r'(?i)(critical accounting estimates)[\s\S]*?(?=liquidity and capital resources|legal proceedings|overview|technology and infrastructure|executive officers and directors|seasonality|business and industry risks|human capital|forward-looking statements|$)',
        'overview': r'(?i)(overview)[\s\S]*?(?=liquidity and capital resources|legal proceedings|critical accounting estimates|technology and infrastructure|executive officers and directors|seasonality|business and industry risks|human capital|forward-looking statements|$)',
        'technology_and_infrastructure': r'(?i)(technology and infrastructure)[\s\S]*?(?=liquidity and capital resources|legal proceedings|critical accounting estimates|overview|executive officers and directors|seasonality|business and industry risks|human capital|forward-looking statements|$)',
        'executive_officers_and_directors': r'(?i)(executive officers and directors)[\s\S]*?(?=liquidity and capital resources|legal proceedings|critical accounting estimates|overview|technology and infrastructure|seasonality|business and industry risks|human capital|forward-looking statements|$)',
        'seasonality': r'(?i)(seasonality)[\s\S]*?(?=liquidity and capital resources|legal proceedings|critical accounting estimates|overview|technology and infrastructure|executive officers and directors|business and industry risks|human capital|forward-looking statements|$)',
        'business_and_industry_risks': r'(?i)(business and industry risks)[\s\S]*?(?=liquidity and capital resources|legal proceedings|critical accounting estimates|overview|technology and infrastructure|executive officers and directors|seasonality|human capital|forward-looking statements|$)',
        'human_capital': r'(?i)(human capital)[\s\S]*?(?=liquidity and capital resources|legal proceedings|critical accounting estimates|overview|technology and infrastructure|executive officers and directors|seasonality|business and industry risks|forward-looking statements|$)',
        'forward_looking_statements': r'(?i)(forward-looking statements)[\s\S]*?(?=liquidity and capital resources|legal proceedings|critical accounting estimates|overview|technology and infrastructure|executive officers and directors|seasonality|business and industry risks|human capital|$)'
    }
    for name, pat in patterns.items():  # Iterate over each section pattern
        match = re.search(pat, text)
        if match:
            sections[name] = match.group(0)
        logger.info(f'Section found: {name}')
    # Log up to 10 segments with their first 100 characters
    logger.info('--- Segmentation Preview (up to 10 sections) ---')
    for i, (seg_name, seg_text) in enumerate(sections.items()):
        if i >= 10:
            break
        preview = seg_text[:100].replace('\n', ' ')
        logger.info(f'Segment {i+1}: {seg_name} | Preview: {preview}')
    # Detect and log all section headings in the cleaned text
    heading_pattern = r'(?m)^([A-Z][A-Za-z\'\-\s&]+):?$'
    headings = set(re.findall(heading_pattern, text))
    #logger.info(f"Detected section headings in document: {sorted(headings)}")
    logger.info(f"Segmented sections: {list(sections.keys())}")
    return sections

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
    logger.info(f'load_qa_pairs_from_json called with json_path: {json_path}')  # Log function call
    try:
        with open(json_path, 'r', encoding='utf-8') as f:  # Open JSON file
            qa_pairs = json.load(f)  # Load Q/A pairs from file
        logger.info(f"Loaded {len(qa_pairs)} Q/A pairs from JSON.")  # Log number of pairs loaded
        return qa_pairs  # Return loaded Q/A pairs
    except Exception as e:
        logger.error(f"Error loading Q/A pairs from JSON: {e}")  # Log error if loading fails
        return []  # Return empty list on error
