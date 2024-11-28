# Extracting text from PDF

import fitz  # PyMuPDF

# Function to extract text from each page in the PDF
def extract_text_from_pdf(file_path):

    print("\nExtracting text from the file...")

    # Open the PDF file
    document = fitz.open(file_path)
    text_data = []

    # Iterate through each page
    for page_num in range(document.page_count):
        page = document[page_num]
        page_text = page.get_text()  # Extract text from page
        text_data.append(page_text)

    document.close()
    print("\nCompleted successfully!")
    return text_data


# Path to the PDF file
# file_path = 'files/Final_SAS 2023_Annual Report.pdf'
file_path = 'files/Agricult_data.pdf'
pdf_text = extract_text_from_pdf(file_path)


def display_extracted_text(pdf_text):
    print("\nSample of first few pages of extracted text:\n")

    # Check the first few pages to see the extracted text
    for i, page in enumerate(pdf_text[:3]):
        print(f"--- Page {i+1} ---")
        print(page[:500])  # Print first 500 characters for preview


display_extracted_text(pdf_text)



# Step 2: Text Preprocessing

import re
import nltk
from nltk.tokenize import sent_tokenize

# download nltk toketizer
nltk.download('punkt')

nltk.download('punkt_tab')

# Function to preprocess text
def preprocess_text(text_data):
    
    print("\nPreprocessing extracted text...")

    processed_text = []

    for page_text in text_data:
        # Remove any extraneous whitespace and newlines
        page_text = page_text.replace('\n', ' ').strip()

        # Remove unwanted characters like page numbers or table of contents markers
        page_text = re.sub(r'\bPage\s\d+\b', '', page_text)
        page_text = re.sub(r'[^a-zA-Z0-9\s.,]', '', page_text)

        # Convert text to lowercase
        page_text = page_text.lower()

        # Tokenize text into sentences
        sentences = sent_tokenize(page_text)

        # Store cleaned sentences
        processed_text.extend(sentences)

    print("\nCompleted successfully!")

    return processed_text


# Apply preprocessing to the extracted text
cleaned_text = preprocess_text(pdf_text)



# Function to display preprocessed text
def display_preprocessed_text(cleaned_text):

    print("\nSample of first few cleaned sentences:\n")

    # Display the first few cleaned sentences
    for i, sentence in enumerate(cleaned_text[:20]):
        print(f"Sentence {i+1}: {sentence}")


# display preprocessed text
display_preprocessed_text(cleaned_text)




# Step 3: Entity Extraction

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForTokenClassification.from_pretrained("orkg/orkgnlp-agri-ner")

# Create a pipeline for NER
nlp_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)


# Function to extract entities from the text
def extract_entities(text_data):
    
    print("\nExtracting entitites...")

    entities = []

    for sentence in text_data:

        # Process each sentence using spaCy's NLP pipeline
        # doc = nlp(sentence)
        ner_results = nlp_pipeline(sentence)

        # for ent in doc.ents:
        #     # Append each recognized entity and its label
        #     entities.append((ent.text, ent.label_))

        # Collect and format the recognized entities
        for result in ner_results:
            entities.append((result['word'], result['entity'], result['score']))

    print("\nCompleted successfully!")
    

    return entities


# apply entity extraction on the cleaned text
extracted_entities = extract_entities(cleaned_text)


# Function to display a sample of extracted entities
def display_extracted_entities(extracted_entities):

    print("\nSample of extracted entities:\n")

    for i, entity in enumerate(extracted_entities[:100]):
        print(f"Entity {i+1}: Text: '{entity[0]}', Label: {entity[1]}")


# Display a sample of extracted entities
display_extracted_entities(extracted_entities)



# Step 4: Relationship Extraction

