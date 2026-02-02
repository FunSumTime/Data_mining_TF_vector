from pypdf import PdfReader
import pickle

# creating a pdf reader object
text_data = {}
# It's best practice to open the file in binary mode ('rb') within a 'with' statement for proper resource management.
with open('5G.pdf', 'rb') as pdf_file:
    reader = PdfReader(pdf_file)

    # printing number of pages in pdf file
    # print(f"Number of pages: {len(reader.pages)}") # Use f-string for cleaner formatting

    # Initialize a variable to store all extracted text
    full_text = ""

    # iterating through each page to extract text
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text = page.extract_text()
        if text: # Check if text was successfully extracted (some PDFs might not have extractable text)
            full_text += text + "\n" # Append the text and a newline character for readability

    # Print the full extracted text
    # print("\nFull Extracted Text:")
    # print(full_text)
    text_data["5G"] = full_text
# print(text_data)

with open('6G.pdf', 'rb') as pdf_file:
    reader = PdfReader(pdf_file)
    
    full_text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text = page.extract_text()
        if text:
            full_text += text + "\n"
        
    text_data["6G"] = full_text

with open("Software-defined_networking.pdf", "rb") as pdf_file:
    reader = PdfReader(pdf_file)

    full_text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    
    text_data["SDN"] = full_text
print(text_data.keys())


with open("data.pkl", "wb") as f:
    pickle.dump(text_data,f)


