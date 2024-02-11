import pdfplumber
import io
import logging

from vecotrdb import vectordb
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(filename='laodfiles.log', level=logging.INFO)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def file_uploader(st, st_document):
    st.write("File has been uploaded!")
    bytes_data = st_document.getvalue()
    with pdfplumber.open(io.BytesIO(bytes_data)) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    
    # separators = ["\n\n", "\n", "(?<=\, )", " ", ""]
    # pattern = '|'.join(map(re.escape, separators))

    # split_text = re.split(pattern, text)
    # separator = ' '  # Define the separator you want to use, e.g., a space
    # my_string = separator.join(split_text)
    texts = text_splitter.split_text(text)
    logging.info(f"Upload Text: {texts}")
    vectordb.add_texts(texts)

    logging.info(f"Successful add text")
    
    

    # add_to_collection(text)

    # for segment in split_text:
    #     if segment:  # Check if segment is not empty
            # add_to_collection(segment)
            # print(f"Segment: {segment}")
        # add_to_collection(segment)  # Assuming this is a function you've defined
        # print(f"Segment: {segment}")
    
    # add_to_collection(text)
    # reader = PyPDF2.PdfReader(st_document)
    # print(len(reader.pages))
    # page = reader.pages[0]
    # text = page.extract_text()
    # print(f"Page: {page}")
    