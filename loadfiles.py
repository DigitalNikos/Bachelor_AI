import pdfplumber
import io
import logging
# from helper_utils import word_wrap
from vecotrdb import collection
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from pypdf import PdfReader
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from vecotrdb import collection

logging.basicConfig(filename='laodfiles.log', level=logging.INFO)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def file_uploader(st, st_document):
    st.write("File has been uploaded!")
    reader = PdfReader(st_document)
    pdf_texts = [p.extract_text().strip() for p in reader.pages]
    
    # Filter the empty strings
    pdf_texts = [text for text in pdf_texts if text]

    # bytes_data = st_document.getvalue()
    # with pdfplumber.open(io.BytesIO(bytes_data)) as pdf:
    #     text = ''
    #     for page in pdf.pages:
    #         text += page.extract_text() + "\n"
    
    # separators = ["\n\n", "\n", "(?<=\, )", " ", ""]
    # pattern = '|'.join(map(re.escape, separators))

    # split_text = re.split(pattern, text)
    # separator = ' '  # Define the separator you want to use, e.g., a space
    # my_string = separator.join(split_text)
    # texts = text_splitter.split_text(text)
    logging.info(f"Upload Text1: {pdf_texts}")
    # logging.info(f"Upload Text: {word_wrap(pdf_texts[0])}")
    # vectordb.add_texts(texts)

    logging.info(f"Successful add text")

    character_splitter = RecursiveCharacterTextSplitter(
      separators=["\n\n", "\n", ". ", " ", ""],
      chunk_size=1000,
      chunk_overlap=0
    )

    character_split_texts = character_splitter.split_text('\n\n'.join(pdf_texts))

    logging.info(f"\nTotal chunks: {len(character_split_texts)}")

    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)

    token_split_texts = []
    for text in character_split_texts:
      token_split_texts += token_splitter.split_text(text)

    logging.info(f"\nTotal chunks1: {len(token_split_texts)}")

    embedding_function = SentenceTransformerEmbeddingFunction()
    # logging.info(embedding_function([token_split_texts[10]]))

    ids = [str(i) for i in range(len(token_split_texts))]

    logging.info(f"\nIDs: {ids}")

    collection.add(ids=ids, documents=token_split_texts)
    collection.count()

    logging.info(f"\ncollection.count(): {collection.count()}")
    logging.info(f"\ADD TO COLLECTION")
    

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
    