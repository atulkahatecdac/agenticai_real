# We may need to install these first:
# pip install langchain langchain-community pypdf langchain-text-splitters

# This reads the PDF file and creates one "Document" object 
# per page. It also captures metadata like the page number.
from langchain_community.document_loaders import PyPDFLoader

# This is the recommended splitter for generic text. 
# It tries to split by paragraphs, then sentences, 
# then words, until the chunks are small enough. 
# This prevents words or sentences from being cut in half 
# awkwardly.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load the PDF
loader = PyPDFLoader(r"C:\code\agenticai_realpage\module_2\Introduction_to_Insurance.pdf")
data = loader.load()

# 2. Split the PDF into smaller documents
# chunk_size is the number of characters, chunk_overlap keeps context between splits
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(data)

# 3. Display the results
print(f"Total documents created: {len(docs)}")
print("-" * 30)

print("FIRST DOCUMENT:")
print(docs[0].page_content)
print("-" * 30)

print("LAST DOCUMENT:")
print(docs[-1].page_content)