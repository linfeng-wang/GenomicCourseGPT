import os
import sys
import datetime
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment
load_dotenv(find_dotenv())
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_KEY = "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"

# LLM model selection
# llm_name = "gpt-4-turbo"  # or "gpt-3.5-turbo"
llm_name = "gpt-3.5-turbo"
print(f"- Using model: {llm_name}")

# Load markdown files as documents
print("> Loading course content...")
markdown_files = [
    './docs/introduction/assembly.md',
    './docs/introduction/intro-to-linux.md',
    './docs/introduction/variant-detection.md',
    './docs/introduction/mapping.md',
    './docs/other-omics/eqtl.md',
    './docs/other-omics/methylation.md',
    './docs/other-omics/ml.md',
    './docs/other-omics/tb-resistance.md',
    './docs/other-omics/transcriptomics.md',
    './docs/advanced/gwas.md',
    './docs/advanced/phylogenetics.md',
    './docs/advanced/third-generation-sequencing.md',
]
loaders = [UnstructuredMarkdownLoader(path) for path in markdown_files]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Create vector database
persist_directory = 'chroma/'
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
vectordb = Chroma.from_documents(splits, embedding=embedding, persist_directory=persist_directory)

# Setup LLM
llm = ChatOpenAI(model_name=llm_name, temperature=0.15, openai_api_key=OPENAI_API_KEY)

# Custom prompt template
prompt_template = """
You are a bioinformatics teaching assistant for a pathogen genomics workshop.
You have access to lecture notes from 13 distinct lectures.
Using only the provided context from these lecture notes, concisely answer the student's question below in no more than four sentences.
If the answer is not clear from the context provided, explicitly say you don't know and recommend the student ask Jody, the course instructor.

Context:
{context}

Student's Question: {question}

Concise Answer:
"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=prompt_template.strip())

# Setup conversational retrieval chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 4})
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
)

PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Chat loop in terminal
print(CYAN + "\nBioinformatics TA chatbot is ready! Type 'exit' to end the conversation." + RESET_COLOR)

while True:
    question = input("\nYou (type 'exit' to quit): ")
    if question.lower() in ["exit", "quit"]:
        print(CYAN + "Ending the session. Goodbye!" + RESET_COLOR)
        break
    response = qa.invoke({"question": question})
    answer = response['answer']
    print(NEON_GREEN + f"\nAssistant: {answer}" + RESET_COLOR)
