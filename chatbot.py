#%%
import os
import sys
import datetime
import openai
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("API key not found. Set the OPENAI_API_KEY environment variable.")

openai.api_key = OPENAI_API_KEY

#%%
# LLM model selection
# llm_name = "gpt-4-turbo"  # or "gpt-3.5-turbo"
llm_name = "gpt-3.5-turbo"
print(f"- Using model: {llm_name}")

# Load markdown files as documents
print("> Loading course content...")

from subprocess import run
run('git pull https://github.com/lshtm-genomics/omics-course.git',shell=True)



markdown_files = []
for root, dirs, files in os.walk("omics-course"):
    for file in files:
        if file.endswith(".md"):
            markdown_files.append(os.path.join(root, file))
            
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
Your name is LinBot2000.
You have access to lecture notes from 12 distinct lectures.
Using only the provided context from these lecture notes, with linebreaks after each sentence for readability, concisely answer the student's question below in no more than four sentences. 
But be specific when explaining code and errors.
If the answer is not clear from the context provided, explicitly say you don't know and recommend the student to ask a course instructor.
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
print(CYAN + "*"*20 + RESET_COLOR)
print(CYAN + "\nBioinformatics TA chatbot is ready! Type 'exit' to end the conversation." + RESET_COLOR)
print(CYAN + "*"*20 + RESET_COLOR)


while True:
    question = input("\nYou (type 'exit' to quit): ")
    if question.lower() in ["exit", "quit"]:
        print(CYAN + "Ending the session. Goodbye!" + RESET_COLOR)
        break
    response = qa.invoke({"question": question})
    answer = response['answer']
    print(NEON_GREEN + f"\nAssistant: {answer}" + RESET_COLOR)

# %%
