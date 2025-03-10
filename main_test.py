
#%%
import os
from openai import OpenAI
import openai
import sys
from dotenv import load_dotenv, find_dotenv
sys.path.append('../..')

# import panel as pn  # GUI
# pn.extension()
# Load the .env file so OPENAI_API_KEY is in your environment
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ.get("OPENAI_API_KEY")

OPENAI_API_KEY = "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"
#%%
import datetime
current_date = datetime.datetime.now().date()
# llm_name = "gpt-3.5-turbo"
llm_name = "gpt-4-turbo"
print(llm_name)

from langchain.document_loaders import UnstructuredMarkdownLoader
print("> Loading course content...")
# Load markdown documents
loaders = [
    # introduction
    UnstructuredMarkdownLoader('./docs/introduction/assembly.md'),
    UnstructuredMarkdownLoader('./docs/introduction/intro-to-linux.md'),
    UnstructuredMarkdownLoader('./docs/introduction/variant-detection.md'),
    UnstructuredMarkdownLoader('./docs/introduction/mapping.md'),
    # other-omics.
    UnstructuredMarkdownLoader('./docs/other-omics/eqtl.md'),
    UnstructuredMarkdownLoader('./docs/other-omics/methylation.md'),
    UnstructuredMarkdownLoader('./docs/other-omics/ml.md'),
    UnstructuredMarkdownLoader('./docs/other-omics/tb-resistance.md'),
    UnstructuredMarkdownLoader('./docs/other-omics/transcriptomics.md'),
    # advanced.
    UnstructuredMarkdownLoader('./docs/advanced/gwas.md'),
    UnstructuredMarkdownLoader('./docs/advanced/phylogenetics.md'),
    UnstructuredMarkdownLoader('./docs/advanced/third-generation-sequencing.md'),
]

docs = []
for loader in loaders:
    docs.extend(loader.load())

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1200,
    chunk_overlap = 200
)
splits = text_splitter.split_documents(docs)

# creating db
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
persist_directory = 'chroma/'
# embedding = OpenAIEmbeddings()
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

#%%
# question = "What are major topics for this class?"
# # docs = vectordb.similarity_search(question,k=4)
# docs = vectordb.max_marginal_relevance_search(question,k=4)


#%%
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name=llm_name, temperature=0.15, openai_api_key=OPENAI_API_KEY)
# llm.predict("Hello world!")
# Build prompt
from langchain.prompts import PromptTemplate

template = """
You are a bioinformatics teaching assistant for a pathogen genomics workshop. 
You have access to lecture notes from 13 distinct lectures. 
Using only the provided context from these lecture notes, concisely answer the student's question below in no more than four sentences.
If the answer is not clear from the context provided, explicitly say you don't know and recommend the student ask Jody, the course instructor.

Context:
{context}

Student's Question: {question}

Concise Answer:
"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template.strip())


# ==========Single question and answering================
# from langchain.chains import RetrievalQA
# question = "Is probability a class topic?"
# qa_chain = RetrievalQA.from_chain_type(llm,
#                                        retriever=vectordb.as_retriever(),
#                                        return_source_documents=True,
#                                        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})


# result = qa_chain({"query": question})
# result["result"]


# ==========Multi question and answering================
print('> Building virtual assistant...')
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

from langchain.chains import ConversationalRetrievalChain
# retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 4})
retriever = vectordb.as_retriever()

qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT})  # Change here
    # chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

#%%
question = "what is mapping?"
result = qa({"question": question})
print(result['answer'])



#%%      
question = "show me the code for trimmomatics then"
result = qa({"question": question})
print(result['answer'])
# %%
question = "explain each part of the code for me"
result = qa({"question": question})
print(result['answer'])
# %%
question = "how is the average quality calculated?"
result = qa({"question": question})
print(result['answer'])



# %%
question = "print all the topics in the course?"
result = qa({"question": question})
print(result['answer'])
# %%
question = "tell me about the genome assembly?"
result = qa({"question": question})
print(result['answer'])
# %%
question = "how does it work?"
result = qa({"question": question})
print(result['answer'])
# %%
question = "from which part of the workshop can I find relevant information regarding beast?"
result = qa({"question": question})
print(result['answer'])
# %%
