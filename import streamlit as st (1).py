import streamlit as st
import openai
import faiss
import numpy as np
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# Set up OpenAI API Key
openai.api_key = 'YOUR_OPENAI_API_KEY'

# Step 1: Initialize Streamlit interface
st.title("Equity Research News Research Tool")

# Input section for the URLs of news articles
urls = st.text_area("Enter URLs of news articles (separate with commas)", 
                    "https://example1.com/article1, https://example2.com/article2")

# Function to load and process URLs
def load_documents(url_list):
    loader = UnstructuredURLLoader(urls=url_list)
    documents = loader.load()
    return documents

# Function to embed documents and store them in FAISS
def embed_documents(documents):
    embeddings = OpenAIEmbeddings()
    embedded_docs = [embeddings.embed_document(doc.page_content) for doc in documents]
    
    # Initialize FAISS Index
    dim = len(embedded_docs[0])
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(np.array(embedded_docs))
    
    return faiss_index, embeddings

# Function for similarity search
def search_documents(query, faiss_index, embeddings, k=3):
    query_embedding = embeddings.embed_query(query)
    D, I = faiss_index.search(np.array([query_embedding]), k)
    return I, D

# Function to setup conversational chain for summarization and analysis
def setup_conversational_chain():
    chat_model = ChatOpenAI(temperature=0)
    conversation_chain = ConversationalChain(chat_model=chat_model)
    return conversation_chain

# Step 2: Process user input and perform search
if st.button("Process Articles"):
    if urls.strip() == "":
        st.error("Please enter valid URLs.")
    else:
        # Step 3: Load and process the documents
        url_list = [url.strip() for url in urls.split(',')]
        documents = load_documents(url_list)
        
        # Step 4: Embed documents and store them in FAISS index
        faiss_index, embeddings = embed_documents(documents)
        
        # Step 5: Query for summarization or analysis
        query = st.text_input("Enter a query to search or summarize the articles")
        
        if query:
            I, D = search_documents(query, faiss_index, embeddings)
            relevant_docs = [documents[i] for i in I[0]]
            st.subheader("Relevant Articles Found:")
            for doc in relevant_docs:
                st.write(f"- {doc.page_content[:200]}...")  # Display first 200 characters
            
            # Step 6: Use LangChain's ConversationalChain for deeper analysis
            conversation_chain = setup_conversational_chain()
            result = conversation_chain.run(input=query)
            
            # Display the chatbot's response
            st.subheader("Research Tool Response:")
            st.write(result)
            
            # Display the processed documents for the query
            st.write("Full Documents Retrieved:")
            for doc in relevant_docs:
                st.write(doc.page_content)

# Customization: Display instructions or explanation on how this tool works
st.markdown("""
This tool leverages **LangChain**, **OpenAI**, and **FAISS** to provide an interactive research assistant for equity analysts. 
It processes URLs containing news articles, extracts relevant content, and uses machine learning models to analyze and summarize the information.
Simply input URLs, enter a research query, and explore the responses!
""")