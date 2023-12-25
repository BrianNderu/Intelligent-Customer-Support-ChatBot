# Intelligent-Customer-Support-ChatBot

## Work Flow
At first, I installed the necessary modules i.e hugging face, langchain, chromadb, sentence transformers etc necessary for the project.

I then loaded the LLM which i used the mistraial7B model, and set configurations such as temperature to handle hallucinations

I then loaded the training data which was a CSV using the CSVLoader from langchain

I then split the data into chunks then Embedded the data. I then created a database from ChromaDB and used Vectordb and saved the embeddings as a pickle file.

I then built a retriever where I set k=3 which was well effective as a RAG technique

I then compiled the Question answer chain to give responses from the model

I then built a function to display the source page of where the response is built. 

I then built a streamlit interface for chatting with the model
