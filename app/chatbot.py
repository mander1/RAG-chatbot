from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()  # Loads variables from .env

llm = OpenAI(temperature=0)  # Uses the env var automatically

def load_qa_chain():
    embedding_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function=embedding_fn)

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = OpenAI(temperature=0)  # You can also use Hugging Face LLMs

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain