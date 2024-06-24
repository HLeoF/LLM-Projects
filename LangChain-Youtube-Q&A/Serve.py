from langchain import hub
from fastapi import FastAPI
from dotenv import load_dotenv
from langserve import add_routes
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv("D:/LLMs Projects/enviroment.env")

url = "https://www.youtube.com/watch?v=wd7TZ4w1mSw&list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x&index=1"
loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
docs = loader.load()

text_spliter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
spliter = text_spliter.split_documents(docs)

embedding = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(documents=spliter,embedding=embedding)

retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")

llm = ChatOpenAI()

parser = StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context":retriever|format_docs, "question":RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

# Add definition

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A API server using Langchain's Runnable interfaces",
)

# Add chain route
# need install
# pip install sse_starlette
add_routes(
    app,
    rag_chain,
    path="/rag-chain",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)