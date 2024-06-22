from langchain import hub
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import  ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the UR OpenAI API Key
load_dotenv("")

# Load Youtube URL that You want to analysis
# Using Youtubeloader, you need
# 1. pip install youtube-transcript-api
# 2. pip install pytube
# Once you install both of them, you are good to go :)
url = "https://www.youtube.com/watch?v=wd7TZ4w1mSw&list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x&index=1"
loader = YoutubeLoader.from_youtube_url(url, add_video_info = True)
docs = loader.load()
print("You pick video content has been loaded..\n")

# We need use text-split to split the docs
text_spliter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
spliter = text_spliter.split_documents(docs)
print("You video content has been splitted.... \n")

#After splitted, we need embed content that has been splitted well
# and store embedding content into a vector database
# Also you need install
# pip install faiss-cpu
embedding = OpenAIEmbeddings(model = "text-embedding-3-small")
vectorstore = FAISS.from_documents(documents=spliter, embedding=embedding)

print("You video content has been embedded and stored into a vector database...... \n")


#Add Retriever to find the info in vectordata
retriever = vectorstore.as_retriever()

#Using a format prompt and install
# pip install langchainhub
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#Select LLM you would like & Declare the output_parser
llm = ChatOpenAI()
parser = StrOutputParser()

rag_chain = (
    {"context":retriever|format_docs, "question":RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

question = "这个视频讲了什么？"
print(f"Questio:{question}\n")

print("Answer: ")
for chunk in rag_chain.stream(question):
    print(chunk,end="", flush=True)

