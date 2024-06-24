import asyncio
import streamlit as st
from langserve import RemoteRunnable

# 定义服务器地址和端点
server_url = "http://localhost:8000/rag-chain"
rag_chain = RemoteRunnable(server_url)

st.title("Youtube 视频总结")

question = st.text_input("Enter your question: ", value="")
print(f"Q:{question}\n",type(question))

#同步调用
def sync_invoke(question):
    return rag_chain.invoke(question)

#异步调用
async def async_invoke(question):
    return await rag_chain.ainvoke(question)

#流式输出
async def async_stream(question):
    responses = []
    async for msg in rag_chain.astream(question):
        responses.append(msg)
    return responses

# 添加按钮进行同步调用
if st.button("同步调用"):
    with st.spinner("等待服务器响应..."):
        response = sync_invoke(question)
        st.write("服务器响应：")
        st.write(response)

# 添加按钮进行异步调用
if st.button("异步调用"):
    with st.spinner("等待服务器响应..."):
        response = asyncio.run(async_invoke(question))
        st.write("服务器异步响应：")
        st.write(response)

# 添加按钮进行流式输出
if st.button("流式输出"):
    with st.spinner("等待服务器响应..."):
        responses = asyncio.run(async_stream(question))
        st.write("服务器流式响应：")
        for msg in responses:
            st.write(msg)