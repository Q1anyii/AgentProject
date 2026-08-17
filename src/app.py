import streamlit as st
from dotenv import load_dotenv
import os

# 加载环境变量，放最顶部
load_dotenv()

st.title("Agent对话演示")
st.markdown("基于DeepSeek + LangGraph")

# 输入框
user_input = st.text_area("请输入你的问题：")

if st.button("发送"):
    st.write(f"用户输入：{user_input}")
    st.write("模型回复：这里调用你的Agent逻辑")
