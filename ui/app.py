import streamlit as st
import requests

st.set_page_config(page_title="RAG UI", layout="centered")

st.title("Ask your JSON files 📄")

query = st.text_input("Your question")

if query:
    response = requests.post("http://localhost:8080/qa/invoke", json={"input": query})
    st.write(response.json()["output"])
