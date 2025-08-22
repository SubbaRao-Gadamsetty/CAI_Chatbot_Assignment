"""
Main entry point for Financial Chatbot Assignment.
Launches Streamlit UI to compare RAG and Fine-Tuned chatbot systems.
"""
import os
import streamlit as st
from modules.data_preprocessing import load_and_preprocess_documents
from modules.rag_system import RAGChatbot
from modules.finetune_system import FineTunedChatbot
from modules.evaluation import run_evaluation, display_results_table

st.set_page_config(page_title="Financial Chatbot Comparison", layout="wide")
st.title("Financial Chatbot: RAG vs Fine-Tuned Model")

# Load and preprocess financial statement PDFs
with st.spinner("Loading and preprocessing financial statements..."):
    docs, sections = load_and_preprocess_documents([
        "data/NASDAQ_AMZN_2023.pdf",
        "data/NASDAQ_AMZN_2024.pdf"
    ])

# Initialize chatbots
rag_bot = RAGChatbot(docs, sections)
ft_bot = FineTunedChatbot(docs, sections)

# Sidebar radio button for selecting chatbot method
method = st.sidebar.radio("Select Chatbot Method", ["RAG", "Fine-Tuned Model"])

# Text input for user to enter financial question
query = st.text_input("Enter your financial question:")

# If user has entered a query, process it
if query:
    if method == "RAG":
        response, confidence, time_taken = rag_bot.answer(query)
    else:
        response, confidence, time_taken = ft_bot.answer(query)
    # Display answer and details in UI
    st.markdown(f"**Answer:** {response}")
    st.markdown(f"**Confidence:** {confidence}")
    st.markdown(f"**Method:** {method}")
    st.markdown(f"**Response Time:** {time_taken:.2f} seconds")

# Button to run evaluation and comparison of both chatbots
if st.button("Run Evaluation & Comparison"):
    results = run_evaluation(rag_bot, ft_bot)
    display_results_table(results)
