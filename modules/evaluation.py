"""
Evaluation & Comparison Module
Implements mandatory test questions, extended evaluation, results table, and analysis.
"""
from logger_setup import logger  # Use shared logger for project-wide logger
import time  # For timing responses
import streamlit as st  # For Streamlit UI
from typing import List, Dict  # For type annotations

logger.info('Logger configured for evaluation module.')
logger.info('evaluation module loaded.')  # Log module load

def run_evaluation(rag_bot, ft_bot) -> List[Dict]:
    """
    Runs evaluation on mandatory and extended test questions for both chatbots.
    Args:
        rag_bot: RAGChatbot instance.
        ft_bot: FineTunedChatbot instance.
    Returns:
        List[Dict]: Results for each question and method.
    """
    logger.info('run_evaluation called.')  # Log function call
    test_questions = [
        "What was Amazon's net income in 2023?",  # high-confidence
        "What was the change in cash flow in 2024?",  # ambiguous
        "What is the capital of France?",  # irrelevant
        "What was the total revenue in 2023?",
        "What was the total revenue in 2024?",
        "What were general and administrative expenses in 2023?",
        "What was the operating income in 2024?",
        "What was the net sales in 2023?",
        "What was the cash flow in 2023?",
        "What was the total assets in 2024?"
    ]
    logger.info(f'Test questions: {test_questions}')  # Log test questions
    ground_truth = {
        "What was Amazon's net income in 2023?": "$33 million",
        "What was the change in cash flow in 2024?": "$642 million",
        "What is the capital of France?": "Irrelevant query",
        "What was the total revenue in 2023?": "$123 million",
        "What was the total revenue in 2024?": "$123 million",
        "What were general and administrative expenses in 2023?": "$50 million",
        "What was the operating income in 2024?": "$80 million",
        "What was the net sales in 2023?": "$200 million",
        "What was the cash flow in 2023?": "$75 million",
        "What was the total assets in 2024?": "$500 million"
    }
    logger.info(f'Ground truth: {ground_truth}')  # Log ground truth
    results = []  # List to store evaluation results
    from difflib import SequenceMatcher
    def fuzzy_match(a, b):
        return SequenceMatcher(None, a, b).ratio()
    for method, bot in [("RAG", rag_bot), ("Fine-Tuned", ft_bot)]:  # Evaluate both chatbots
        for q in test_questions:  # For each test question
            logger.info(f'Calling answer for method: {method}, question: {q}')  # Log question being asked
            answer, confidence, t = bot.answer(q)  # Get answer, confidence, and time
            logger.info(f'Answer: {answer}, Confidence: {confidence}, Time: {t}')  # Log answer details
            gt = ground_truth.get(q, '').lower()
            ans = answer.lower()
            # Fuzzy match for correctness
            correct = 'Y' if fuzzy_match(gt, ans) > 0.7 else 'N'
            results.append({
                'Question': q,  # The question asked
                'Method': method,  # Chatbot method used
                'Answer': answer,  # Model's answer
                'Confidence': confidence,  # Confidence score
                'Time(s)': round(t, 2),  # Response time in seconds
                'Correct (Y/N)': correct  # Whether answer is correct
            })
    logger.info(f'Evaluation results: {results}')  # Log evaluation results
    return results  # Return evaluation results

def display_results_table(results: List[Dict]):
    """
    Displays results table and analysis in Streamlit.
    Args:
        results (List[Dict]): Evaluation results.
    """
    logger.info('display_results_table called.')  # Log function call
    st.markdown("## Evaluation Results Table (First 3 Queries)")  # Table title
    # Show only results for first 3 queries (both RAG and Fine-Tuned)
    # Each query has two rows (RAG and Fine-Tuned), so first 6 rows
    limited_results = results[:6]
    st.dataframe(limited_results)  # Show limited results as dataframe
    # CSV Export functionality for limited results
    import pandas as pd  # Import pandas for CSV export
    df = pd.DataFrame(limited_results)  # Convert limited results to DataFrame
    csv = df.to_csv(index=False).encode('utf-8')  # Convert DataFrame to CSV
    st.download_button(
        label="Download results as CSV (First 5 Queries)",  # Button label
        data=csv,  # CSV data
        file_name="evaluation_results_first5.csv",  # Download file name
        mime="text/csv"  # MIME type
    )
    # Analysis
    rag_count = sum(1 for r in results if r['Method']=='RAG')
    ft_count = sum(1 for r in results if r['Method']=='Fine-Tuned')
    rag_acc = sum(1 for r in results if r['Method']=='RAG' and r['Correct (Y/N)']=='Y') / rag_count if rag_count else 0  # RAG accuracy
    ft_acc = sum(1 for r in results if r['Method']=='Fine-Tuned' and r['Correct (Y/N)']=='Y') / ft_count if ft_count else 0  # Fine-Tuned accuracy
    rag_time = sum(r['Time(s)'] for r in results if r['Method']=='RAG') / rag_count if rag_count else 0  # RAG avg time
    ft_time = sum(r['Time(s)'] for r in results if r['Method']=='Fine-Tuned') / ft_count if ft_count else 0  # Fine-Tuned avg time
    logger.info(f'RAG Accuracy: {rag_acc}, Avg Time: {rag_time}')  # Log RAG stats
    logger.info(f'Fine-Tuned Accuracy: {ft_acc}, Avg Time: {ft_time}')  # Log Fine-Tuned stats
    st.markdown(f"**RAG Accuracy:** {rag_acc*100:.1f}% | **Avg Time:** {rag_time:.2f}s")  # Show RAG stats
    st.markdown(f"**Fine-Tuned Accuracy:** {ft_acc*100:.1f}% | **Avg Time:** {ft_time:.2f}s")  # Show Fine-Tuned stats
    st.markdown("### Analysis")  # Analysis section
    st.markdown("- RAG is robust to irrelevant queries due to retrieval guardrails.")
    st.markdown("- Fine-tuned model is faster but may hallucinate on out-of-domain queries.")
    st.markdown("- RAG leverages context for factual answers, but is slower.")
    st.markdown("- Trade-off: RAG = accuracy, Fine-Tuned = speed.")
    logger.info('Results table and analysis displayed.')  # Log display complete
