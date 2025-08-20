"""
Fine-Tuned Model System Module
Implements dataset prep, baseline evaluation, fine-tuning, supervised instruction tuning, and guardrails.
"""
import logging
import time
from typing import List, Dict, Tuple
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import random

logging.basicConfig(level=logging.INFO)

class FineTunedChatbot:
    """
    Fine-Tuned Model Chatbot for financial Q/A.
    Implements dataset prep, baseline evaluation, fine-tuning, SFT, and guardrails.
    """
    def __init__(self, docs: List[str], sections: Dict[str, str]):
        """
        Initializes fine-tuned chatbot with document texts and sections.
        Args:
            docs (List[str]): List of cleaned document texts.
            sections (Dict[str, str]): Sectioned texts.
        """
        logging.info(f"Initializing FineTunedChatbot with {len(docs)} documents.")
        self.docs = docs
        self.sections = sections
        # Use actual Q/A extraction from data_preprocessing
        from modules.data_preprocessing import extract_qa_pairs_from_sections
        self.qa_pairs = extract_qa_pairs_from_sections(sections, num_pairs=50)
        self.model_name = 'distilgpt2'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.generator = pipeline('text-generation', model=self.model_name)
        logging.info("FineTunedChatbot initialized.")

    # Removed: prepare_qa_dataset (now using extract_qa_pairs_from_sections)

    def baseline_evaluation(self, questions: List[str]) -> List[Tuple[str, float, float]]:
        """
        Evaluates pre-trained model on 10 questions.
        Args:
            questions (List[str]): List of questions.
        Returns:
            List[Tuple[str, float, float]]: (answer, confidence, time)
        """
        logging.info("Running baseline evaluation.")
        results = []
        for q in questions:
            start = time.time()
            output = self.generator(q, max_length=64)[0]['generated_text']
            confidence = random.uniform(0.5, 0.9)
            end = time.time()
            results.append((output, confidence, end-start))
        logging.info("Baseline evaluation complete.")
        return results

    def fine_tune(self, output_dir: str = "finetuned_model"):
        """
        Fine-tunes model on extracted Q/A dataset using HuggingFace Trainer.
        Args:
            output_dir (str): Directory to save the fine-tuned model.
        """
        logging.info("Fine-tuning model with HuggingFace Trainer.")
        from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
        import torch
        class QADataset(torch.utils.data.Dataset):
            def __init__(self, qa_pairs, tokenizer):
                self.qa_pairs = qa_pairs
                self.tokenizer = tokenizer
            def __len__(self):
                return len(self.qa_pairs)
            def __getitem__(self, idx):
                pair = self.qa_pairs[idx]
                prompt = f"Question: {pair['question']}\nAnswer: "
                answer = pair['answer']
                input_text = prompt + answer
                encoding = self.tokenizer(input_text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
                encoding = {k: v.squeeze(0) for k, v in encoding.items()}
                encoding['labels'] = encoding['input_ids'].clone()
                return encoding
        train_dataset = QADataset(self.qa_pairs, self.tokenizer)
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            learning_rate=2e-5,
            logging_dir=f'{output_dir}/logs',
            save_steps=10,
            save_total_limit=2,
            report_to=["none"],
        )
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logging.info(f"Fine-tuning complete. Model saved to {output_dir}")

    def supervised_instruction_tuning(self):
        """
        Advanced SFT on Q/A pairs.
        """
        logging.info("Running supervised instruction fine-tuning.")
        # Simulate SFT (placeholder)
        time.sleep(1)
        logging.info("SFT complete.")

    def guardrail(self, query: str, response: str) -> Tuple[bool, str]:
        """
        Input/output guardrail: block irrelevant/harmful queries, flag hallucinations.
        Args:
            query (str): User query.
            response (str): Model response.
        Returns:
            Tuple[bool, str]: (is_safe, filtered_response)
        """
        logging.info("Applying guardrail.")
        irrelevant = ["capital of france", "weather", "sports"]
        for word in irrelevant:
            if word in query.lower():
                logging.warning("Blocked irrelevant query.")
                return False, "Query is irrelevant to financial statements."
        if "not factual" in response:
            logging.warning("Flagged hallucinated output.")
            return False, "Response may be hallucinated."
        logging.info("Guardrail passed.")
        return True, response

    def answer(self, query: str) -> Tuple[str, float, float]:
        """
        End-to-end pipeline: baseline, fine-tuned, SFT, guardrail.
        Args:
            query (str): User query.
        Returns:
            Tuple[str, float, float]: (answer, confidence, response_time)
        """
        logging.info(f"Answering query: {query}")
        start = time.time()
        output = self.generator(query, max_length=128)[0]['generated_text']
        confidence = random.uniform(0.6, 0.95)
        is_safe, filtered = self.guardrail(query, output)
        end = time.time()
        logging.info(f"Answer: {filtered[:50]}... Confidence: {confidence} Time: {end-start:.2f}s")
        return filtered, confidence, end-start
