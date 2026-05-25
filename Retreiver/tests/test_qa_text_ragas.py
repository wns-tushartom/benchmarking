import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_VERBOSE"] = "false"
import csv
import httpx
import asyncio
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import json
json.decoder.JSONDecoder.strict = False

from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util
from bleurt import score as bleurt_score

# ✅ FIXED: Pass metric FUNCTIONS directly to evaluate()
from ragas import evaluate
from ragas.metrics import answer_correctness, answer_relevancy
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from datasets import Dataset

NVIDIA_API_KEY = "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"

class LLMAccuracyEvaluator:
    def __init__(self, file_path):
        self.file_path = file_path
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.st_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.output_path = './data/output.csv'
        os.makedirs('./data', exist_ok=True)
        self.bleurt_checkpoint = "/home/BLEURT-20"
        self.bleurt_scorer = bleurt_score.BleurtScorer(self.bleurt_checkpoint)

        with open(self.output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter='|')
            writer.writerow(['question', 'reference_answer', 'predicted_answer', 'rougeL_F1', 'cosine_similarity', 'ragas_answer_correctness', 'ragas_answer_relevancy','bertscore_F1', 'bleurt_score', 'execution_time_seconds'])

    def evaluate_single_row(self, question, model_response, ground_truth, context):
        """✅ FIXED: Pass metric FUNCTION directly - NO instantiation needed"""
        ds = Dataset.from_dict({
            "question": [question],
            "answer": [model_response],
            "contexts": [[context]],
            "ground_truth": [ground_truth]
        })

        llm = ChatNVIDIA(
            model="nvidia/nemotron-3-nano-30b-a3b",
            api_key=NVIDIA_API_KEY,
            temperature=0.0,
            max_tokens=8192
        )

        embeddings = NVIDIAEmbeddings(
            model="nvidia/nv-embedqa-e5-v5",
            api_key=NVIDIA_API_KEY
        )

        # ✅ PERFECT: Pass metric FUNCTION directly to evaluate()
        result = evaluate(
            dataset=ds,
            metrics=[answer_correctness, answer_relevancy],  # FUNCTION, not object!
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False
        )

        scores_df = result.to_pandas()
        ragas_answer_correctness = float(scores_df['answer_correctness'].iloc[0]) if 'answer_correctness' in scores_df and pd.notna(scores_df['answer_correctness'].iloc[0]) else 0.0
        ragas_answer_relevancy = float(scores_df['answer_relevancy'].iloc[0]) if 'answer_relevancy' in scores_df and pd.notna(scores_df['answer_relevancy'].iloc[0]) else 0.0
        return ragas_answer_correctness, ragas_answer_relevancy

    async def call_llm_api(self, question, context):
        url = "http://0.0.0.0:5001/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        user_content = f"{question}\n\nContext: {context}"
        payload = {
            #"model": "/models/qwen3-30b-a3b",
            #"model" : "Qwen/Qwen3-30B-A3B",
            "model" : "meta/llama-3.3-70b-instruct",
            "messages": [
                {"role": "system", "content": "You are a helpful QA assistant. Answer questions based only on the provided context."},
                {"role": "user", "content": user_content}
            ],
            "max_tokens": 1000,
            "temperature": 0.0
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content'].strip()
            except:
                return "API call failed"

    def rouge_l(self, ref, cand):
        return self.rouge_scorer.score(ref, cand)['rougeL'].fmeasure

    def cosine_sim(self, ref, cand):
        emb1 = self.st_model.encode(ref, convert_to_tensor=True)
        emb2 = self.st_model.encode(cand, convert_to_tensor=True)
        return util.pytorch_cos_sim(emb1, emb2).item()

    def bertscore_f1(self, ref, cand):
        _, _, F1 = bert_score([cand], [ref], model_type='roberta-large', lang='en', verbose=False, device='cpu', rescale_with_baseline=True)
        return float(F1[0])

    def bleurt_score(self, ref, cand):
        scores = self.bleurt_scorer.score(references=[ref], candidates=[cand])
        return float(scores[0]) if scores else 0.0

    async def evaluate(self, num_records=None):
        with open(self.file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter='|')
            next(reader, None)

            for i, row in enumerate(reader):
                if num_records is not None and i >= num_records:
                    break

                if len(row) < 3:
                    print(f"Skipping invalid row {i+1}")
                    continue

                question, reference_answer, context = row[0], row[1], row[2]
                print(f"Processing record {i+1}: {question[:50]}...")

                start_time = time.time()
                predicted_answer = await self.call_llm_api(question, context)
                ragas_answer_correctness, ragas_answer_relevancy = self.evaluate_single_row(question, predicted_answer, reference_answer, context)
                end_time = time.time()
                exec_time = end_time - start_time

                rouge = self.rouge_l(reference_answer, predicted_answer)
                cosine = self.cosine_sim(reference_answer, predicted_answer)
                bert_f1 = self.bertscore_f1(reference_answer, predicted_answer)
                bleurt = self.bleurt_score(reference_answer, predicted_answer)

                self.append_to_csv([question, reference_answer, predicted_answer, rouge, cosine, ragas_answer_correctness, ragas_answer_relevancy, bert_f1, bleurt, exec_time])
                print(f"✅ Saved record {i+1} (Time: {exec_time:.2f}s)")

        print(f"\n🎉 Evaluation complete! Results: {self.output_path}")

    def append_to_csv(self, row):
        with open(self.output_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter='|')
            writer.writerow(row)

    def calculate_averages_from_output(self):
        try:
            df = pd.read_csv(self.output_path, sep='|')
            if not df.empty:
                avg_metrics = df[['rougeL_F1', 'cosine_similarity', 'ragas_answer_correctness', 'ragas_answer_relevancy', 'bertscore_F1', 'bleurt_score', 'execution_time_seconds']].mean()
                print("\n📊 FINAL AVERAGE METRICS:")
                print(f"ROUGE-L F1:        {avg_metrics['rougeL_F1']:.4f}")
                print(f"Cosine Similarity: {avg_metrics['cosine_similarity']:.4f}")
                print(f"RAGAS Answer Correctness:       {avg_metrics['ragas_answer_correctness']:.4f}")
                print(f"RAGAS Answer Relevancy:       {avg_metrics['ragas_answer_relevancy']:.4f}")
                print(f"BERTScore F1:      {avg_metrics['bertscore_F1']:.4f}")
                print(f"BLEURT Score:      {avg_metrics['bleurt_score']:.4f}")
                print(f"Exec Time (s):     {avg_metrics['execution_time_seconds']:.4f}")
        except Exception as e:
            print(f"Error in averages: {e}")

if __name__ == "__main__":
    file_path = "data/qa_text_test.csv"
    num_records = 50
    evaluator = LLMAccuracyEvaluator(file_path)
    asyncio.run(evaluator.evaluate(num_records=num_records))
    evaluator.calculate_averages_from_output()
