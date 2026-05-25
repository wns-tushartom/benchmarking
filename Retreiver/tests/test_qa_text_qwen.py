import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import csv
import httpx
import asyncio
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util
from bleurt import score as bleurt_score

class LLMAccuracyEvaluator:
    def __init__(self, file_path):
        self.file_path = file_path
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.st_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.output_path = './data/output.csv'
        #os.makedirs('./data', exist_ok=True)
        # Initialize BLEURT scorer with downloaded checkpoint folder path
        self.bleurt_checkpoint = "/home/BLEURT-20"  # you need to download and extract this
        self.bleurt_scorer = bleurt_score.BleurtScorer(self.bleurt_checkpoint)
        with open(self.output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter='|')
            writer.writerow(['question', 'reference_answer', 'predicted_answer', 'rougeL_F1', 'cosine_similarity', 'bertscore_F1', 'bleurt_score', 'execution_time_seconds'])

    async def call_llm_api(self, question, context):
        """API call with question + context format for your pipe-separated CSV"""
        url = "http://0.0.0.0:5001/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        user_content = f"{question}\n\nContext: {context}"
        payload = {
            "model": "llama3.3-70bn",
            #"model": "Qwen/Qwen3-30B-A3B",
            #"model": "/models/qwen3-30b-a3b",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful QA assistant. Answer questions based only on the provided context."
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            "max_tokens": 1000
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']

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
            for i, row in enumerate(reader):
                try:
                    if i == 0:  # Skip header if present
                        continue
                    if num_records is not None and i >= num_records:
                        break

                    # Handle your 3-column format: question|ground_truth|context
                    question, reference_answer, context = row[0], row[1], row[2]

                    start_time = time.time()
                    predicted_answer = await self.call_llm_api(question, context)
                    end_time = time.time()
                    exec_time = end_time - start_time

                    rouge = self.rouge_l(reference_answer, predicted_answer)
                    cosine = self.cosine_sim(reference_answer, predicted_answer)
                    bert_f1 = self.bertscore_f1(reference_answer, predicted_answer)
                    bleurt = self.bleurt_score(reference_answer, predicted_answer)

                    self.append_to_csv([question, reference_answer, predicted_answer, rouge, cosine, bert_f1, bleurt, exec_time])
                    print(f"Saved record {i} to ./data/output.csv (Execution time: {exec_time:.2f} seconds)")

                except Exception as e:
                    print(f"Error processing record {i}: {e}")
                    continue

    def append_to_csv(self, row):
        with open(self.output_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter='|')
            writer.writerow(row)

    def calculate_averages_from_output(self):
        sum_bert, sum_cosine, sum_rouge, sum_bleurt, sum_time = 0.0, 0.0, 0.0, 0.0, 0.0
        count = 0
        with open(self.output_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter='|')
            next(reader)  # skip header
            for row in reader:
                if len(row) < 8:
                    continue
                try:
                    rouge_l_f1 = float(row[3])
                    cosine_sim = float(row[4])
                    bert_score_f1 = float(row[5])
                    bleurt_score = float(row[6])
                    exec_time = float(row[7])
                except ValueError:
                    continue
                sum_rouge += rouge_l_f1
                sum_cosine += cosine_sim
                sum_bert += bert_score_f1
                sum_bleurt += bleurt_score
                sum_time += exec_time
                count += 1
        if count == 0:
            print("No data records found for averaging.")
            return None
        avg_rouge = sum_rouge / count
        avg_cosine = sum_cosine / count
        avg_bert = sum_bert / count
        avg_bleurt = sum_bleurt / count
        avg_time = sum_time / count
        print(f"Average rougeL_F1: {avg_rouge:.4f}")
        print(f"Average cosine_similarity: {avg_cosine:.4f}")
        print(f"Average bertscore_F1: {avg_bert:.4f}")
        print(f"Average bleurt_score: {avg_bleurt:.4f}")
        print(f"Average execution time (seconds): {avg_time:.4f}")
        return avg_rouge, avg_cosine, avg_bert, avg_bleurt, avg_time

if __name__ == "__main__":
    file_path = "data/qa_text_test.csv"  # Your pipe-separated file: question|ground_truth|context
    num_records = 200
    evaluator = LLMAccuracyEvaluator(file_path)
    asyncio.run(evaluator.evaluate(num_records=num_records))
    evaluator.calculate_averages_from_output()
