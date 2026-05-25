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
        os.makedirs('./data', exist_ok=True)

        # Initialize BLEURT scorer
        self.bleurt_checkpoint = "/home/BLEURT-20"
        self.bleurt_scorer = bleurt_score.BleurtScorer(self.bleurt_checkpoint)

        # Initialize output CSV with header
        with open(self.output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter='|')
            writer.writerow(['question', 'reference_answer', 'predicted_answer', 'rougeL_F1',
                           'cosine_similarity', 'bertscore_F1', 'bleurt_score', 'execution_time_seconds'])

    async def call_llm_api(self, question, context):
        """API call matching exact curl format for vLLM OpenAI-compatible endpoint"""
        url = "http://0.0.0.0:5001/v1/chat/completions"
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }

        user_content = f"{question}\n\nContext: {context}"
        payload = {
            "model": "meta/llama-3.3-70b-instruct",  # Matches your curl model name
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
            "top_p": 1,
            "n": 1,
            "max_tokens": 1000,
            "stream": False,
            "frequency_penalty": 0.0,  # Changed from 1.0 to avoid repetition
            "stop": None
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']

    def rouge_l(self, ref, cand):
        return self.rouge_scorer.score(ref, cand)['rougeL'].fmeasure

    def cosine_sim(self, ref, cand):
        emb1 = self.st_model.encode(ref, convert_to_tensor=True)
        emb2 = self.st_model.encode(cand, convert_to_tensor=True)
        return util.pytorch_cos_sim(emb1, emb2).item()

    def bertscore_f1(self, ref, cand):
        _, _, F1 = bert_score([cand], [ref], model_type='roberta-large', lang='en',
                             verbose=False, device='cpu', rescale_with_baseline=True)
        return float(F1[0])

    def bleurt_score(self, ref, cand):
        scores = self.bleurt_scorer.score(references=[ref], candidates=[cand])
        return float(scores[0]) if scores else 0.0

    async def evaluate(self, num_records=None):
        """Process CSV records and evaluate LLM responses"""
        with open(self.file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter='|')
            for i, row in enumerate(reader):
                try:
                    if i == 0:  # Skip header
                        continue
                    if num_records and i >= num_records:
                        break

                    question, reference_answer, context = row[0].strip(), row[1].strip(), row[2].strip()

                    print(f"Processing record {i}: {question[:60]}...")

                    start_time = time.time()
                    predicted_answer = await self.call_llm_api(question, context)
                    end_time = time.time()
                    exec_time = end_time - start_time

                    # Calculate metrics
                    rouge = self.rouge_l(reference_answer, predicted_answer)
                    cosine = self.cosine_sim(reference_answer, predicted_answer)
                    bert_f1 = self.bertscore_f1(reference_answer, predicted_answer)
                    bleurt = self.bleurt_score(reference_answer, predicted_answer)

                    # Save results
                    self.append_to_csv([question, reference_answer, predicted_answer,
                                      rouge, cosine, bert_f1, bleurt, exec_time])
                    print(f"✅ Saved record {i} (ROUGE-L: {rouge:.3f}, Time: {exec_time:.2f}s)")

                except Exception as e:
                    print(f"❌ Error processing record {i}: {e}")
                    continue

    def append_to_csv(self, row):
        """Append single row to output CSV"""
        with open(self.output_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter='|')
            writer.writerow(row)

    def calculate_averages_from_output(self):
        """Calculate average metrics from output CSV"""
        sums = {'rouge': 0.0, 'cosine': 0.0, 'bert': 0.0, 'bleurt': 0.0, 'time': 0.0}
        count = 0

        with open(self.output_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter='|')
            next(reader)  # Skip header

            for row in reader:
                if len(row) < 8:
                    continue
                try:
                    sums['rouge'] += float(row[3])
                    sums['cosine'] += float(row[4])
                    sums['bert'] += float(row[5])
                    sums['bleurt'] += float(row[6])
                    sums['time'] += float(row[7])
                    count += 1
                except ValueError:
                    continue

        if count == 0:
            print("No valid records found for averaging.")
            return None

        avgs = {k: v/count for k, v in sums.items()}
        print("\n📊 FINAL AVERAGES (%d records):" % count)
        print(f"   ROUGE-L F1:        {avgs['rouge']:.4f}")
        print(f"   Cosine Similarity: {avgs['cosine']:.4f}")
        print(f"   BERTScore F1:      {avgs['bert']:.4f}")
        print(f"   BLEURT Score:      {avgs['bleurt']:.4f}")
        print(f"   Avg Time (s):      {avgs['time']:.4f}")

        return avgs

if __name__ == "__main__":
    file_path = "data/qa_text_test.csv"  # question|reference_answer|context
    num_records = 200

    evaluator = LLMAccuracyEvaluator(file_path)
    asyncio.run(evaluator.evaluate(num_records=num_records))
    evaluator.calculate_averages_from_output()
