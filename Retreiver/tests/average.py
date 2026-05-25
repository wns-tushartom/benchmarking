import csv
import statistics

def calculate_averages(csv_file='data/output.csv'):
    """Calculate averages for numeric columns from row 1 (skip header)"""
    rouge_scores = []
    cosine_scores = []
    bert_scores = []
    bleurt_scores = []
    exec_times = []

    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='|')
        next(reader)  # Skip header row

        for i, row in enumerate(reader, 1):
            if len(row) < 8:
                continue

            try:
                # Columns: [0]question [1]ref [2]pred [3]rouge [4]cosine [5]bert [6]bleurt [7]time
                rouge_scores.append(float(row[3]))
                cosine_scores.append(float(row[4]))
                bert_scores.append(float(row[5]))
                bleurt_scores.append(float(row[6]))
                exec_times.append(float(row[7]))

            except ValueError:
                print(f"Skipping invalid row {i}")
                continue

    # Calculate and print averages
    print(f"📊 AVERAGES (from {len(rouge_scores)} valid rows):")
    print(f"   ROUGE-L F1:        {statistics.mean(rouge_scores):.4f}")
    print(f"   Cosine Similarity: {statistics.mean(cosine_scores):.4f}")
    print(f"   BERTScore F1:      {statistics.mean(bert_scores):.4f}")
    print(f"   BLEURT Score:      {statistics.mean(bleurt_scores):.4f}")
    print(f"   Exec Time (s):     {statistics.mean(exec_times):.4f}")

    return {
        'rougeL_F1': statistics.mean(rouge_scores),
        'cosine_similarity': statistics.mean(cosine_scores),
        'bertscore_F1': statistics.mean(bert_scores),
        'bleurt_score': statistics.mean(bleurt_scores),
        'execution_time_seconds': statistics.mean(exec_times)
    }

if __name__ == "__main__":
    averages = calculate_averages('data/output.csv')
