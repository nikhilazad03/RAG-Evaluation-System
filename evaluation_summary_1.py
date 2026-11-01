import json

# --- Config: Update these for each experiment version ---
EVALUATED_FILE = "evaluated_results_1.json"
EXPERIMENT_OUTPUT = "experiment_v3.jsonl"
EXPERIMENT_NAME = "experiment_v3_k2"
RETRIEVER_K = 2
THRESHOLD = 0.7

def summarize_evaluation():
    with open(EVALUATED_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_cases = len(data)
    if total_cases == 0:
        print("No evaluation records found.")
        return

    metrics = ["answer_relevancy", "faithfulness", "contextual_relevancy", "correctness"]
    
    averages = {}
    pass_counts = {m: 0 for m in metrics}

    for m in metrics:
        scores = [item.get(m, 0.0) for item in data]
        averages[m] = round(sum(scores) / total_cases, 3)
        pass_counts[m] = sum(1 for s in scores if s >= THRESHOLD)

    pass_rates = {m: round((pass_counts[m] / total_cases) * 100, 1) for m in metrics}

    # Dictionary containing the run summary
    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "retriever_k": RETRIEVER_K,
        "total_test_cases": total_cases,
        "answer_relevancy": averages["answer_relevancy"],
        "faithfulness": averages["faithfulness"],
        "context_relevancy": averages["contextual_relevancy"],
        "correctness": averages["correctness"],
        "pass_rates": pass_rates,
    }

    print(f"\n================== EVALUATION SUMMARY ({EXPERIMENT_NAME}) ==================")
    print(f"Total Cases Evaluated: {total_cases} | Threshold: {THRESHOLD}")
    print("-" * 65)
    for m in metrics:
        print(f"{m.replace('_', ' ').title():<22}: Avg = {averages[m]:<5} | Pass Rate = {pass_rates[m]}%")
    print("=================================================================\n")

    with open(EXPERIMENT_OUTPUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"Saved run summary to {EXPERIMENT_OUTPUT}")

if __name__ == "__main__":
    summarize_evaluation()