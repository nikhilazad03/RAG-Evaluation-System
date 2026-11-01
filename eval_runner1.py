import sys
import os
import json

# Ensure Python can resolve modules inside RAG1
sys.path.append(os.path.join(os.path.dirname(__file__), "RAG1"))

from agent import ask_question

DATASET_PATH = "eval_dataset.json"
OUTPUT_PATH = "results_1.json"

def run_evaluation():
    # Read line-delimited JSON (JSONL) dataset
    test_cases = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_cases.append(json.loads(line))

    results = []
    print(f"Loaded {len(test_cases)} evaluation questions.\nRunning RAG Pipeline...")

    for i, test_case in enumerate(test_cases, start=1):
        question = test_case["question"]
        expected_answer = test_case["expected_answer"]

        print(f"[{i}/{len(test_cases)}] {question}")

        # Query the RAG agent
        rag_output = ask_question(question)
        actual_answer = rag_output["answer"]
        context = rag_output["context"]

        results.append({
            "question": question,
            "expected_answer": expected_answer,
            "actual_answer": actual_answer,
            "context": context
        })

    # Save compiled outputs to JSON
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nDone! Saved {len(results)} outputs to {OUTPUT_PATH}")

if __name__ == "__main__":
    run_evaluation()