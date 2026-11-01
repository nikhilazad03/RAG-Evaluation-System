import json
import os
import time
import asyncio
from dotenv import load_dotenv

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams
from openai import OpenAI, AsyncOpenAI, RateLimitError

load_dotenv()

RESULTS_FILE = "results_1.json"
OUTPUT_FILE = "evaluated_results_1.json"
JUDGE_MODEL = "openai/gpt-oss-20b"

# Custom DeepEval LLM Judge wrapper for Groq
class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model_name=JUDGE_MODEL):
        self.model_name = model_name
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
        )
        self.async_client = AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        for attempt in range(5):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                return response.choices[0].message.content
            except RateLimitError:
                time.sleep(4)
        raise RuntimeError("Exceeded rate limit retries")

    async def a_generate(self, prompt: str) -> str:
        for attempt in range(5):
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                return response.choices[0].message.content
            except RateLimitError:
                await asyncio.sleep(4)
        raise RuntimeError("Exceeded rate limit retries")

    def get_model_name(self):
        return self.model_name


# Initialize judge
judge = GroqJudge()

# Initialize 4 standard RAG evaluation metrics
answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model=judge, async_mode=False)
faithfulness = FaithfulnessMetric(threshold=0.7, model=judge, async_mode=False)
context_relevancy = ContextualRelevancyMetric(threshold=0.7, model=judge, async_mode=False)

correctness = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
    threshold=0.7,
    model=judge,
    async_mode=False,
)

metrics = [
    ("answer_relevancy", answer_relevancy),
    ("faithfulness", faithfulness),
    ("contextual_relevancy", context_relevancy),
    ("correctness", correctness),
]


def run_all_evaluations():
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    # Slice to evaluate only the top 3 test cases
    results = all_results[:3]

    evaluated_records = []
    total = len(results)
    print(f"Evaluating top {total} test cases across 4 metrics...\n")

    for i, item in enumerate(results, start=1):
        print(f"[{i}/{total}] Evaluating: {item['question']}")

        tc = LLMTestCase(
            input=item["question"],
            actual_output=item["actual_answer"],
            expected_output=item["expected_answer"],
            retrieval_context=item["context"] if isinstance(item["context"], list) else [item["context"]],
        )

        record = dict(item)
        for metric_name, metric_instance in metrics:
            metric_instance.measure(tc)
            record[metric_name] = metric_instance.score
            print(f"   -> {metric_name}: {metric_instance.score}")
            time.sleep(1)

        evaluated_records.append(record)
        print()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(evaluated_records, f, indent=4, ensure_ascii=False)

    print(f"Done! Evaluated results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_all_evaluations()