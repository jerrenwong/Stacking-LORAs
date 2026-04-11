"""Translate English responses to Spanish using Llama 3.1 8B via OpenRouter."""

import argparse
import json
import os
import time

from openai import OpenAI
from datasets import load_dataset

SYSTEM_PROMPT = "You are a translator. Translate the following text to Spanish. Output only the translation, nothing else."


def generate_spanish_responses(rows, output_path="data/spanish_responses.json", batch_size=20):
    """Translate English responses to Spanish using OpenRouter API."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key and os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("OPENROUTER_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip('\u201c\u201d')

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Resume from existing file
    existing = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        print(f"Resuming from {len(existing)} existing responses")

    done_instructions = {e["instruction"] for e in existing}
    remaining = [r for r in rows if r["instruction"] not in done_instructions]
    print(f"Translating {len(remaining)} responses ({len(existing)} already done)")

    results = list(existing)
    for i, row in enumerate(remaining):
        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-3.1-8b-instruct",
                max_tokens=512,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row["output"]},
                ],
            )
            spanish_text = response.choices[0].message.content
        except Exception as e:
            print(f"  Error on item {i}: {e}")
            spanish_text = ""
            time.sleep(2)

        results.append({
            "instruction": row["instruction"],
            "input": row["input"],
            "output_es": spanish_text,
        })

        if (i + 1) % batch_size == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  Saved {len(results)} translations ({i + 1}/{len(remaining)})")

    with open(output_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Done. Saved {len(results)} Spanish translations to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Translate English responses to Spanish via Llama 3.1 8B")
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--output", type=str, default="data/spanish_responses.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    ds = load_dataset("silk-road/alpaca-data-gpt4-chinese", split="train")
    ds = ds.shuffle(seed=args.seed)
    rows = [{"instruction": r["instruction"], "input": r["input"], "output": r["output"]}
            for r in ds.select(range(args.n))]

    generate_spanish_responses(rows, output_path=args.output)


if __name__ == "__main__":
    main()
