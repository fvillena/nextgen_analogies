#! /opt/conda/bin/python
# -*- coding: utf-8 -*-

from transformers import pipeline, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# import defaultdict module
import argparse
from pathlib import Path
from tqdm import tqdm
import json
from nextgen_analogies import preprocess, is_word

parser = argparse.ArgumentParser(description="Solve analogies using a language model.")
parser.add_argument("--model", type=str, help="language model to use")
parser.add_argument(
    "--analogies_file",
    type=Path,
    help="path to analogies file",
    default=Path("data/interim/analogies.json"),
)
parser.add_argument("--predictions_file", type=Path, help="path to predictions file")
parser.add_argument(
    "-k", type=int, help="k most probable words to consider", default=20
)
parser.add_argument("--analogy", type=str, help="analogy to solve", default=None)
# quantize argument
parser.add_argument(
    "--quantize",
    action="store_true",
    help="quantize model to 8-bit",
    default=False,
)
args = parser.parse_args()

with open(args.analogies_file, "r") as f:
    analogies = json.load(f)

if args.analogy:
    analogies = {args.analogy: analogies[args.analogy]}

model_kwargs = {
    "load_in_8bit": args.quantize,
    "device_map": "auto",
    # "max_memory": {0: "24GiB", 1: "0GiB"},
}
if "bert" in args.model:
    pipe = pipeline("fill-mask", model=args.model, device=0)
elif "llama" in args.model:
    pipe = pipeline("text-generation", model=args.model, model_kwargs=model_kwargs)

    class CustomStoppingCriteria(StoppingCriteria):
        def __init__(self):
            self.word_delimiter_count = 0

        def __call__(self, input_ids, score, **kwargs) -> bool:
            last_token = pipe.tokenizer.convert_ids_to_tokens(input_ids[0])[-1]
            word_delimiters = [
                ".",
                "?",
                "!",
                ".",
                ":",
                ";",
                "(",
                ")",
                "[",
                "]",
                "{",
                "}",
                '"',
                "'",
                " ",
                "\n",
                "▁",
                ",",
            ]
            for delimiter in word_delimiters:
                if delimiter in last_token:
                    self.word_delimiter_count += 1
            stop = self.word_delimiter_count >= 2
            if stop:
                self.word_delimiter_count = 0
            return stop

    stopping_criteria = StoppingCriteriaList([CustomStoppingCriteria()])
elif "aguila" in args.model:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pipe = pipeline(
        "text-generation",
        model=args.model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        model_kwargs=model_kwargs,
    )

K = args.k


def predict_k_words(analogies, K):
    if "bert" in args.model:
        questions = [f"{analogy['question']} <mask>." for analogy in analogies]
        predictions = []
        for prediction in pipe(questions, top_k=K * 2, batch_size=8):
            predicted_words = [
                preprocess(p["token_str"])
                for p in prediction
                if is_word(p["token_str"])
            ][:K]
            predictions.append(predicted_words)
    elif ("lama" in args.model) | ("aguila" in args.model):
        questions = [f"{analogy['question']}" for analogy in analogies]
        predictions = []
        for question in tqdm(questions, leave=False):
            try:
                generated_text = pipe(
                    question,
                    do_sample=False,
                    return_full_text=False,
                    clean_up_tokenization_spaces=True,
                    max_new_tokens=100,
                    stopping_criteria=stopping_criteria,
                    num_beams=K,
                    num_return_sequences=K,
                    temperature=1.0,
                    top_p=1.0,
                )
                texts = [preprocess(x["generated_text"]) for x in generated_text]
                predictions.append(texts)
            except IndexError:
                print(f"error in {question}")
                predictions.append([])
    return predictions


for rela, current_analogies in tqdm(analogies.items()):
    print(f"predicting {rela}")
    predicted_analogies = predict_k_words(current_analogies, K)
    for i, analogy in enumerate(current_analogies):
        analogy["predicted_words"] = predicted_analogies[i]

with open(args.predictions_file, "w", encoding="utf-8") as f:
    json.dump(analogies, f, indent=4, ensure_ascii=False)
