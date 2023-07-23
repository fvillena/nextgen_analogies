#! /opt/conda/bin/python
# -*- coding: utf-8 -*-

from transformers import pipeline
import pandas as pd
#import defaultdict module  
import argparse
from pathlib import Path
from tqdm import tqdm
import json
from nextgen_analogies import preprocess, is_word

parser = argparse.ArgumentParser(description='Solve analogies using a language model.')
parser.add_argument('--model', type=str, help='language model to use')
parser.add_argument('--analogies_file', type=Path, help='path to analogies file', default=Path("data/interim/analogies.json"))
parser.add_argument('--predictions_file', type=Path, help='path to predictions file')
parser.add_argument('-k', type=int, help='k most probable words to consider', default=20)
parser.add_argument('--analogy', type=str, help='analogy to solve', default=None)
args = parser.parse_args()

with open(args.analogies_file, "r") as f:
    analogies = json.load(f)

if args.analogy:
    analogies = {args.analogy: analogies[args.analogy]}

if "bert" in args.model:
    pipe = pipeline("fill-mask", model=args.model, device=0)
elif "llama" in args.model:
    pipe = pipeline("text-generation", model=args.model, device=0)

K = args.k
def predict_k_words(analogy,K):
    if "bert" in args.model:
        predictions = pipe(f"{analogy['question']} <mask>.", top_k=K)
        predicted_words = [preprocess(prediction["token_str"]) for prediction in predictions if is_word(prediction["token_str"])][:K]
    elif "llama" in args.model:
        generated_texts = pipe(
            analogy['question'], 
            do_sample=False,
            return_full_text=False,
            clean_up_tokenization_spaces=True,
            max_new_tokens=5,
            num_beams = K,
            num_return_sequences = K, 
        )
        first_word = lambda x: x.strip().split()[0]
        words = [first_word(x["generated_text"]) for x in generated_texts]
        unique_words = []
        for word in words:
            if (not word in unique_words) & (is_word(word)):
                unique_words.append(word)
        predicted_words = unique_words
    return predicted_words

for rela, current_analogies in tqdm(analogies.items()):
    for analogy in tqdm(current_analogies, leave=False):
        i = 2
        predicted_words = []
        while len(predicted_words) < K:
            predicted_words = predict_k_words(analogy, K*i)
            i += 1
        analogy["predicted_words"] = predicted_words

with open(args.predictions_file, "w", encoding="utf-8") as f:
    json.dump(analogies, f, indent=4, ensure_ascii=False)