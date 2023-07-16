#! /opt/conda/bin/python
# -*- coding: utf-8 -*-

from transformers import pipeline
import pandas as pd
#import defaultdict module  
from collections import defaultdict
import string
import argparse
from pathlib import Path
from tqdm import tqdm
import json

parser = argparse.ArgumentParser(description='Solve analogies using a language model.')
parser.add_argument('--model', type=str, help='language model to use')
parser.add_argument('--analogies_file', type=Path, help='path to analogies file')
parser.add_argument('--results_file', type=Path, help='path to results file')
parser.add_argument('-k', type=int, help='k most probable words to consider')
args = parser.parse_args()

def preprocess(sentence):
    return sentence.strip().lower()

def is_word(word):
    return not all(c.isdigit() or c in string.punctuation for c in word)

def accuracy_at_k(true, predicted, k):
    result = []
    for i in range(len(predicted)):
        if true[i] in predicted[i][:k]:
            result.append(1)
        else:
            result.append(0)
    return sum(result)/len(result)

relations = pd.read_csv(args.analogies_file).applymap(preprocess)

analogies = defaultdict(list)
# iterate for every group in a groupby object
for name, group in relations.groupby("RELA"):
    #shuffle group dataframe
    group = group.sample(frac=1)
    #iterate every two rows over group dataframe   
    for i in range(0, len(group), 2):
        #get the first row
        row1 = group.iloc[i]
        row2 = group[group["STR.1"] != row1["STR.1"]].sample(1).iloc[0]
        question = "{} es a {} como {} es a".format(row1["STR"], row1["STR.1"], row2["STR"])
        answer = row2["STR.1"]
        analogies[name].append({"question": question, "answer": answer})

pipe = pipeline("fill-mask", model=args.model, device=0)

K = args.k
results = defaultdict(list)
for rela, current_analogies in tqdm(analogies.items()):
    for analogy in tqdm(current_analogies, leave=False):
        i = 2
        predicted_words = []
        while len(predicted_words) < K:
            predictions = pipe(f"{analogy['question']} <mask>.", top_k=i*K)
            predicted_words = [preprocess(prediction["token_str"]) for prediction in predictions if is_word(prediction["token_str"])][:K]
            i += 1
        results[rela].append(predicted_words)

accuracies = defaultdict(list)

for analogy, result in zip(analogies.items(), results.items()):
    true = [x["answer"] for x in analogy[1]]
    predicted = result[1]
    for k in range(1, K+1):
        accuracies[analogy[0]].append(accuracy_at_k(true, predicted, k))

with open(args.results_file, "w") as f:
    json.dump(accuracies, f, indent=4)