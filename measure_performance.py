#! /opt/conda/bin/python
# -*- coding: utf-8 -*-

from nextgen_analogies import accuracy_at_k, mrr
from pathlib import Path
import argparse
import json
from collections import defaultdict

parser = argparse.ArgumentParser(description='Evaluate analogies predictions.')
parser.add_argument('--predictions_file', type=Path, help='path to predictions file')
parser.add_argument('--output_file', type=Path, help='path to output file')
args = parser.parse_args()

with open(args.predictions_file, "r") as f:
    predictions = json.load(f)

accuracies = defaultdict(list)
mrrs = {}
for rela, current_analogies in predictions.items():
    true = [analogy["answer"] for analogy in current_analogies]
    predicted = [analogy["predicted_words"] for analogy in current_analogies]
    for k in range(1, len(predicted[0])+1):
        accuracies[rela].append(accuracy_at_k(true, predicted, k))
    mrrs[rela] = mrr(true, predicted)

report = {
    "accuracy_at_k": accuracies,
    "mrr": mrrs
}

with open(args.output_file, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4, ensure_ascii=False)