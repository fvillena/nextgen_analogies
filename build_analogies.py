#! /opt/conda/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
from nextgen_analogies import preprocess
from pathlib import Path
from collections import defaultdict
import json

parser = argparse.ArgumentParser(description='Build analogies.')
parser.add_argument('--relations_file', type=Path, help='path to relations file')
parser.add_argument('--output_file' , type=Path, help='path to output file', default=Path("data/interim/analogies.json"))
args = parser.parse_args()

relations = pd.read_csv(args.relations_file).applymap(preprocess)

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

with open(args.output_file, "w", encoding="utf-8") as f:
    json.dump(analogies, f, indent=4, ensure_ascii=False)