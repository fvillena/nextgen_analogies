#! /opt/conda/bin/python
# -*- coding: utf-8 -*-

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import operator
import re

# import defaultdict module
import argparse
from pathlib import Path
from tqdm import tqdm
import json
from nextgen_analogies import preprocess, is_word

dev = False

if not dev:
    parser = argparse.ArgumentParser(
        description="Solve analogies using a language model."
    )
    parser.add_argument("--model", type=str, help="language model to use")
    parser.add_argument(
        "--analogies_file",
        type=Path,
        help="path to analogies file",
        default=Path("data/interim/analogies.json"),
    )
    parser.add_argument(
        "--predictions_file", type=Path, help="path to predictions file"
    )
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
    parser.add_argument("--few_shot", type=int, help="few shot", default=0)
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
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if "biogpt" in args.model.lower():
        model = AutoModelForCausalLM.from_pretrained(
            args.model, pad_token_id=tokenizer.eos_token_id
        )
        model.cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, pad_token_id=tokenizer.eos_token_id, **model_kwargs
        )
    model_name = args.model
    K = args.k
    predictions_file = args.predictions_file
    few_shot = args.few_shot
else:
    model_kwargs = {
        "load_in_8bit": True,
        "device_map": "auto",
    }
    with open("data/interim/analogies_100.en.json", "r") as f:
        analogies = json.load(f)
    model_name = "microsoft/biogpt"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "biogpt" in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_name, pad_token_id=tokenizer.eos_token_id
        )
        model.cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, pad_token_id=tokenizer.eos_token_id, **model_kwargs
        )

    predictions_file = "microsoft--biogpt.json"
    K = 20
    few_shot = 0


def predict_k_words_gpt(sentence: str, k: int, max_new_tokens: int = 10) -> list:
    initial_sentence = sentence
    max_new_tokens = max_new_tokens
    num_beams = k

    dict_end = {}
    dict_aux = {initial_sentence: 1}

    tokens_begin = [" ", "\n", ".", ",", '"', ";", "_", "*", "?", "!", "\xa0", "▁", "Ġ"]
    first = True

    for k in range(max_new_tokens):
        dict_sentence = {}
        for s in dict_aux:
            encoded_text = tokenizer(s, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                outputs = model(**encoded_text)

            next_token_logits = outputs.logits[0, -1, :]
            next_token_probs = torch.softmax(next_token_logits, -1)
            topk_next_tokens = torch.topk(next_token_probs, num_beams)

            l = [
                (tokenizer.decode(idx), float(prob.numpy()))
                for idx, prob in zip(
                    topk_next_tokens.indices.cpu(), topk_next_tokens.values.cpu()
                )
            ]
            l_end = [e for e in l if ((e[0] == "") or (e[0][0] in tokens_begin))]
            l_subword = [e for e in l if ((e[0] != "") and (e[0][0] not in tokens_begin))]

            if first:
                for e in l:
                    dict_sentence[s + e[0]] = (
                        dict_aux[s] * e[1]
                    )  # *(l_end[0][1])#*((p_mean)**(max_new_tokens-k-1)) # revisar

            else:
                for e in l_subword:
                    dict_sentence[s + e[0]] = (
                        dict_aux[s] * e[1]
                    )  # si hay un token que no tiene espacio seguir con beam search

                if l_end != []:
                    sum_p = sum([e[1] for e in l_end])
                    dict_end[s + tokenizer.eos_token] = (
                        dict_aux[s] * sum_p
                    )  # *(l_end[0][1])#*((p_mean)**(max_new_tokens-k-1)) # revisar

        aux = dict(
            sorted(dict_sentence.items(), key=operator.itemgetter(1), reverse=True)
        )
        sdict = {A: N for (A, N) in [x for x in aux.items()][:num_beams]}
        dict_aux = sdict
        first = False

    for key, val in dict.items(dict_aux):
        dict_end[key] = val

    aux = dict(sorted(dict_end.items(), key=operator.itemgetter(1), reverse=True))
    words = []
    for s in aux.keys():
        if len(words) == num_beams:
            break
        try:
            word = re.search(r".* (\w+)(<\|endoftext\|>)?", s).group(1)
            words.append(word)
        except:
            pass
    return words


def predict_k_words_biogpt(sentence: str, k: int, max_new_tokens: int = 10) -> list:
    words = predict_k_words_gpt(sentence, k, max_new_tokens)
    return [word[2:] for word in words]


def predict_k_words_llama(sentence: str, k: int, max_new_tokens: int = 10) -> list:
    initial_sentence = sentence
    max_new_tokens = max_new_tokens
    num_beams = k

    dict_end = {}
    dict_aux = {initial_sentence: 1}

    tokens_begin = [" ", "\n", ".", ",", '"', ";", "_", "*", "?", "!", "\xa0", "▁", "Ġ"]
    first = True

    for k in range(max_new_tokens):
        dict_sentence = {}
        for s in dict_aux:
            encoded_text = tokenizer(s, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                outputs = model(**encoded_text)

            next_token_logits = outputs.logits[0, -1, :]
            next_token_probs = torch.softmax(next_token_logits, -1)
            topk_next_tokens = torch.topk(next_token_probs, num_beams)

            l = [
                (token.replace("▁", " "), float(prob.numpy()))
                for token, prob in zip(
                    tokenizer.convert_ids_to_tokens(topk_next_tokens.indices),
                    topk_next_tokens.values.cpu(),
                )
            ]
            l_end = [e for e in l if e[0][0] in tokens_begin]
            l_subword = [e for e in l if e[0][0] not in tokens_begin]

            if first:
                for e in l:
                    dict_sentence[s + e[0]] = (
                        dict_aux[s] * e[1]
                    )  # *(l_end[0][1])#*((p_mean)**(max_new_tokens-k-1)) # revisar

            else:
                for e in l_subword:
                    dict_sentence[s + e[0]] = (
                        dict_aux[s] * e[1]
                    )  # si hay un token que no tiene espacio seguir con beam search

                if l_end != []:
                    sum_p = sum([e[1] for e in l_end])
                    dict_end[s + tokenizer.eos_token] = (
                        dict_aux[s] * sum_p
                    )  # *(l_end[0][1])#*((p_mean)**(max_new_tokens-k-1)) # revisar

        aux = dict(
            sorted(dict_sentence.items(), key=operator.itemgetter(1), reverse=True)
        )
        sdict = {A: N for (A, N) in [x for x in aux.items()][:num_beams]}
        dict_aux = sdict
        first = False

    for key, val in dict.items(dict_aux):
        dict_end[key] = val

    aux = dict(sorted(dict_end.items(), key=operator.itemgetter(1), reverse=True))
    words = []
    for s in aux.keys():
        if len(words) == num_beams:
            break
        try:
            word = re.search(r".* (\w+)", s).group(1)
            if word != "to":
                words.append(word)
        except:
            pass
    return words


def get_k_shot_sentence(analogies, current_analogy, shots):
    sentence = ""
    i = 0
    answers = set()
    for analogy in analogies:
        if analogy["question"] == current_analogy["question"]:
            continue
        if i == shots:
            break
        if analogy["answer"] in answers:
            continue
        sentence += f'{analogy["question"]} {analogy["answer"]}, '
        answers.add(analogy["answer"])
        i += 1
    return f"{sentence}{current_analogy['question']}"


for rela, current_analogies in tqdm(analogies.items()):
    print(f"predicting {rela}")
    for current_analogy in tqdm(current_analogies):
        if few_shot > 0:
            sentence = get_k_shot_sentence(current_analogies, current_analogy, few_shot)
        else:
            sentence = current_analogy["question"]
        if ("biogpt" in model_name.lower()) | ("biomistral" in model_name.lower()):
            current_analogy["predicted_words"] = predict_k_words_biogpt(sentence, K)
        elif ("llama" in model_name.lower()) | ("meditron" in model_name.lower()):
            current_analogy["predicted_words"] = predict_k_words_llama(sentence, K)
        else:
            current_analogy["predicted_words"] = predict_k_words_gpt(sentence, K)

with open(predictions_file, "w", encoding="utf-8") as f:
    json.dump(analogies, f, indent=4, ensure_ascii=False)
