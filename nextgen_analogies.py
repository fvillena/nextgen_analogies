import string

def preprocess(sentence):
    return sentence.strip().lower()

def is_word(word):
    return not all(c.isdigit() or c in string.punctuation for c in word)

def is_in(true_word, predicted_words):
    result = False
    for word in predicted_words:
        if true_word in word:
            result = True
            break
    return result

def accuracy_at_k(true, predicted, k):
    result = []
    for i in range(len(predicted)):
        if is_in(true[i], predicted[i][:k]):
            result.append(1)
        else:
            result.append(0)
    return sum(result)/len(result)