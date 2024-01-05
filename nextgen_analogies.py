import string


def preprocess(sentence):
    return sentence.strip().lower()


def is_word(word):
    return not all(c.isdigit() or c in string.punctuation for c in word)


def is_in(true_word, predicted_words):
    result = False
    for word in predicted_words:
        if true_word == word:
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
    return sum(result) / len(result)


def mrr(true, predicted):
    result = []
    for true_word, predicted_words in zip(true, predicted):
        current_result = 0
        for predicted_word in predicted_words:
            if true_word in predicted_word:
                current_result = 1 / (predicted_words.index(predicted_word) + 1)
                break
        result.append(current_result)
    return result
