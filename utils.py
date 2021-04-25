import numpy as np
from symbol import *
import regex as re
from unidecode import unidecode
from hparams import *
from nltk import ngrams


# function for adding mistake( noise)
def teen_code(sentence, pivot):
    random = np.random.uniform(0, 1, 1)[0]
    new_sentence = str(sentence)
    if random > pivot:
        for word in acronym.keys():
            if re.search(word, new_sentence):
                random2 = np.random.uniform(0, 1, 1)[0]
                if random2 < 0.5:
                    new_sentence = new_sentence.replace(word, acronym[word])
        for word in teen.keys():
            if re.search(word, new_sentence):
                random3 = np.random.uniform(0, 1, 1)[0]
                if random3 < 0.05:
                    new_sentence = new_sentence.replace(word, teen[word])
        return new_sentence
    else:
        return sentence


def add_noise(sentence, pivot1, pivot2):
    sentence = teen_code(sentence, 0.5)
    noisy_sentence = ""
    i = 0
    while i < len(sentence):
        if sentence[i] not in letters:
            noisy_sentence += sentence[i]
        else:
            random = np.random.uniform(0, 1, 1)[0]
            if random < pivot1:
                noisy_sentence += (sentence[i])
            elif random < pivot2:
                if sentence[i] in typo.keys() and sentence[i] in region.keys():
                    random2 = np.random.uniform(0, 1, 1)[0]
                    if random2 <= 0.4:
                        noisy_sentence += typo[sentence[i]]
                    elif random2 < 0.8:
                        noisy_sentence += region[sentence[i]]
                    elif random2 < 0.95:
                        noisy_sentence += unidecode(sentence[i])
                    else:
                        noisy_sentence += sentence[i]
                elif sentence[i] in typo.keys():
                    random3 = np.random.uniform(0, 1, 1)[0]
                    if random3 <= 0.6:
                        noisy_sentence += typo[sentence[i]]
                    elif random3 < 0.9:
                        noisy_sentence += unidecode(sentence[i])
                    else:
                        noisy_sentence += sentence[i]
                elif sentence[i] in region.keys():
                    random4 = np.random.uniform(0, 1, 1)[0]
                    if random4 <= 0.6:
                        noisy_sentence += region[sentence[i]]
                    elif random4 < 0.85:
                        noisy_sentence += unidecode(sentence[i])
                    else:
                        noisy_sentence += sentence[i]
                elif i < len(sentence) - 1:
                    if sentence[i] in region2.keys() and (i == 0 or sentence[i - 1] not in letters) and sentence[
                        i + 1] in vowel:
                        random5 = np.random.uniform(0, 1, 1)[0]
                        if random5 <= 0.9:
                            noisy_sentence += region2[sentence[i]]
                        else:
                            noisy_sentence += sentence[i]
                    else:
                        noisy_sentence += sentence[i]

            else:
                new_random = np.random.uniform(0, 1, 1)[0]
                if new_random <= 0.33:
                    if i == (len(sentence) - 1):
                        continue
                    else:
                        noisy_sentence += (sentence[i + 1])
                        noisy_sentence += (sentence[i])
                        i += 1
                elif new_random <= 0.66:
                    random_letter = np.random.choice(letters2, 1)[0]
                    noisy_sentence += random_letter
                else:
                    pass

        i += 1
    return noisy_sentence


def extract_phrases(text):
    return re.findall(r'\w[\w ]+', text)


def gen_ngrams(words, n=5):
    return ngrams(words.split(), n)


# So a 5-grams contain at most 7*5 = 35 character (except one that has spell mistake)
# add "\x00" padding at the end of 5-grams in order to equal their length
def encoder_data(text, maxlen=MAXLEN):
    text = "\x00" + text
    x = np.zeros((maxlen, len(alphabet)))
    for i, c in enumerate(text[:maxlen]):
        x[i, alphabet.index(c)] = 1
    if i < maxlen - 1:
        for j in range(i + 1, maxlen):
            x[j, 0] = 1
    return x


def decoder_data(x):
    x = x.argmax(axis=-1)
    return ''.join(alphabet[i] for i in x)
