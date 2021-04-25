import pickle
import regex as re
import itertools
from utils import extract_phrases, gen_ngrams
from tqdm import tqdm
from hparams import NGRAM, MAXLEN


# load the data
data = pickle.load(open('./data/VNTC_data.pkl', 'rb'))
print(len(data), len(set(i for i in data)))

# extract Latin- characters only
alphabet = '^[ _abcdefghijklmnopqrstuvwxyz0123456789áàảãạâấầẩẫậăắằẳẵặóòỏõọôốồổỗộơớờởỡợéèẻẽẹêếềểễệúùủũụưứừửữựíìỉĩịýỳỷỹỵđ!\"\',\-\.:;?_\(\)]+$'
training_data = []
for i in data:
    i = i.replace("\n", ".")
    sentences = i.split(".")
    for j in sentences:
        if len(j.split()) > 2 and re.match(alphabet, j.lower()):
            training_data.append(j)
print(len(training_data))


phrases = itertools.chain.from_iterable(extract_phrases(text) for text in training_data)
phrases = [p.strip() for p in phrases if len(p.split()) > 1]

print(len(phrases))
print(phrases[-10:])

list_ngrams = []
for p in tqdm(phrases):
  if not re.match(alphabet, p.lower()):
    continue
  for ngr in gen_ngrams(p, NGRAM):
    if len(" ".join(ngr)) < MAXLEN:
      list_ngrams.append(" ".join(ngr))

del phrases
list_ngrams = list((list_ngrams))
print(len(list_ngrams))