import re
from bs4 import BeautifulSoup
import distance
from fuzzywuzzy import fuzz
import pickle
import numpy as np
from nltk.corpus import stopwords

cv = pickle.load(open('cv.pkl','rb'))

def test_common_words(q1,q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    return len(w1 & w2)

def test_total_words(q1,q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    return (len(w1) + len(w2))


def test_fetch_token_features(q1, q2):
    safe = 0.0001

    stop = set(stopwords.words('english'))

    tokfeat = [0.0] * 8

    # Converting the Sentence into Tokens:
    q1tokens = q1.split()
    q2tokens = q2.split()

    if len(q1tokens) == 0 or len(q2tokens) == 0:
        return tokfeat

    # Get the non-stopwords in Questions
    q1words = set([word for word in q1tokens if word not in stop])
    q2words = set([word for word in q2tokens if word not in stop])

    # Get the stopwords in Questions
    q1stops = set([word for word in q1tokens if word in stop])
    q2stops = set([word for word in q2tokens if word in stop])

    # Get the common non-stopwords from Question pair
    common_word_count = len(q1words.intersection(q2words))

    # Get the common stopwords from Question pair
    common_stop_count = len(q1stops.intersection(q2stops))

    # Get the common Tokens from Question pair
    common_token_count = len(set(q1tokens).intersection(set(q2tokens)))

    tokfeat[0] = common_word_count / (min(len(q1words), len(q2words)) + safe)
    tokfeat[1] = common_word_count / (max(len(q1words), len(q2words)) + safe)
    tokfeat[2] = common_stop_count / (min(len(q1stops), len(q2stops)) + safe)
    tokfeat[3] = common_stop_count / (max(len(q1stops), len(q2stops)) + safe)
    tokfeat[4] = common_token_count / (min(len(q1tokens), len(q2tokens)) + safe)
    tokfeat[5] = common_token_count / (max(len(q1tokens), len(q2tokens)) + safe)

    # Last word of both question is same or not
    tokfeat[6] = int(q1tokens[-1] == q2tokens[-1])

    # First word of both question is same or not
    tokfeat[7] = int(q1tokens[0] == q2tokens[0])

    return tokfeat


def test_fetch_length_features(q1, q2):
    lenfeat = [0.0] * 3

    # Converting the Sentence into Tokens:
    q1tokens = q1.split()
    q2tokens = q2.split()

    if len(q1tokens) == 0 or len(q2tokens) == 0:
        return lenfeat

    # Absolute length features
    lenfeat[0] = abs(len(q1tokens) - len(q2tokens))

    # Average Token Length of both Questions
    lenfeat[1] = (len(q1tokens) + len(q2tokens)) / 2

    strs = list(distance.lcsubstrings(q1, q2))
    lenfeat[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)

    return lenfeat


def test_fetch_fuzzy_features(q1, q2):
    fuzfeat = [0.0] * 4

    # fuzz_ratio
    fuzfeat[0] = fuzz.QRatio(q1, q2)

    # fuzz_partial_ratio
    fuzfeat[1] = fuzz.partial_ratio(q1, q2)

    # token_sort_ratio
    fuzfeat[2] = fuzz.token_sort_ratio(q1, q2)

    # token_set_ratio
    fuzfeat[3] = fuzz.token_set_ratio(q1, q2)

    return fuzfeat


def preprocess(q):
    q = str(q).lower().strip()

    # Replace certain special characters with their string equivalents
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')

    # The pattern '[math]' appears around 900 times in the whole dataset.
    q = q.replace('[math]', '')

    # Replacing some numbers with string equivalents (not perfect, can be done better to account for more cases)
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    # Decontracting words
    # https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions
    # https://stackoverflow.com/a/19794953
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "can not",
        "can't've": "can not have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
    }

    q_decontracted = []

    for word in q.split():
        if word in contractions:
            word = contractions[word]

        q_decontracted.append(word)

    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")

    # Removing HTML tags
    q = BeautifulSoup(q)
    q = q.get_text()

    # Remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()

    return q


def query_point_creator(q1, q2):
    ip = []

    # preprocess
    q1 = preprocess(q1)
    q2 = preprocess(q2)

    # fetch basic features
    ip.append(len(q1))
    ip.append(len(q2))

    ip.append(len(q1.split(" ")))
    ip.append(len(q2.split(" ")))

    ip.append(test_common_words(q1, q2))
    ip.append(test_total_words(q1, q2))
    ip.append(round(test_common_words(q1, q2) / test_total_words(q1, q2), 2))

    # fetch token features
    tokfeat = test_fetch_token_features(q1, q2)
    ip.extend(tokfeat)

    # fetch length based features
    lenfeat = test_fetch_length_features(q1, q2)
    ip.extend(lenfeat)

    # fetch fuzzy features
    fuzfeat = test_fetch_fuzzy_features(q1, q2)
    ip.extend(fuzfeat)

    # bow feature for q1
    q1bow = cv.transform([q1]).toarray()

    # bow feature for q2
    q2bow = cv.transform([q2]).toarray()

    return np.hstack((np.array(ip).reshape(1, 22), q1bow, q2bow))