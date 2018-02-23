import sklearn

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

import numpy as np

#import matplotlib.pyplot as plt

from tqdm import tqdm
from random import shuffle

import gensim

from gensim.models import KeyedVectors

import nltk

from nltk.corpus import wordnet as wn

nltk.download('wordnet')



#get infos in wordnet
def get_wornet_synonyms_hypernyms(noun, words_filter=None):
    """Return le list of hypernyms and hyponyms of a word from wordvec.
    
    :param noun: the noun to find some synonyms
    :param words_filter: a list, if not None, return only words contained in this list

    :type noun: str
    :type words_filter: list(str)

    :return: a list of synonyms 
    :rtype: list(str)
    """
    try:
        hypernyms = wn.synsets(noun)[0].hypernyms()[0].lemma_names() #NB: on ne prend que le premier
    except:
        hypernyms = []
    try:
        hyponyms = wn.synsets(noun)[0].hyponyms()[0].lemma_names() #NB: on ne prend que le premier
    except:
        hyponyms = []
    if words_filter:
        out = []
        for w in hypernyms + hyponyms:
            if w in words_filter:
                out.append(w)
        return out
    return hypernyms + hyponyms

def get_wornet_synonyms(noun, words_filter=None):
    """Return le list of synonyms of a word from wordvec.
    
    :param noun: the noun to find some synonyms
    :param words_filter: a list, if not None, return only words contained in this list

    :type noun: str
    :type words_filter: list(str)

    :return: a list of synonyms 
    :rtype: list(str)
    """
    try:
        w = []
        for sn in wn.synsets(noun): 
            #TODO: check if the synset is a nouns synset and not verbs 
            #DONE: si on passe les mots qui sont des nomns seulemnt dans words_filter
            w.extend(sn.lemma_names())
    except:
        w = []
    w = list(set(w))
    return list(filter(lambda s: s in words_filter, w)) if words_filter else w



def similar_word2vec_words(word_vectors, noun, words_filter=None, n=10):
    """Return a list of the `n` most similar words from word2vec.
    
    :param word_vectors: the gensim word2vec model
    :param noun: the noun to find some similar words
    :param words_filter: a list, if not None, return only words contained in this list
    :param n: only return this number of words (max)

    :type word_vectors: gensim.models.keyedvectors.Word2VecKeyedVector
    :type noun: str
    :type words_filter: list(str)
    :type n: int
    
    :return: a list of similar meaning words
    :rtype: list(str)
    """
    tmp = list(list(zip(*word_vectors.similar_by_word(noun, n)))[0])
    if not words_filter:
        return tmp
    return list(filter(lambda s: s in words_filter, tmp))

def get_positive_and_negatives_synonyms_pairs(word_vectors, noun, words_filter=None, n=20):
    """Return 2 lists containing the synonyms int the first, 
    and similiars words in the second.
    
    :param word_vectors: the gensim word2vec model
    :param noun: the noun to find some similar words
    :param words_filter: a list, if not None, return only words contained in this list
    :param n: only return this number of words (max)
    
    :type word_vectors: gensim.models.keyedvectors.Word2VecKeyedVector
    :type noun: str
    :type words_filter: list(str)
    :type n: int
    
    :return: a tuple containg 2 lists: ([synonyms], [similars])
    :rtype: tuple(list(str), list(str))
    """
    synonyms = get_wornet_synonyms(noun, words_filter=words_filter)
    similars = similar_word2vec_words(word_vectors, noun, words_filter=words_filter)
    not_synonyms = list(filter(lambda s: not s in synonyms, similars))
    return synonyms, not_synonyms

#list all word2vec nouns exploitables
def is_exploitable(s):
    """not accept urls, names, patterns and weird stuffs... 
    AND is in wordnet database !!"""
    if s != s.lower():
        return False
    for c in '._+#/@':
        if c in s:
            return False
    #simple heuristic: remove all words ending with 's' to get only singular nouns
    if s[-1] == 's':
        return False
    if s.isalpha():
        return wn.synsets(s) != [] #check if is in wordnnet databse



print("* load word2vec pretrained from google")
word_vectors = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

print("* load nouns")

print(" - list all wordnet nouns")
wordnet_nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}

print(" - build a list of exploitable words")
words = list(filter(is_exploitable, word_vectors.index2word))
words = list(filter(lambda s: s in wordnet_nouns, words))

#example
synonyms, not_synonyms = get_positive_and_negatives_synonyms_pairs(word_vectors, "dog", words)


print("* build a dataset of synonyms / not synonyms")
positives = []
negatives = []

allready_seen_words = [] #allow us to remove some simple duplicates

shuffle(words)

nb_max_words = len(words)

print(" - build pairs")
for word in tqdm(words[:nb_max_words]):
    if word in allready_seen_words:
        continue
    allready_seen_words.append(word)
    pos, neg = get_positive_and_negatives_synonyms_pairs(word_vectors, word, words)
    for p in pos:
        positives.append((word, p))
    for n in neg:
        negatives.append((word, n))

np.save("positives", np.array(positives))
np.save("negatives", np.array(negativess))

#del words, wordnet_nouns
#del allready_seen_words

shuffle(positives)
shuffle(negatives)

print(" - compute embedings")
positive_embedings = np.array(list(map(
    lambda c: [word_vectors.word_vec(c[0]), word_vectors.word_vec(c[1])], 
    positives
)))

negative_embedings = np.array(list(map(
    lambda c: [word_vectors.word_vec(c[0]), word_vectors.word_vec(c[1])], 
    negatives
)))

np.save("positive_embedings", positive_embedings)
np.save("negative_embedings", negative_embedings)

nb_max_rows = min(len(positives), len(negatives))

positive_train = positive_embedings[:int(nb_max_rows * .8)]
positive_test  = positive_embedings[int(nb_max_rows * .8):nb_max_rows]

negative_train = negative_embedings[:int(nb_max_rows * .8)]
negative_test  = negative_embedings[int(nb_max_rows * .8):nb_max_rows]

#del positives, negatives

print(" - split dataset in training/test sets")
x_train = np.array(list(positive_train) + list(negative_train))
y_train = np.array([0] * len(positive_train) + [1] * len(negative_train)) #0 means synonym

x_test = np.array(list(positive_test) + list(negative_test))
y_test = np.array([0] * len(positive_test) + [1] * len(negative_test))

print(" - augment the dataset by adding the symetry")
#adding a symetry to the dataset let us to augment data for free 
#and remove some noise (the anchor words is not allways on the same side...)
build_symetry = lambda x_set : np.array([x_set[:, 0, :], x_set[:, 1, :]]).swapaxes(0, 1)

assert (build_symetry(build_symetry(x_train)) == x_train).mean() == 1. #check the axis stuff

x_train = np.array(list(x_train) + list(build_symetry(x_train)))
x_test  = np.array(list(x_test)  + list(build_symetry(x_test)))

y_train = np.array(list(y_train) + list(y_train))
y_test  = np.array(list(y_test)  + list(y_test))

print(" - blip bloup)")
"""
#we need to concat the 2 embedings into a same vector to pass it to the classifier
x_train = np.array(list(map(lambda c: list(c[0]) + list(c[1]), x_train)))
x_test  = np.array(list(map(lambda c: list(c[0]) + list(c[1]), x_test)))
"""
#substract w1 - w2 to get a difference of words to find if there is a dimension which encode synonymy
x_train = x_train[:, 0, :] - x_train[:, 1, :]
x_test  = x_test[:, 0, :]  - x_test[:, 1, :]

print(" - shuffle data")
data_train = list(zip(x_train, y_train))
data_test  = list(zip(x_test,  y_test))

shuffle(data_train)
shuffle(data_test)

x_train, y_train = list(zip(*data_train))
x_test,  y_test  = list(zip(*data_test))

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test  = np.array(x_test)
y_test  = np.array(y_test)

#del data_train, data_test

print("* train a linear model to interpret the features importance for synonyms classification task")
#build a linear model to learn the feature importance for synonymy classification
clf = LinearSVC(verbose=1)
clf.fit(x_train, y_train)

acc = (clf.predict(x_test) == y_test).mean()
print("synonyms calssification accyracy : ", acc * 100, " %")

coefs = clf.coef_[0]

#save coefs
np.save("coefs_synonyms_clasif", coefs)

plt.title("Importance d'une Dimension pour Determiner la Synonymie")
plt.plot(abs(coefs))

#set a threshold
t = abs(coefs)
t = t.mean() + t.std() * 2
plt.plot([t] * len(coefs))

plt.show()








