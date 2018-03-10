import re
import glob
import codecs
import numpy as np

from collections import Counter

import sklearn

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans, MiniBatchKMeans, AffinityPropagation


def read_file(fn):
    with codecs.open(fn,encoding="latin-1") as f:
        return f.read()

files_path = glob.glob('20news-bydate-train/*/*')
classes = map(lambda t: t.split('/')[1], files_path)

def get_words(txt):
    t = filter(lambda c: c.isalpha() or c in ["'", ' ', '\n', '@'], txt.replace('-', ' ')).split()
    return filter(lambda w: (not '@' in w) and (not w[-3:] in ["edu", "com"]) and (w[:3] != "www"), t) 

#on lit les fichiers
all_txt = map(read_file, files_path)
all_txt = map(lambda t: t.lower(), all_txt)

#on trouve tous les mots
all_words = map(get_words, all_txt)
all_words = map(set, all_words)
all_words = Counter([item for sublist in all_words for item in sublist])

#suppression des mots n'apparaissant pas suffisament souvent (5)
all_words = {k : v for (k, v) in filter(lambda c: c[1] > 15, zip(all_words.keys(), all_words.values()))}
#suppression des 200 premiers mots
all_words = {k: i for (k, i) in sorted(zip(all_words.keys(), all_words.values()), key=lambda k: k[1])[::-1][200:]}

all_words = sorted(all_words.keys())

all_words = {i : k for (k, i) in enumerate(all_words)}

def vectorizer(txt):
    out = [0 for _ in range(len(all_words))]
    words = get_words(txt)
    words = filter(lambda w: w in all_words, words)
    for w in set(words):
        out[all_words[w]] += words.count(w)
    out = np.array(out)
    return out #/ float(sum(out)) #faut pas normalizer pour la LDA

#on vectorize tous les documents (binary)
vectorized = np.array(np.array(map(vectorizer, all_txt))) # > 0, dtype=int)


#on fait du clustering trop styley
clst = LatentDirichletAllocation(20, n_jobs=-1)
predicted_classes = clst.fit_transform(vectorized).argmax(1)

#evaluation des clusters
clusters = set(predicted_classes)


def purity_of_one_cluster(y, y_hat, id_cluster):
    "return purity, size_of_cluster"
    tmp = zip(y, y_hat)
    tmp = filter(lambda c: c[1] == id_cluster, tmp)
    tmp = zip(*tmp)[0]
    c = Counter(tmp)
    nb_most_redundant = sorted(zip(c.keys(), c.values()), key=lambda c: c[1])[-1][1]
    cluster_size = sum(c.values()) #taille du cluster
    return nb_most_redundant / float(cluster_size), cluster_size


def purity(y, y_hat, clusters, p=False):
    tmp = map(lambda c: purity_of_one_cluster(y, y_hat, c), clusters)
    if p:
        from pprint import pprint
        pprint(tmp) 
    return sum(map(lambda c: c[0] * c[1], tmp)) / float(len(classes))


print("purity {}".format(purity(classes, predicted_classes, clusters)))


## Quelques optimisation a faire pour tenter les perfs
# + suppimer les 200 premiers mots
# + plus faire du binaire
# + virer les clusters nuls
## DONE: passage de 13% a 28%

# - trouver les mots clefs des clusters

def n_most_important_words_one_component_naive(component, n):
    idx = component.argsort()[::-1][:n]
    words = sorted(all_words.keys())
    return map(lambda i: words[i], idx)

def n_mots_important_words_by_cluster_naive(components, n):
    return map(lambda c: n_most_important_words_one_component_naive(c, n), components)

print("mots caracteristiques par clusters : ")
print(n_mots_important_words_by_cluster_naive(clst.components_, 10))


#fusion de categories
def fusion(cat):
    if cat in ['alt.atheism', 'soc.religion.christian', 'talk.religion.misc']:
         return 'religion'
    elif cat in ['comp.sys.mac.hardware', 'comp.sys.ibm.pc.hardware', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.windows.x']:
        return 'computer'
    elif cat in ['rec.autos', 'rec.motorcycles']:
        return 'vehicule'
    elif cat in ['rec.sport.baseball', 'rec.sport.hockey']:
        return 'sport'
    elif cat in ['talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc']:
        return 'politic'
    else:
        return cat

new_classes = map(fusion, classes)
print("new purity {}".format(purity(new_classes, predicted_classes, clusters)))
#new purity 51%

