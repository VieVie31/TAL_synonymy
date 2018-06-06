# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:01:10 2018

@author: 3770640
"""

def load(filename):
    listeDoc = list()
    with open(filename, "r") as f:
        doc = list()
        for ligne in f:
            #print "l : ",len(ligne)," ",ligne
            if len(ligne) < 2: # fin de doc
                listeDoc.append(doc)
                doc = list()
                continue
            mots = ligne.split(" ")
            doc.append((mots[0],mots[1]))
    return listeDoc

# =============== chargement ============
filename = "dataWapiti/wapiti/chtrain.txt" # a modifier
filenameT = "dataWapiti/wapiti/chtest.txt" # a modifier

alldocs = load(filename)
alldocsT = load(filenameT)

print (len(alldocs)," docs read")
print (len(alldocsT)," docs (T) read")

#methode a base de dico
from collections import defaultdict
from itertools import groupby

def most_common(L):
    return max(
        groupby(sorted(L)),
        key=lambda c:(len(list(c[1])),-L.index(c[0]))
    )[0]

def preprocess(w):
    return w.lower()

def make_classifier(train, preprocesss_function=preprocess, default_value='<unk>'):
    dico = defaultdict(lambda: default_value, key="<UNK>")
    for s in train:
        for m, c in s:
            try:
                dico[preprocesss_function(m)].append(c)
            except: 
                dico[preprocesss_function(m)] = [c]
    return dico

def get_acc(test, dico, preprocesss_function=preprocess):
    t, b = 0, 0
    errors = []
    for s in test:
        for m, c in s:
            if not most_common(dico[preprocesss_function(m)]) == c:
                b -= 1
                errors.append(m)
            b += 1
            t += 1
    return  b / float(t), errors


#sans preprocessing : 75.58%
#avec lowering :      74.73%
#avec casse - les s : 70.99%
#lower - s :          69.46%
#avec un vote sur la classe majoritaire : 76.74%


clf_good = make_classifier(alldocs, preprocesss_function=lambda w: w)
errors_1 =  get_acc(alldocsT, clf_good, preprocesss_function=lambda w: w)[1]

clf = make_classifier(alldocs, preprocesss_function=lambda w: w.lower())
errors_2 =  get_acc(alldocsT, clf, preprocesss_function=lambda w: w.lower())[1]

errors_1 = set(errors_1)
errors_2 = set(errors_2)
mots_devenant_faux_apres_lowering =  errors_2 - errors_1


def get_err(test,dico, preprocesss_function=preprocess):
    return 1 - get_acc(test, dico, preprocesss_function=preprocess)[0]

#pourcentage d'erreur sans preprocessing:
perr_1 = get_err(alldocsT, clf, preprocesss_function=lambda w: w)
#pourcentage d'erreur avec lower preprocessing:
perr_2 = get_err(alldocsT, clf, preprocesss_function=lambda w: w.lower())

from sklearn.metrics import confusion_matrix

y = []
y_hat = []

for s in alldocsT:
    for m, c in s:
        y.append(c)
        y_hat.append(most_common(clf_good[m]))
        
labels = sorted(set(y))
cm = confusion_matrix(y, y_hat, labels=labels)

#passage au log pour mieux voir
import numpy as np

cm += 1
cm = np.log(cm)

import matplotlib.pyplot as plt

"""
print(labels)
plt.title("confusion matrix")
plt.imshow(cm, cmap='gray')
plt.show()
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, xlabel='Predicted label', ylabel='True label'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    """for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    """
    plt.tight_layout()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

#plot_confusion_matrix(cm, labels, normalize=True)
#plot_confusion_matrix(cm, labels, normalize=True)


##########################################################################


# allx: liste de séquences d'observations
# allq: liste de séquences d'états
# N: nb états
# K: nb observation

liste_des_matrices_A = []

def learnHMM(allx, allq, N, K, initTo1=True):
    global liste_des_matrices_A
    liste_des_matrices_A = []
    if initTo1:
        eps = 1e-5
        A = np.ones((N,N))*eps
        B = np.ones((N,K))*eps
        Pi = np.ones(N)*eps
    else:
        A = np.zeros((N,N))
        B = np.zeros((N,K))
        Pi = np.zeros(N)
    liste_des_matrices_A.append(A)
    for x,q in zip(allx,allq):
        Pi[int(q[0])] += 1
        for i in range(len(q)-1):
            A[int(q[i]),int(q[i+1])] += 1
            B[int(q[i]),int(x[i])] += 1
            liste_des_matrices_A.append(A)
        B[int(q[-1]),int(x[-1])] += 1 # derniere transition
    A = A/np.maximum(A.sum(1).reshape(N,1),1) # normalisation
    B = B/np.maximum(B.sum(1).reshape(N,1),1) # normalisation
    Pi = Pi/Pi.sum()
    liste_des_matrices_A.append(A)
    return Pi , A, B

def viterbi(x,Pi,A,B):
    T = len(x)
    N = len(Pi)
    logA = np.log(A)
    logB = np.log(B)
    logdelta = np.zeros((N,T))
    psi = np.zeros((N,T), dtype=int)
    S = np.zeros(T)
    logdelta[:,0] = np.log(Pi) + logB[:,x[0]]
    #forward
    for t in range(1,T):
        logdelta[:,t] = (logdelta[:,t-1].reshape(N,1) + logA).max(0) + logB[:,x[t]]
        psi[:,t] = (logdelta[:,t-1].reshape(N,1) + logA).argmax(0)
    # backward
    logp = logdelta[:,-1].max()
    S[T-1] = logdelta[:,-1].argmax()
    for i in range(2,T+1):
        S[T-i] = psi[int(S[T-i+1]),T-i+1]
    return S, logp #, delta, psi

import numpy as np
# alldocs etant issu du chargement des données

buf = [[m for m,c in d ] for d in alldocs]
mots = []
[mots.extend(b) for b in buf]
mots = np.unique(np.array(mots))
nMots = len(mots)+1 # mot inconnu

mots2ind = dict(zip(mots,range(len(mots))))
mots2ind["UUUUUUUU"] = len(mots)

buf2 = [[c for m,c in d ] for d in alldocs]
cles = []
[cles.extend(b) for b in buf2]
cles = np.unique(np.array(cles))
cles2ind = dict(zip(cles,range(len(cles))))

nCles = len(cles)

print(nMots,nCles," in the dictionary")

# mise en forme des données
allx  = [[mots2ind[m] for m,c in d] for d in alldocs]
allxT = [[mots2ind.setdefault(m,len(mots)) for m,c in d] for d in alldocsT]

allq  = [[cles2ind[c] for m,c in d] for d in alldocs]
allqT = [[cles2ind.setdefault(c,len(cles)) for m,c in d] for d in alldocsT]

#Apprentissage HMM
Pi, A, B = learnHMM(allx, allq, nCles, nMots)

#évaluation HMM
y_hat = []
for s in allxT:
    y_hat.extend(viterbi(s, Pi, A, B)[0])

y = []
for s in allqT:
    y.extend(s)

y = np.array(y)
y_hat = np.array(y_hat)

print("hmm acc : ", (y == y_hat).mean()) #on a trouve parail, on est tro for, sé vré, on gair sa mèr

#new state of the art : 81.01%

ind2cles = {v: k for k, v in cles2ind.items()}

#affichage des transitions entre tags
plot_confusion_matrix(A, ind2cles.values(), title="matrice de transition", ylabel="from", xlabel="to")
"""
#affichage de la matrice de confusion (plus mieux quavanyt)
labels = sorted(set(y))
cm = confusion_matrix(y, y_hat, labels=list(map(lambda k: ind2cles[k], labels)))
plot_confusion_matrix(cm, list(map(lambda k: ind2cles[k], labels)), normalize=True)
"""

####################
#Preprocessing pour HMM

#ajout du tag : <EOS>   --> 81.22% (sans biais)
#ajout du lowering      --> 81.43% (avec le EOS)
alldocs = load(filename)
alldocsT = load(filenameT)

alldocs  = list(map(lambda s: list(map(lambda c: (c[0].lower(), c[1]), s)) + [('<EOS>', '<EOS>')], alldocs))
alldocsT = list(map(lambda s: list(map(lambda c: (c[0].lower(), c[1]), s)) + [('<EOS>', '<EOS>')], alldocsT))

buf = [[m for m,c in d ] for d in alldocs]
mots = []
[mots.extend(b) for b in buf]
mots = np.unique(np.array(mots))
nMots = len(mots)+1 # mot inconnu

mots2ind = dict(zip(mots,range(len(mots))))
mots2ind["UUUUUUUU"] = len(mots)

buf2 = [[c for m,c in d ] for d in alldocs]
cles = []
[cles.extend(b) for b in buf2]
cles = np.unique(np.array(cles))
cles2ind = dict(zip(cles,range(len(cles))))

nCles = len(cles)
allx  = [[mots2ind[m] for m,c in d] for d in alldocs]
allxT = [[mots2ind.setdefault(m,len(mots)) for m,c in d] for d in alldocsT]

allq  = [[cles2ind[c] for m,c in d] for d in alldocs]
allqT = [[cles2ind.setdefault(c,len(cles)) for m,c in d] for d in alldocsT]

Pi, A, B = learnHMM(allx, allq, nCles, nMots)

y_hat = []
for s in allxT:
    y_hat.extend(viterbi(s, Pi, A, B)[0][:-1])

y = []
for s in allqT:
    y.extend(s[:-1])

y = np.array(y)
y_hat = np.array(y_hat)

print("hmm acc : ", (y == y_hat).mean()) 



##################################################
#autre tentative : renverser l'ordre de la phrase

alldocs = load(filename)
alldocsT = load(filenameT)

alldocs  = list(map(lambda s: list(map(lambda c: (c[0].lower(), c[1]), s))[::-1] + [('<EOS>', '<EOS>')], alldocs))
alldocsT = list(map(lambda s: list(map(lambda c: (c[0].lower(), c[1]), s))[::-1] + [('<EOS>', '<EOS>')], alldocsT))

buf = [[m for m,c in d ] for d in alldocs]
mots = []
[mots.extend(b) for b in buf]
mots = np.unique(np.array(mots))
nMots = len(mots)+1 # mot inconnu

mots2ind = dict(zip(mots,range(len(mots))))
mots2ind["UUUUUUUU"] = len(mots)

buf2 = [[c for m,c in d ] for d in alldocs]
cles = []
[cles.extend(b) for b in buf2]
cles = np.unique(np.array(cles))
cles2ind = dict(zip(cles,range(len(cles))))

nCles = len(cles)
allx  = [[mots2ind[m] for m,c in d] for d in alldocs]
allxT = [[mots2ind.setdefault(m,len(mots)) for m,c in d] for d in alldocsT]

allq  = [[cles2ind[c] for m,c in d] for d in alldocs]
allqT = [[cles2ind.setdefault(c,len(cles)) for m,c in d] for d in alldocsT]

Pi, A, B = learnHMM(allx, allq, nCles, nMots)

y_hat = []
for s in allxT:
    y_hat.extend(viterbi(s, Pi, A, B)[0][:-1])

y = []
for s in allqT:
    y.extend(s[:-1])

y = np.array(y)
y_hat = np.array(y_hat)

print("hmm acc : ", (y == y_hat).mean()) 





#named entities recognition
import pickle

a = pickle.load(open("/users/Etu0/3770640/M1/Sem2/TAL/TME1/maxent_ne_chunker/PY3/english_ace_multiclass.pickle", "rb"))

from nltk.tag.crf import CRFTagger

tagger = CRFTagger()
tagger.train(alldocs, u'crf.model') # donner en plus le fichier de stockage du calcul des features

tagger.tag(['Je suis à la maison']) 
print (tagger._get_features([u"Je"], 0))

from nltk.tag.perceptron import PerceptronTagger
tagger = PerceptronTagger(load=False)
tagger.train(alldocs)

# adT_seq: liste de liste de mots (=liste de phrase)
allpred_smart  = [[t for w,t in tagger.tag(adT_seq[i])] for i in range(len(adT_seq))]
allpred_stupid = [[tagger.tag([w])[0][1] for w in adT_seq[i]] for i in range(len(adT_seq))]