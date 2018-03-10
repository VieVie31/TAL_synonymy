# Document Clustering

:warning: Because of the python generators this code need to be executed with Python :two: :exclamation:

Trying to cluster the 20 Newsgroup dataset' documents.

The dataset is available [here](!http://qwone.com/~jason/20Newsgroups/).


The clustering methods tested where :

 - KMeans
 - MiniBatchKMeans
 - Lattent Dirichlet Allocation (performed best)

The documents encodings tested where (BoW and with stopwords removal):

 - Binary encoding
 - Count encoding (worked best with LDA)
 - Freq encoding

The initial purity evalutation was _9%_ then goes up to _13.5%_ and finaly reached ___28%___.
Whith topics merging (as some of them was redundants) it reachead _51%_.

For the keywords extraction of each cluster found, the words was allready pertinent by taking the most proables because all the stopwords where removed before. 
Here the 200 most frequent words are considered as stopwords.

Here are the keywords by cluster :

```
game, play, bike, nhl, toronto, teams, chicago, points, best, illinois
apr, insurance, group, gmt, message, chris, newsgroup, keyboard, posts, dan
b, db, keys, window, windows, dos, ms, gordon, c, ed
ca, bill, vs, technology, institute, dod, inc, michael, james, california
turkish, armenian, medical, armenians, study, during, studies, greek, armenia, water
book, steve, post, andrew, keith, answer, questions, sound, brian, sorry
encryption, team, david, season, win, division, st, league, period, john
space, o, nasa, research, center, earth, moon, orbit, van, software
m, s, p, ', t, u, n, r, c, k
christian, evidence, christians, bible, religion, jim, word, christianity, hell, love
faith, paul, reason, health, banks, argument, non, different, rather, less
gun, q, mr, president, clinton, escrow, house, government, federal, enforcement
privacy, pittsburgh, national, security, agencies, technology, money, trade, american, cost
key, file, program, available, information, data, clipper, number, code, server
drive, card, scsi, disk, windows, pc, mb, memory, sale, hard
him, car, didn't, she, her, day, put, again, left, enough
god, jesus, life, church, john, man, wrong, god's, moral, christ
government, law, against, hockey, rights, jews, israel, men, israeli, fact
chip, games, power, mark, baseball, cd, devices, msg, serial, runs
newsreader, tin, pl, version, hp, xr, speed, high, unit, v
```

