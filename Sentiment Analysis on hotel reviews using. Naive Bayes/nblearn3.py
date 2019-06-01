# use this file to learn naive-bayes classifier 
# Expected: generate nbmodel.txt

import sys


if __name__ == "main":
    model_file = "nbmodel.txt"
    input_path = str(sys.argv[0])
    
import math
import sys
import string
import io
import os as os
from os import walk
import collections
from collections import Counter
import numpy as np
import re

fModel = open('nbmodel.txt', 'w')

data = ['', '', '', '']
countDocs = [0, 0, 0, 0]


# print(type(data[0]))

def readFile(filename):
    # print(filename)
    data = ''
    with open(filename, 'r') as f2:
        data = f2.read()
    # print(data)
    return data


#print(sys.argv[0])
for root, directories, filenames in os.walk(sys.argv[1]):
#for root, directories, filenames in os.walk("op_spam_training_data"):
    for filename in filenames:
        name = str(os.path.join(root, filename))
        #print(name)
        if (name.endswith(".txt") and 'README' not in name.upper()):
        #if ('README' not in name.upper() and name.endswith(".txt")):
            # docCount += 1
            #print(name)
            
            if ('negative' in name.lower()):
                #print("neg tru")
                if ('deceptive' in name.lower()):
                    #print("neg dec")
                    data[3] += readFile(name)
                    countDocs[3] += 1
                elif ('truthful' in name.lower()):
                    data[2] += readFile(name)
                    countDocs[2] += 1
                
            elif ('positive' in name.lower()):
                if ('deceptive' in name.lower()):
                    #print("pos dec")
                    data[1] += readFile(name)
                    countDocs[1] += 1
                elif ('truthful' in name.lower()):
                    #print("pos tru")
                    data[0] += readFile(name)
                    countDocs[0] += 1
                

#print(countDocs)
#print(data[0])


def preprocess(data):
    # global data
   
    
    noPunc = ''
    for x in data:
        if x in string.punctuation:
            noPunc += ' '
        else:
            noPunc += x
    
         
    noNum = ''.join([i for i in noPunc if not i.isdigit()])
    # print(noNum.split())
    noSmallWords = ' '.join([j for j in noNum.split() if len(j) > 2])
    lowerCase = ' '.join([k.lower() for k in noSmallWords.split()])
    #lowerCase = ' '.join([k.lower() for k in noNum.split()])
    
    
    stopWords = ['the','we','they','you','and','a', 'able', 'about', 'above', 'abst', 'accordance', 'according', 'accordingly', 'across', 'act',
                 'actually', 'added', 'adj', 'affected', 'affecting', 'affects', 'after', 'afterwards', 'again',
                 'against', 'ah', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am',
                 'among', 'amongst', 'an', 'and', 'announce', 'another', 'any', 'anybody', 'anyhow', 'anymore',
                 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'apparently', 'approximately', 'are', 'aren',
                 'arent', 'arise', 'around', 'as', 'aside', 'ask', 'asking', 'at', 'auth', 'available', 'away',
                 'awfully', 'b', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before',
                 'beforehand', 'begin', 'beginning', 'beginnings', 'begins', 'behind', 'being', 'believe', 'below',
                 'beside', 'besides', 'between', 'beyond', 'biol', 'both', 'brief', 'briefly', 'but', 'by', 'c', 'ca',
                 'came', 'can', 'cannot', "can't", 'cause', 'causes', 'certain', 'certainly', 'co', 'com', 'come',
                 'comes', 'contain', 'containing', 'contains', 'could', 'couldnt', 'd', 'date', 'did', "didn't",
                 'different', 'do', 'does', "doesn't", 'doing', 'done', "don't", 'down', 'downwards', 'due', 'during',
                 'e', 'each', 'ed', 'edu', 'effect', 'eg', 'eight', 'eighty', 'either', 'else', 'elsewhere', 'end',
                 'ending', 'enough', 'especially', 'et', 'et-al', 'etc', 'even', 'ever', 'every', 'everybody',
                 'everyone', 'everything', 'everywhere', 'ex', 'except', 'f', 'far', 'few', 'ff', 'fifth', 'first',
                 'five', 'fix', 'followed', 'following', 'follows', 'for', 'former', 'formerly', 'forth', 'found',
                 'four', 'from', 'further', 'furthermore', 'g', 'gave', 'get', 'gets', 'getting', 'give', 'given',
                 'gives', 'giving', 'go', 'goes', 'gone', 'got', 'gotten', 'h', 'had', 'happens', 'hardly', 'has',
                 "hasn't", 'have', "haven't", 'having', 'he', 'hed', 'hence', 'her', 'here', 'hereafter', 'hereby',
                 'herein', 'heres', 'hereupon', 'hers', 'herself', 'hes', 'hi', 'hid', 'him', 'himself', 'his',
                 'hither', 'home', 'how', 'howbeit', 'however', 'hundred', 'i', 'id', 'ie', 'if', "i'll", 'im',
                 'immediate', 'immediately', 'importance', 'important', 'in', 'inc', 'indeed', 'index', 'information',
                 'instead', 'into', 'invention', 'inward', 'is', "isn't", 'it', 'itd', "it'll", 'its', 'itself', "i've",
                 'j', 'just', 'k', 'keep\tkeeps', 'kept', 'kg', 'km', 'know', 'known', 'knows', 'l', 'largely', 'last',
                 'lately', 'later', 'latter', 'latterly', 'least', 'less', 'lest', 'let', 'lets', 'like', 'liked',
                 'likely', 'line', 'little', "'ll", 'look', 'looking', 'looks', 'ltd', 'm', 'made', 'mainly', 'make',
                 'makes', 'many', 'may', 'maybe', 'me', 'mean', 'means', 'meantime', 'meanwhile', 'merely', 'mg',
                 'might', 'million', 'miss', 'ml', 'more', 'moreover', 'most', 'mostly', 'mr', 'mrs', 'much', 'mug',
                 'must', 'my', 'myself', 'n', 'na', 'name', 'namely', 'nay', 'nd', 'near', 'nearly', 'necessarily',
                 'necessary', 'need', 'needs', 'neither', 'never', 'nevertheless', 'new', 'next', 'nine', 'ninety',
                 'no', 'nobody', 'non', 'none', 'nonetheless', 'noone', 'nor', 'normally', 'nos', 'not', 'noted',
                 'nothing', 'now', 'nowhere', 'o', 'obtain', 'obtained', 'obviously', 'of', 'off', 'often', 'oh', 'ok',
                 'okay', 'old', 'omitted', 'on', 'once', 'one', 'ones', 'only', 'onto', 'or', 'ord', 'other', 'others',
                 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'owing', 'own',
                 'p', 'page', 'pages', 'part', 'particular', 'particularly', 'past', 'per', 'perhaps', 'placed',
                 'please', 'plus', 'poorly', 'possible', 'possibly', 'potentially', 'pp', 'predominantly', 'present',
                 'previously', 'primarily', 'probably', 'promptly', 'proud', 'provides', 'put', 'q', 'que', 'quickly',
                 'quite', 'qv', 'r', 'ran', 'rather', 'rd', 're', 'readily', 'really', 'recent', 'recently', 'ref',
                 'refs', 'regarding', 'regardless', 'regards', 'related', 'relatively', 'research', 'respectively',
                 'resulted', 'resulting', 'results', 'right', 'run', 's', 'said', 'same', 'saw', 'say', 'saying',
                 'says', 'sec', 'section', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'self',
                 'selves', 'sent', 'seven', 'several', 'shall', 'she', 'shed', "she'll", 'shes', 'should', "shouldn't",
                 'show', 'showed', 'shown', 'showns', 'shows', 'significant', 'significantly', 'similar', 'similarly',
                 'since', 'six', 'slightly', 'so', 'some', 'somebody', 'somehow', 'someone', 'somethan', 'something',
                 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'specifically', 'specified',
                 'specify', 'specifying', 'still', 'stop', 'strongly', 'sub', 'substantially', 'successfully', 'such',
                 'sufficiently', 'suggest', 'sup', 'sure\tt', 'take', 'taken', 'taking', 'tell', 'tends', 'th', 'than',
                 'thank', 'thanks', 'thanx', 'that', "that'll", 'thats', "that've", 'the', 'their', 'theirs', 'them',
                 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'thered', 'therefore', 'therein',
                 "there'll", 'thereof', 'therere', 'theres', 'thereto', 'thereupon', "there've", 'these', 'they',
                 'theyd', "they'll", 'theyre', "they've", 'think', 'this', 'those', 'thou', 'though', 'thoughh',
                 'thousand', 'throug', 'through', 'throughout', 'thru', 'thus', 'til', 'tip', 'to', 'together', 'too',
                 'took', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', 'ts', 'twice', 'two', 'u',
                 'un', 'under', 'unfortunately', 'unless', 'unlike', 'unlikely', 'until', 'unto', 'up', 'upon', 'ups',
                 'us', 'use', 'used', 'useful', 'usefully', 'usefulness', 'uses', 'using', 'usually', 'v', 'value',
                 'various', "'ve", 'very', 'via', 'viz', 'vol', 'vols', 'vs', 'w', 'want', 'wants', 'was', 'wasnt',
                 'way', 'we', 'wed', 'welcome', "we'll", 'went', 'were', 'werent', "we've", 'what', 'whatever',
                 "what'll", 'whats', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby',
                 'wherein', 'wheres', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whim', 'whither', 'who',
                 'whod', 'whoever', 'whole', "who'll", 'whom', 'whomever', 'whos', 'whose', 'why', 'widely', 'willing',
                 'wish', 'with', 'within', 'without', 'wont', 'words', 'world', 'would', 'wouldnt', 'www', 'x', 'y',
                 'yes', 'yet', 'you', 'youd', "you'll", 'your', 'youre', 'yours', 'yourself', 'yourselves', "you've",
                 'z', 'zero']
                 
               
    

    noStopWords = ' '.join([k for k in lowerCase.split() if k not in stopWords])
    
    
    listOfWords = noStopWords.split()
    
    counts = Counter(listOfWords)
    
    return counts


countsPosTruth = preprocess(data[0])
countsPosDec = preprocess(data[1])
countsNegTruth = preprocess(data[2])
countsNegDec = preprocess(data[3])

vocab = countsPosTruth+countsNegTruth+countsPosDec+countsNegDec

C = 4
logPrior = list()
logLiklihood = collections.defaultdict(list)
print(countDocs)
ndoc = countDocs[0] + countDocs[1] + countDocs[2] + countDocs[3]


def train_NB():
    for i in range(C):
        nc = countDocs[i]
        if (i == 0):
            bigDoc = countsPosTruth
        if (i == 1):
            bigDoc = countsPosDec
        if (i == 2):
            bigDoc = countsNegTruth
        if (i == 3):
            bigDoc = countsNegDec

        
        logPriorValue = float(nc)/float(ndoc)
        logPrior.append(np.log(float(logPriorValue)))

        #var = vocab.keys()
        
        for word in vocab.keys():
            countWC = bigDoc[word]
            countWPC = vocab[word]
            # print(countWPC)
            prob = (countWC + 1) / (sum(bigDoc.values()) + len(vocab))
            # prob = (countWC+1)/(countWPC)
            logProb = np.log(prob)
            logLiklihood[word].append(logProb)

train_NB()

fModel = open('nbmodel.txt', 'w')
fModel.write(str(logPrior[0])+","+str(logPrior[1])+","+str(logPrior[2])+","+str(logPrior[3])+"\n")
fModel.write(str(len(logLiklihood))+"\n")
for key,value in logLiklihood.items():
    fModel.write(str(key)+","+str(value[0])+","+str(value[1])+","+str(value[2])+","+str(value[3])+"\n")
#fModel.write(str(len(vocab))+"\n")
for key,value in vocab.items():
    fModel.write(key+"\n")
fModel.close()