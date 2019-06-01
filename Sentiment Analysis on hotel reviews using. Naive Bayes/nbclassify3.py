# use this file to classify using naive-bayes classifier 
# Expected: generate nboutput.txt

import sys


if __name__ == "main":
    model_file = "nbmodel.txt"
    output_file = "nboutput.txt"
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

vocab = list()
with open('nbmodel.txt', 'r') as fModel:
    lines = fModel.readlines()
    logPrior = [float(x) for x in lines[0].split(",")]
    length1 = int(lines[1])
    count = length1

    logLiklihood = collections.defaultdict(list)
    length1 = length1+2
    #print(length1)
    for i in range(2,length1):
        tempStringList = lines[i].split(",")
        key = tempStringList[0].strip()
        logLiklihood[key] = [float(x) for x in tempStringList[1:]]
    #print(logLiklihood)
    #num = length1
    #print(num)
    #print(lines[num])
    #print("length1 = " + str(length1))
    length2 = length1+count
    #print("length2 = "+str(length2))
    #print(length2-1)
    #print(lines[length2-1])
    for j in range(length1, length2-1):
        vocab.append(lines[j].strip())
        #j+=1

C = 4
"""
print("logPrior")
print(logPrior)
"""
#print(logLiklihood['general'])

def test_NB(testDoc):
    #print(testDoc)
    sumList = list()
    for i in range(C):
        sumList.append(logPrior[i])
        for word in testDoc.split(' '):


            if word in vocab:
                """
                print(word)
                print (type(word))
                print (len(word))
                """
                sumList[i] += logLiklihood[word.strip()][i]
    #print(sumList)
    ind = np.argmax(sumList)
    return ind
count0=0
count1=0
count2=0
count3=0

fout = open('nboutput.txt', 'w')
#rootDir = ['op_spam_training_data/positive_polarity/truthful_from_TripAdvisor/fold1/','op_spam_training_data/positive_polarity/deceptive_from_MTurk/fold1/','op_spam_training_data/negative_polarity/truthful_from_Web/fold1/','op_spam_training_data/negative_polarity/deceptive_from_MTurk/fold1/']
#for root, directories, filenames in os.walk(sys.argv[0]):
for root, directories, filenames in os.walk(sys.argv[1]):
#rootDir = 'op_spam_training_data/positive_polarity/truthful_from_TripAdvisor/fold1/'
#rootDir = 'op_spam_training_data/positive_polarity/deceptive_from_MTurk/fold1/'
#rootDir = 'op_spam_training_data/negative_polarity/truthful_from_Web/fold1/'
#rootDir = 'op_spam_training_data/negative_polarity/deceptive_from_MTurk/fold1/'
#for root, directories, filenames in os.walk(rootDir):
    for filename in filenames:
        name = str(os.path.join(root, filename))

        if (name.endswith(".txt") and 'README' not in name.upper() ):

            with open(name, 'r') as f:
                text = f.read().replace('\n',' ')
                ind = test_NB(text)
                #print(ind)
                res = ''
                if ind == 0:
                    count0 +=1
                    res = "truthful positive " + name + "\n"

                elif ind == 1:
                    count1 += 1
                    res = "deceptive positive " + name + "\n"

                elif ind == 2:
                    count2 += 1
                    res = "truthful negative " + name + "\n"

                elif ind == 3:
                    count3 += 1
                    res = "deceptive negative " + name + "\n"
                #print(res)
                fout.write(res)
fout.close()
"""
print(count0)
print(count1)
print(count2)
print(count3)
"""


