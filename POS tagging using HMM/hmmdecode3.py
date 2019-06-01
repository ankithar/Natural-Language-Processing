#!/usr/bin/env python
# coding: utf-8

# In[4]:


import sys
from collections import defaultdict
import json
import numpy as np
import math


# In[5]:


transition = {}
emission = {}
setOfTags = set()
setOfWords = set()
wordAndTags = defaultdict(list)
tagsCount = defaultdict(int)
allTagsCount = 0
sentencesToOutput = []

with open('hmmmodel.txt','r') as f1:
    data = f1.readlines()
f1.close()
index = 2
for i in range(int(data[1])):
    line = data[index+i].strip().split(' ')
    transition[(line[0],line[1])] = float(line[2])

#print(transition)
index = int(data[1]) + 4
for i in range(int(data[index-1])):
    line = data[index+i].strip().split(' ')
    emission[(line[0],line[1])]  = float(line[2])
    setOfWords.add(line[0])
    setOfTags.add(line[1])
    wordAndTags[line[0]].append(line[1])
    tagsCount[line[1]]+=1
    allTagsCount += 1


# In[6]:


def backtrack(sentence,back):
    print("backtrack")
    #sentence = "Tutti gli esseri umani sanno di poter essere più di ciò che sono ."
    testData = sentence.strip().split(' ')
    lastDict = {k:v for k, v in back.items() if k[1] == len(testData)}
    #print(lastDict)
    probMax = float("-inf")
    stateMax = ''
    prevState = ''
    for k,v in lastDict.items():
        if v[1]>probMax:
            probMax = v[1]
            stateMax = k[0]
            prevState = v[0]

    #print(probMax)
    #print(stateMax)
    #print(prevState)

    cnt = len(testData)-1
    output = testData[cnt]+'/'+stateMax+'\n'
    while cnt>0:
        #print(prevState)
        ls = back[(prevState,cnt)]
        output = testData[cnt-1]+'/'+prevState+' '+output
        #print(ls)
        prevState = ls[0]
        cnt -=1
    #print(output)
    sentencesToOutput.append(output)


# In[7]:


def viterbi(sentence):
    print("viterbi")
    #sentence = "Corriere Sport da pagina 23 a pagina 26"
    #sentence = "Tutti gli esseri umani sanno di poter essere più di ciò che sono ."
    words = sentence.split(' ')
    #numOfWords = len(word)
    back = {}
    prev = {}

    for word in words:
        #print(word)
        if word not in setOfWords:
            #print("not there")
            #print(word)
            for tag in setOfTags:
                emission[(word,tag)]= float(tagsCount[tag])/float(allTagsCount)
                wordAndTags[word].append(tag)

    for tag in wordAndTags[words[0]]:
        if ('q0',tag) not in transition:
            transition[('q0',tag)] = 1e-15

        prob = np.log(transition[('q0',tag)]) + np.log(emission[(words[0],tag)])
        prev[tag] = prob
        back[(tag,1)] = ["q0",prob]

    cnt = 0
    for word in words[1:]:
        tagsOfWord = wordAndTags[word]
        temp = {}

        for s in tagsOfWord:
            maxProbab = float("-inf")
            maxProbabState = ''
            for prevState in prev:
                if (prevState,s) not in transition:
                    transition[(prevState,s)] = 1e-15

                prob = np.log(transition[(prevState,s)]) + np.log(emission[word,s])  + prev[prevState]
                if prob>maxProbab:
                    maxProbab = prob
                    maxProbabState  = prevState
            temp[s] = maxProbab
            back[(s,cnt+2)] = [maxProbabState,maxProbab]
        cnt += 1
        prev = temp  
    #return "done"
    backtrack(sentence, back)


# In[9]:


#with open('it_isdt_dev_raw.txt', 'r') as f2:
with open(sys.argv[1],'r') as f2:
    data = f2.readlines()
    count = 0
    fout = open("hmmoutput.txt", 'w')
    for sentence in data:
        sentence = sentence.strip()
        viterbi(sentence)
    for sentence in sentencesToOutput:
        fout.write(sentence)
    fout.close()


# In[10]:


def calculate_accuracy():
    #global all_tags_count
    output = open("hmmoutput.txt", 'r')
    #compare = open("../coding1-data-corpus/en_dev_tagged.txt", 'r')
    compare = open("it_isdt_dev_tagged.txt", 'r')
    data1 = compare.readlines()
    data2 = output.readlines()
    correct_words = 0
    total_words = 0
    tags = list()
    tags_mine = list()
    words = list()
    words1 = list()
    for count, line in enumerate(data1):
        line = line.strip()
        words = line.split(' ')
        for i in words:
            tags.append(i)
    
    for count, l in enumerate(data2):
        l = l.strip()
        words1 = l.split(' ')
        for i in words1:
            tags_mine.append(i)

    for i in range(len(tags)):
        if (tags[i] == tags_mine[i]):
            correct_words += 1
        #else:
            #pass
            #print ("{0}  {1}".format(tags[i], tags_mine[i]))
    
    print(float(correct_words)/float(len(tags)))


# In[11]:


#calculate_accuracy()


# In[ ]:




