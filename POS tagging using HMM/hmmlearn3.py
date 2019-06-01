#!/usr/bin/env python
# coding: utf-8

# In[6]:


import sys
from collections import defaultdict
import json


# In[7]:


#implement smoothing
transition = defaultdict(int)
emission = defaultdict(int)
countTransitions = defaultdict(int)
countTags = defaultdict(int)


#filename = "test.txt"
#filename = "it_isdt_train_tagged.txt"
filename = sys.argv[1]
with open(filename, 'r') as f:
    lines = f.readlines()
f.close()

for line in lines:
    prev_tag = "q0"
    for data in line.split(' '):
        #print(data)
        data = data.strip()
        indexOfSeparator = data.rindex('/')
        word = data[:indexOfSeparator]
        tag = data[indexOfSeparator+1:]
        countTransitions[prev_tag] += 1
        transition[(prev_tag,tag)] += 1
        prev_tag = tag
        emission[(word,tag)] += 1
        countTags[(tag)] += 1
        

with open('hmmmodel.txt','w') as f2:
    f2.write("Transition probabilities\n")
    f2.write(str(len(transition)) + '\n')
    #transitionProbabilities = defaultdict(float)
    for key,value in transition.items():
        transition_probability = float(value)/float(countTransitions[key[0]])
        #transitionProbabilities[key] = transition_probability
        strEntry = str(key[0])+' '+str(key[1])+' '+str(transition_probability)+'\n'
        f2.write(strEntry)
    
    f2.write("Emission probabilities\n")
    f2.write(str(len(emission)) + '\n')
    #emissionProbabilities = defaultdict(float)
    for key,value in emission.items():
        emission_probability = float(value)/float(countTags[key[1]])
        #emissionProbabilities[key] = emission_probability
        strEntry = str(key[0])+' '+str(key[1])+' '+str(emission_probability)+'\n'
        f2.write(strEntry)
f2.close()


# In[5]:





# In[ ]:




