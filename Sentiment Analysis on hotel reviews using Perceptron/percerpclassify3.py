# use this file to classify using perceptron classifier 
# Expected: generate percepoutput.txt

import sys


if __name__ == "__main__":
    model_file = str(sys.argv[1])
    output_file = "percepoutput.txt"
    input_path = str(sys.argv[2])
    
import numpy as np
import os as os
from os import walk
import string
from collections import defaultdict
from collections import Counter
import json

modelFile = sys.argv[1]
with open(modelFile, 'r') as fModel:
    lines = fModel.readlines()
bSentiment = float(lines[0])
wSentiment = json.loads(lines[1])
bAuthenticity = float(lines[2])
wAuthenticity = json.loads(lines[3])
fModel.close()

def preprocess(fileName):
    data = ''
    with open(name, 'r') as f2:
        data = f2.read()
    str1 = data.replace('\n',' ')
    exclude = set(string.punctuation)
    noPunc = ''.join(ch for ch in str1 if ch not in exclude)
    noNum = ''.join([i for i in noPunc if not i.isdigit()])
    lowerCase = ' '.join([k.lower() for k in noNum.split()])
    return lowerCase 

def calculate(wnew, bnew, x, typeOfClassification):
    a=0
    for d in x.keys():
        if d in wnew.keys():
            a += wnew[d]*x[d]    
    a += bnew

    #print("================")

    if a>0:
        if(typeOfClassification == 'sentiment'):
            value = "positive"
        else:
            value = "truthful"
        #cp +=1
    else:
        if(typeOfClassification == 'sentiment'):
            value = "negative"
        else:
            value = "deceptive"
        #cn+=1
        
    return value
           
    

fout = open('percepoutput.txt', 'w')
for root, directories, filenames in os.walk(sys.argv[2]):
    for filename in filenames:
        name = str(os.path.join(root, filename))
        #if ('README' not in name.upper() and 'fold1' in name and name.endswith(".txt")):
        if ('README' not in name.upper() and name.endswith(".txt")):
            #print(name)
            
            lowerCase = preprocess(name)
            x = Counter(lowerCase.split(' '))
            #print(len(x))
            
            sentiment = calculate(wSentiment, bSentiment, x, "sentiment")
            truth = calculate(wAuthenticity, bAuthenticity, x, "truth")
            
            strOutput = truth+" "+sentiment+" "+name+"\n"
            fout.write(strOutput)
fout.close()
                
