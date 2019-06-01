# use this file to learn perceptron classifier 
# Expected: generate vanillamodel.txt and averagemodel.txt

import sys


if __name__ == "__main__":
    model_file = "vanillamodel.txt"
    avg_model_file = "averagemodel.txt"
    
    input_path = str(sys.argv[1])
  
    
import numpy as np
import os as os
from os import walk
import string
from collections import defaultdict
from collections import Counter
import json



stopWords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

def preprocess(review):
    str1 = review.replace('\n',' ')
    exclude = set(string.punctuation)
    noPunc = ''.join(ch for ch in str1 if ch not in exclude)
    noNum = ''.join([i for i in noPunc if not i.isdigit()])
    noSmallWords = ' '.join([j for j in noNum.split() if len(j) > 2])
    lowerCase = ' '.join([k.lower() for k in noSmallWords.split()])
    noStopWords = ' '.join([k for k in lowerCase.split() if k not in stopWords])
    return noStopWords

def readFile(filename):
    # print(filename)
    data = ''
    with open(filename, 'r') as f2:
        data = f2.read()
    preProcessedData = preprocess(data)
    # print(data)
    return preProcessedData


def randomize(a, b, c):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    shuffled_c = c[permutation]
    return shuffled_a, shuffled_b, shuffled_c


def perceptron(reviews, y, w, b, u, beta):
    c=1
    for iterNo in range(50):
        for i in range(reviews.size):
            #reviews, y, authenticity = randomize(reviews,y,authenticity)
            review = reviews[i]
        #for review in reviews:
            a = 0
            x = Counter(review.split(' '))
            for d in x.keys():
                a += x[d]*w[d]
            a += b
            #if(typeOfClassification == 'PositiveNegative'):
            #print(a)
            if y[i]=="pos" or y[i]=="tru":
                classY = 1
            elif y[i]=="neg" or y[i]=="dec":
                classY = -1
            a *= classY;
            if a<=0:
                for d in x.keys():
                    w[d] += (x[d]*classY)
                    u[d] += (classY * c * x[d])
                    #print(str(x[d])+" "+str(w[d]))
                #tmp = b
                b += classY  
                beta += (classY*c)
                #print(str(tmp)+" "+str(b))
            c+=1
            
            
    
    #print(w)
    #print(b)
    unew = {}
    for word in w.keys():
        unew[word] = w[word] - (u[word]/c)
    betanew = b - (beta/c)
    return w,b,unew,betanew

sentiment = []
authenticity = []
reviews = []
vocab = defaultdict(int)
#for root, directories, filenames in os.walk("op_spam_training_data"):
for root, directories, filenames in os.walk(sys.argv[1]):
    for filename in filenames:
        name = str(os.path.join(root, filename))
        #if ('README' not in name.upper() and 'fold1' not in name and name.endswith(".txt")):
        if ('README' not in name.upper() and name.endswith(".txt")):
            # docCount += 1
            #print(name)
            review = readFile(name)
            reviews.append(review)
            if ('positive' in name.lower()):
                sentiment.append('pos')
            elif('negative' in name.lower()):
                sentiment.append('neg')
            if('truthful' in name.lower()):
                authenticity.append('tru')
            elif('deceptive' in name.lower()):
                authenticity.append('dec')
                
reviews = np.array(reviews)
sentiment = np.array(sentiment)
authenticity = np.array(authenticity)
reviews,sentiment,authenticity = randomize(reviews,sentiment,authenticity)

for sentence in reviews:
    for word in sentence.split(' '):
        vocab[word]+=1
        
w = {}
u = {}
for key in vocab.keys():
    w[key]=0
    u[key]=0
    
b=0
beta=0
    
wSentiment,bSentiment, uSentiment, betaSentiment = perceptron(reviews,sentiment, w, b,u,beta)

w = {}
u = {}
for key in vocab.keys():
    w[key]=0
    u[key]=0
    
b=0
beta=0

wAuthenticity, bAuthenticity, uAuthenticity, betaAuthenticity = perceptron(reviews, authenticity, w, b, u, beta)


with open('vanillamodel.txt', 'w') as fModel:
    fModel.write(str(bSentiment)+"\n")
    #fModel.write(str(len(wSentiment))+"\n")
    json.dump(wSentiment, fModel)
    fModel.write("\n")
    fModel.write(str(bAuthenticity)+"\n")
    #fModel.write(str(len(wAuthenticity))+"\n")
    json.dump(wAuthenticity, fModel)
fModel.close()
        
with open('averagedmodel.txt', 'w') as fModel2:
    fModel2.write(str(betaSentiment)+"\n")
    #fModel.write(str(len(wSentiment))+"\n")
    json.dump(uSentiment, fModel2)
    fModel2.write("\n")
    fModel2.write(str(betaAuthenticity)+"\n")
    #fModel.write(str(len(wAuthenticity))+"\n")
    json.dump(uAuthenticity, fModel2)
fModel2.close()
                
