from os import listdir
from collections import Counter
import re
import pickle
import random
import math
import tensorflow as tf

def ngramExtract(unText,gram):
    return [unText[i:i+gram] for i in range(len(unText)-gram + 1)]
    

def fromBlogsToInitial():
    files = listdir("./blogs/")
    k = 1

    for i in range(len(files)):

        f = open("./blogs/" + files[i], 'r')

        try:
            contents = f.read()
        except:
            f.close()
            continue

        toDelete = re.findall("&[^\s]*;", contents)

        seek1 = contents.find('<post>')
        seek2 = contents.find('</post>', seek1+1)
        
        author = []

        while(seek1!=-1):
        
            post = contents[seek1 + 6 : seek2].strip()

            for token in toDelete:
                post = post.replace(token, '')

            author.append(post)

            seek1 = contents.find('<post>', seek1+1)
            seek2 = contents.find('</post>', seek1+1)

        if len(author) >= 10:
            corpusLen = 0
            for post in author:
                corpusLen += len(post.split())

            if (corpusLen >= 10000):
                fout = open('./preprocessed/' + str(k), 'w')
                k += 1

                for post in author:
                    fout.write(post)
                    fout.write('\n')

                fout.close()
        
        f.close()

def fromPreprocessedToDataset(authorCount = 64, topFeatures = 4096, minGrams = 2, maxGrams = 5):
    features = list()
    for i in (authorCount + 1, 2 * authorCount + 1):
        f = open("./preprocessed/{}".format(i) , 'r')

        posts = f.readlines()

        f.close()
        
        extractFeatures(posts, features, minGrams, maxGrams)

    features = normalizeFrequency(Counter(features).most_common())[:topFeatures]

    header = [x for x,_ in features]
    print(header)

    trainingFeatures = list()
    testingFeatures = list()

    for i in range(authorCount + 1, 2 * authorCount + 1):
        f = open("./preprocessed/{}".format(i) , 'r')

        posts = f.readlines()

        f.close()

        l = len(posts)
        trainingL = int(0.6 * l)
        testL = l - trainingL

        partitionSeq = list()

        for j in range(trainingL):
            partitionSeq.append(0)
            
        for j in range(testL):
            partitionSeq.append(1)

        assert(len(partitionSeq) == l)

        random.shuffle(partitionSeq)

        training = ""
        test = ""

        for j in range(l):
            if partitionSeq[j] == 0:
                training += posts[j] + " "
            if partitionSeq[j] == 1:
                test += posts[j] + " "

        trainingAuthor = conformToFeatureSpace(header, training, minGrams, maxGrams)
        testingAuthor = conformToFeatureSpace(header, test, minGrams, maxGrams)

        trainingFeatures.append(listToMat(trainingAuthor))
        testingFeatures.append(listToMat(testingAuthor))
        print("Author {} done".format(i))

    f = open("./trainingDatasetTensor2", 'wb')

    pickle.dump(trainingFeatures,f)

    f.close()
    
    f = open("./testingDatasetTensor2", 'wb')

    pickle.dump(testingFeatures,f)

    f.close()

def conformToFeatureSpace(header, content, minGrams, maxGrams):    
    features = list()

    t = normalizeFrequency(Counter(content.split()).most_common())

    for item in t:
        features.append(item)

    for i in range(minGrams, maxGrams + 1):
        t = normalizeFrequency(Counter(ngramExtract(content,i)).most_common())
        for item in t:
            features.append(item)

    r = list()
    for item in header:
        setZero = True
        for a, freq in features:
            if item == a:
                r.append(freq)
                setZero = False
                break   

        if setZero == True:
            r.append(0.0)

    assert(len(header) == len(r))

    return r

def extractFeatures(posts, allFeatures, minGrams, maxGrams):
    content = ""

    for post in posts:
        content += post + " "

    a = content.split()

    for x in a:
        allFeatures.append(x)

    for i in range(minGrams, maxGrams + 1):
        a = ngramExtract(content,i)
        for x in a:
            allFeatures.append(x)

def listToMat(l):
    n = int(math.sqrt(len(l)))
    assert(n * n == len(l))

    mat = list()
    for i in range(n):
        row = list()
        for j in range(n):
            row.append(l[i * n + j])
        mat.append(row)
    
    return mat        

def pad(l, maxVal, padding):
    assert(len(l) <= maxVal)

    template = [padding for i in range(maxVal)]

    template[:len(l)] = l

    assert(len(template) == maxVal)

    return template

def splitListOfTuples(l):
    a = [x[0] for x in l] 
    b = [x[1] for x in l]
    return (a,b)

def flatten(l):
    return [item for sublist in l for item in sublist]

def formatForSVM(outSDAE):
    returnBox = []
    for element in outSDAE:
        returnBox.append(flatten(element))
    return returnBox

def normalizeFrequency(range):
    temp = []
    min = 0
    max = 0

    for _,b in range:
        if min == 0:
            min = b
        if max == 0:
            max = b
        if min > b:
            min = b
        if max < b:
            max = b

    div = max - min

    if div == 0:
        div = 1

    for word,freq in range:
        temp.append((word ,(freq - min) / div))

    return temp