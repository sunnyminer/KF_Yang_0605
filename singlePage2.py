'''
Code from Kazi Saidul Hasan

Implement the SingleRank appoach for keyphrase extraction in Python

By kai
'''
import sys
import re
#from __future__ import print_function
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import lda
import os

fileList=""
goldKeyList=""
fileDir=""
goldKeyDir=""
outputDir=""

keyCount=0
iteration=20

#document names
files=list()
#gold keyphrase file names
keyList=list()

docCount=0
docLen=0
nodeCount=0
windowSize=0

#actual document
document=list()
#actual document's POS tags
posTags=list()

totalKey=0
matched=0
predicated=0

# Map<string, string> replaced by set<>
goldKeyMap=set()
# string list
goldKey=list()

# Map<String, Map<String, double>>
srGraph=dict()
# Map<String, double>
srScore=dict()
# Map<String, List<Long>>
position=dict()

#paramaters for LDA
n_samples = 2000
n_features = 1000
n_topics = 10
n_top_words = 20

def readFiles(version, file):
    obj=open(file,'r')
    lines=obj.readlines()
    global files, keyList, docCount
    for x in lines:
        if(version==1):
            files.append(x)
            #print(x)
            docCount+=1
        if(version==2):
            keyList.append(x)
    print docCount
    obj.close()

def isGoodPOS(pos):

    if pos =="NN" or pos == "NNS" or pos == "NNP" or pos == "NNPS" or pos == "JJ":
        return True
    else:
        return False



def readTxtFiles(version,file, paperID):
    global  document,posTags,docLen,position
    obj = open(file, 'r')
    lines = obj.readlines()
    for x in lines:
        #print x

        x.replace("\t","")
        x.replace("\r", "")
        x.replace("\n", "")
        token=x.split(" ")
        #remove empty string
        token=filter(None,token)
        if len(token)>0:
            for i in range(0, len(token)):
                index=-1
                index=token[i].rfind('/')
                word = token[i][:index].strip().lower()
                pos = token[i][index+1:]
                if (index==-1) or (len(word)>=2 and (word[0]=="<" or word[len(word)-1]==">")):
                    continue
                if(version==1):
                    #store
                    document.append(word)
                    posTags.append(pos)
                    #print pos,isGoodPOS(pos)
                    if not isGoodPOS(pos):
                        docLen+=1
                        continue
                    docLen+=1
                    if word in position:
                        position[word].append(docLen)
                    else:
                        position[word]=[docLen]
    #print position
    obj.close()







def readParams(file):
    obj = open(file, 'r')
    lines = obj.readlines()
    paras=[]

    for x in lines:
        match = re.search(r'=(.*)', x.replace("\r",""))
        paras.append(match.group(1))
    return paras

def clean():
    global document, posTags, goldKeyMap, goldKey, srGraph, srScore, position,docLen,nodeCount
    del document[:]
    del posTags[:]
    goldKeyMap.clear()
    del goldKey[:]
    srGraph.clear()
    srScore.clear()
    position.clear()
    docLen = 0
    nodeCount = 0


def registerGoldKey(gkey, paperID):
    global goldKeyMap,goldKey,totalKey
    gkey=gkey.strip().lower()

    if gkey not in goldKeyMap:
        goldKey.append(gkey)
        goldKeyMap.add(gkey)
        totalKey+=1


def readGoldKey(file, i):
    obj = open(file, 'r')
    lines = obj.readlines()

    for x in lines:
        row = x
        registerGoldKey(row, i)


def normalize():
    global  srGraph
    for key, value in srGraph.items():
        sum=0.0
        word=key
        for k,v in srGraph[word].items():
            sum+=v
        for k,v in srGraph[word].items():
            if sum>0:
                srGraph[word][k]=v/sum
            else:
                srGraph[word][k]=0.0



def buildGraph():
    global document, position, windowSize, srGraph, nodeCount
    for i in range(0,len(document)):
        word=document[i]
        if(word in position):
            index =i-1
            count=int(windowSize)

            while index>=0 and count>0:
                neighbor=document[index]
                if  neighbor in position:
                    if(word in srGraph):
                        if neighbor in srGraph[word]:
                            srGraph[word][neighbor]+=1
                        else:
                            srGraph[word][neighbor]=1
                    else:
                        srGraph[word]={neighbor:1}

                    if(srGraph[word][neighbor]==1):
                        nodeCount+=1
                index-=1
                count-=1

            index=i+1
            count=int(windowSize)

            while index<len(document) and count>0 :
                neighbor=document[index]
                if(neighbor in position):
                    if neighbor in position:
                        if (word in srGraph):
                            if neighbor in srGraph[word]:
                                srGraph[word][neighbor] += 1
                            else:
                                srGraph[word][neighbor] = 1
                        else:
                            srGraph[word] = {neighbor: 1}
                    if(srGraph[word][neighbor]==1):
                        nodeCount+=1
                index+=1
                count=count-1
    normalize()

def initialize():
    global srGraph,position, srScore,n_topics
    for k,v in  srGraph.items():
        if k in position:
            srScore[k]=dict()
            for i in range(0,n_topics):
                srScore[k][i]=1.0/n_topics


def singleRank():
    global  srScore,iteration, nodeCount,srGraph
    initialize()
    for it in range(1,iteration+1):
        srScoreTemp = dict()
        for key,value in srScore.items():
            for topic, val in value.items():
                word=key
                score=0.0
                for k,v in srGraph[word].items():
                    neighbor=k
                    if neighbor!="":
                        score+=srGraph[neighbor][word] * srScore[neighbor]
                score *=0.85
                score += (0.15 / (1.0 * nodeCount))
                srScoreTemp[word] = score
        srScore.clear()
        srScore=srScoreTemp
        #srScoreTemp.clear()

# candClause  <string , int>
def extractPatterns(candClause):
    global posTags,document
    count=0
    # string list
    pattern=list()
    # string list
    patternPOS=list()

    for i in range(0,len(posTags)):
        if isGoodPOS(posTags[i]):
            pattern.append(document[i])
            patternPOS.append(posTags[i])
        elif (len(pattern)>0 and not isGoodPOS(posTags[i])):
            s=""
            for j in range(0, len(pattern)):
                s+=pattern[j]
                if j+1< len(pattern):
                    s+=" "
            if patternPOS[len(patternPOS)-1]!="JJ":
                if s not in candClause:
                    count+=1
                    candClause[s]=1
                else:
                    candClause[s]+=1

            del pattern[:]
            del patternPOS[:]

    if len(pattern)>0:
        s=""
        for j in range(0, len(pattern)):
            s+=pattern[j]
            if (j+1)< len(pattern):
                s+= " "

        if patternPOS[len(patternPOS)-1]!="JJ":
            if s not in candClause:
                count+=1
                candClause[s]=1
            else:
                candClause[s]+=1

        del pattern[:]
        del patternPOS[:]

    return count


def getTotalScore(s,fileIndex):
    global  srScore
    tokens=s.split(" ")
    d=0.0
    for i in range(0,len(tokens)):
        if(tokens[i] in srScore):
            for topic, val in srScore[tokens[i]].items():
                #build
                d+= srScore[tokens[i]][topic]*ldaModel.doc_topic_[fileIndex][topic]
    return d

def replaceLowest(topKey, topKeyVal, key, val):
    index=0
    minVal=0.0
    for i in range(0,len(topKeyVal)):
        if i==0:
            index=0
            minVal=topKeyVal[i]
        elif topKeyVal[i]<minVal:
            index=i
            minVal=topKeyVal[i]

    if val>minVal:
        topKey[index]=key
        topKeyVal[index]=val

    return topKey, topKeyVal

def isGoldKey(key):
    global  goldKey
    for i in range(0, len(goldKey)):
        if goldKey[i]==key:
            return True
    return False


def score(file,fileIndex):

    global  predicated,matched,keyCount
    obj = open(file, 'w')

    # <string, int>
    candClause=dict()

    topKey=list() # string list
    topKeyVal=list() # double list

    extractPatterns(candClause)

    for key, val in candClause.items():
        if val>0:
            s=key
            score=getTotalScore(s,fileIndex)

            if( len(topKey)< int(keyCount)):
                topKey.append(s)
                topKeyVal.append(score)
            else:
                topKey, topKeyVal=replaceLowest(topKey,topKeyVal,s,score)

    predicated+= len(topKey)

    for i in range(0,len(topKey)):
        pkey=topKey[i]
        obj.write(pkey)
        if(isGoldKey(pkey) or isGoldKey(pkey+"s") or (pkey[len(pkey)-1]=='s' and isGoldKey(pkey[:len(pkey)-1]))):
            matched+=1






def textExtract(doc):
#	print("Currently loading " + doc)
	result = ""
	with open(doc) as d:
		for line in d:
		    result += line
        #print(result)
	return result

#read documents in certain directory
def computLda(fileList):
    global n_samples ,n_features,n_topics,n_top_words
    myPath = "DucRaw"
    allTextDoc = [myPath + '/' + f.replace(".pos\n",".txt") for f in fileList ]
    #allTextDoc = [myPath + '/' + f for f in fileList]
    #print(allTextDoc)


    allText = []
    for text in allTextDoc:
        print text
        result = textExtract(text)
        allText.append(result)

    print "Extracting tf features for LDA..."
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                    max_features=n_features,
                                    stop_words='english')
    tf = tf_vectorizer.fit_transform(allText)

    print("Fitting LDA models with tf features, "
          "n_samples=%d and n_features=%d..."
          % (n_samples, n_features))
    model= lda.LDA(n_topics=n_topics,n_iter=5,random_state=1)
    model.fit(tf)

    tf_feature_names = tf_vectorizer.get_feature_names()
    print tf_feature_names
    global wordIndex
    wordIndex= dict()
    for i in range(len(tf_feature_names)):
        wordIndex[tf_feature_names[i]]=i

    #print model.topic_word_
    #print model.doc_topic_



    return model



def main(arg):
    '''
    :param arg:
    :return:
    '''
    print "Reading params..."
    #reading params
    global  fileList, goldKeyList,fileDir,goldKeyDir,outputDir,keyCount,windowSize,matched,predicated,totalKey,ldaModel
    fileList, goldKeyList, fileDir, goldKeyDir, outputDir, keyCount, windowSize=readParams(arg)

    global  files
    #read document
    readFiles(1,fileList)
    ldaModel = computLda(files)
    #read gold key file list
    if goldKeyList!="":
        readFiles(2,goldKeyList)

    for i in range(0,docCount):
        clean()
        txt=""
        txt=fileDir+files[i]

        print "Processing ", files[i], " ..."
        readTxtFiles(1,txt.rstrip(),files[i])
        #readTxtFiles(1,(fileDir+files[0]).rstrip(),files[0])
        key=""
        if goldKeyDir!="" and i<len(keyList):
            key=goldKeyDir+keyList[i]
            readGoldKey(key.rstrip(),i)


        buildGraph()
        singleRank()

        #score each candidate phrase and output top-scoring phrases
        score((outputDir+files[i]).rstrip()+".phrases",i)


    print "---"*30

    p=(100.0*matched)/(1.0*predicated)
    r=(100.0*matched) / (1.0*totalKey)
    f= (2.0*p*r)/(p+r)

    print "matched= ",matched
    print "predicated= ",predicated
    print "totalKey= ",totalKey

    print "Recall= ", r
    print "Precision= ", p
    print "F-Score= ", f
    print "done"







if __name__ == '__main__':
    main('conf.txt')
