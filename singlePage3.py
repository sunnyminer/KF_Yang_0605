
import sys
import re
import json
#from __future__ import print_function
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import lda
import os
import operator
import sys
import math
reload(sys)
sys.setdefaultencoding('utf8')


fileList=""
goldKeyList=""
fileDir=""
goldKeyDir=""
outputDir=""
JsonDir=""
JsonRelDir=""

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
tagmeTagList = dict()
idEntityMap = dict()
docIndices = list()
conceptTf = dict()
conceptTfIdf = dict()

totalKey=0
matched=0
matchedDouble = 0
matchedTriple = 0
matchedTotal=0
predicated=0
predicatedTotal=0

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

#json for knowledge base graph
#map<string, map<string,double> > jsonRelMap;
jsonRelMap=dict()
#map<string,string> jsonMap;
jsonMap=dict()


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
    global  document,docIndices,posTags,docLen,position
    obj = open(file, 'r')
    lines = obj.readlines()
    for x in lines:
        posLength = 0

        # x.replace("\t","")
        # x.replace("\r", "")
        # x.replace("\n", "")
        # token=x.split(" ")
        matches = [(m.group(0), (m.start(), m.end())) for m in re.finditer(r'\S+', x)]
        # matches = [m.group(0) for m in re.finditer(r'\S+', x)]
        # remove empty string
        # token=filter(None,token)
        if len(matches)>0:
            for i in range(0, len(matches)):
                index=matches[i][0].rfind('/')
                word = matches[i][0][:index].strip().lower()
                pos = matches[i][0][index+1:]
                posLength += (len(pos)+1)
                indices = (matches[i][1][0]-posLength,matches[i][1][1]-posLength)
                # print(word,pos,indices)


                if (index==-1) or (len(word)>=2 and (word[0]=="<" or word[len(word)-1]==">")):
                    continue
                if(version==1):
                    #store
                    document.append(word)
                    posTags.append(pos)
                    docIndices.append(indices)
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
    global document, posTags, goldKeyMap, goldKey, srGraph, srScore, position,docLen,nodeCount,tagmeTags
    del document[:]
    del posTags[:]
    # del tagmeTags[:]
    tagmeTagList.clear()
    goldKeyMap.clear()
    idEntityMap.clear()
    conceptTf.clear()
    conceptTfIdf.clear()
    del goldKey[:]
    del docIndices[:]
    srGraph.clear()
    srScore.clear()
    jsonMap.clear()
    jsonRelMap.clear()
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
                            srGraph[word][neighbor]+= 1
                        else:
                            srGraph[word][neighbor]= 1
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
        if (k in position):
            srScore[k]=1.0

def singleRank():
    global  srScore,iteration, nodeCount,srGraph,ldaModel
    initialize()
    for it in range(1,iteration+1):
        srScoreTemp = dict()
        for key,value in srScore.items():
            # for topic, val in value.items():
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


def tagCandidate(candClause):
    global posTags, document
    count = 0
    # string list
    pattern = list()
    # string list
    patternPOS = list()
    patternTagMe = list()

    for i in range(0,len(posTags)):
        if isGoodPOS(posTags[i]):
            pattern.append(document[i])
            patternPOS.append(posTags[i])

            if i in tagmeTagList:
                patternTagMe.append(tagmeTagList[i])
                # patternTagMe.append(idEntityMap[tagmeTagList[i]])
            else:
                patternTagMe.append("null")
        elif (len(pattern)>0 and not isGoodPOS(posTags[i])):
            s = ""
            for j in range(0, len(pattern)):
                s += pattern[j]
                if j + 1 < len(pattern):
                    s += " "
            if patternPOS[len(patternPOS) - 1] != "JJ":
                if s not in candClause:
                    count += 1
                    patternTmp = list(patternTagMe)
                    patternList = list()
                    patternList.append(patternTmp)
                    candClause[s] = patternList
                else:
                    if patternTagMe not in candClause[s]:
                        # print(s, candClause[s], patternTagMe, "different!")
                        patternTmp = list(patternTagMe)
                        patternList = list(candClause[s])
                        patternList.append(patternTmp)
                        candClause[s] = patternList



            del pattern[:]
            del patternPOS[:]
            del patternTagMe[:]

    if len(pattern)>0:
        s = ""
        for j in range(0, len(pattern)):
            s += pattern[j]
            if j + 1 < len(pattern):
                s += " "
        if patternPOS[len(patternPOS) - 1] != "JJ":
            if s not in candClause:
                count += 1
                patternTmp = list(patternTagMe)
                patternList = list()
                patternList.append(patternTmp)
                candClause[s] = patternList
            else:
                if patternTagMe not in candClause[s]:
                    # print(s, candClause[s], patternTagMe, "different!")
                    patternTmp = list(patternTagMe)
                    patternList = list(candClause[s])
                    patternList.append(patternTmp)
                    candClause[s] = patternList

        del pattern[:]
        del patternPOS[:]
        del patternTagMe[:]


    return count



# candClause  <string , int>
def extractPatternsTagMe(candClause):
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
            begin=len(pattern)
            end=0
            for key, val in jsonMap.items():
                keyList = key.split(" ")
                flag=False
                if keyList[0] in pattern:
                    thisBegin = pattern.index(keyList[0])
                    if thisBegin+len(keyList)<=len(pattern):
                        flag = True
                        if len(keyList)>1:
                            for ikey in range(1,len(keyList)):
                                if keyList[ikey]!=pattern[thisBegin+ikey]:
                                    flag = False
                if flag:
                    thisEnd=thisBegin+len(keyList)
                    if  thisBegin < begin:
                        begin = thisBegin
                    if thisEnd > end:
                        end = thisEnd
                    # print(pattern, key,thisBegin,thisEnd)
            if begin<=end:
                for j in range(begin, end):
                    s+=pattern[j]
                    if j+1< end:
                        s+=" "
                if patternPOS[end-1]!="JJ":
                    if s not in candClause:
                        count+=1
                        candClause[s]=1
                    else:
                        candClause[s]+=1

            del pattern[:]
            del patternPOS[:]

    if len(pattern)>0:
        s = ""
        begin = len(pattern)
        end = 0
        for key, val in jsonMap.items():
            keyList = key.split(" ")
            flag = False
            if keyList[0] in pattern:
                thisBegin = pattern.index(keyList[0])
                if thisBegin + len(keyList) <= len(pattern):
                    flag = True
                    if len(keyList) > 1:
                        for ikey in range(1, len(keyList)):
                            if keyList[ikey] != pattern[thisBegin + ikey]:
                                flag = False
            if flag:
                thisEnd = thisBegin + len(keyList)
                if thisBegin < begin:
                    begin = thisBegin
                if thisEnd > end:
                    end = thisEnd
                # print(pattern, key, thisBegin, thisEnd)
        if begin <= end:
            for j in range(begin, end):
                s += pattern[j]
                if j + 1 < end:
                    s += " "
            if patternPOS[end - 1] != "JJ":
                if s not in candClause:
                    count += 1
                    candClause[s] = 1
                else:
                    candClause[s] += 1


        del pattern[:]
        del patternPOS[:]

    return count


# candClause  <string , int>
def extractPatternsPos(candClause):
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
            s = ""
            for j in range(0, len(pattern)):
                s += pattern[j]
                if j + 1 < len(pattern):
                    s += " "
            if patternPOS[len(patternPOS) - 1] != "JJ":
                if s not in candClause:
                    count += 1
                    candClause[s] = 1
                else:
                    candClause[s] += 1
            del pattern[:]
            del patternPOS[:]

    if len(pattern)>0:
        s = ""
        for j in range(0, len(pattern)):
            s += pattern[j]
            if j + 1 < len(pattern):
                s += " "
        if patternPOS[len(patternPOS) - 1] != "JJ":
            if s not in candClause:
                count += 1
                candClause[s] = 1
            else:
                candClause[s] += 1

        del pattern[:]
        del patternPOS[:]

    return count


def getTotalScore(s,fileIndex):
    global  srScore
    tokens=s.split(" ")
    d=0.0
    for i in range(0,len(tokens)):
        if(tokens[i] in srScore):
            d+= srScore[tokens[i]]

    # d = d/float(len(tokens))
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

def groupTagMe(candClause,clauseCount,conceptMap_sorted,file,fileIndex):
    obj = open((outputDir.replace("Out","Grp") + file).rstrip().replace(".pos", ".grp"), 'w')
    group = dict()
    group_sorted = dict()


    # for clause,junk in candClause.items():
    #     for key, val in jsonMap.items():
    #         wiki_entity = val[1]
    #         # if wiki_entity not in conceptMap_sorted:
    #         #     print(wiki_entity)
    #         if (clause.rstrip().lower()).find((key.rstrip().lower()))!=-1:
    #             if wiki_entity not in group:
    #                 groupItem = list()
    #                 groupItem.append(clause)
    #                 group[wiki_entity]=groupItem
    #             else:
    #                 group[wiki_entity].append(clause)

    for clause,patternList in candClause.items():
        for patterns in patternList:
            for wikiId in patterns:
                if wikiId in idEntityMap:
                    # wiki_entity = idEntityMap[wikiId]
                    # print(wiki_entity,clause)
                    if wikiId not in group:
                        groupItem = list()
                        groupItem.append(clause)
                        group[wikiId]=groupItem
                    else:
                        group[wikiId].append(clause)

    # conceptMap_filtered = list()
    concept_tobeSort = dict()

    for key,val in group.items():
        concept_tobeSort[key] = len(group[key])

    concept_sortGroup = sorted(concept_tobeSort, key=concept_tobeSort.get, reverse=True)


    for conceptId in concept_sortGroup:
        # if (conceptId in group):

            # if (len(group[conceptId]) > 1):
                # conceptMap_filtered.append(conceptId)
        obj.write("\n"+str(conceptMap_sorted.index(conceptId))+"  "+idEntityMap[conceptId]+"  "+str(conceptTfIdf[conceptId])+"   :\n")
        # if conceptId in group:
        cptList = dict()
        #     = group[concept]
        #

        for clauseItem in group[conceptId]:
            cptList[clauseItem] = getTotalScore(clauseItem,fileIndex)
            # cptList[clauseItem] = clauseCount[clauseItem]
        cptList_sorted = sorted(cptList, key=cptList.get, reverse=True)
        group_sorted[conceptId]=cptList_sorted
        for clauseSorted in cptList_sorted:
            if (isGoldKey(clauseSorted) or isGoldKey(clauseSorted + "s") or (clauseSorted[len(clauseSorted) - 1] == 's' and isGoldKey(clauseSorted[:len(clauseSorted) - 1]))):
                obj.write("\t ----" + clauseSorted + "\n")
            else:
                obj.write("\t"+clauseSorted+"\n")




    return group_sorted,concept_sortGroup










def score(file,fileIndex):

    global  predicated,matched,keyCount,matchedDouble,matchedTriple,matchedTotal,predicatedTotal
    # conceptMap_sorted = extractConcept(file)
    extractConcept(file)
    obj = open((outputDir+file).rstrip().replace(".pos",".phrases"), 'w')

    # <string, int>
    candClause=dict()
    clauseCount=dict()

    topKey=list() # string list
    topKeyVal=list() # double list

    extractPatternsPos(clauseCount)
    # extractPatternsTagMe(candClause)
    count = tagCandidate(candClause)

    # for key,val in candClause.items():
    #     print(key,val)

    conceptMap_sorted_tfidf = computeCptTfIdf()

    group_sorted,conceptMap_filtered = groupTagMe(candClause,clauseCount,conceptMap_sorted_tfidf,file,fileIndex)



    # for key, val in candClause.items():
    #     if val>0:
    #         s=key
    #         score=getTotalScore(s,fileIndex)
    #         # if( len(topKey)< int(keyCount)):
    #         topKey.append(s)
    #         topKeyVal.append(score)
    #         # else:
    #         #     topKey, topKeyVal=replaceLowest(topKey,topKeyVal,s,score)

    for key, val in candClause.items():
        s=key
        score = getTotalScore(s, fileIndex)
        if s not in topKey:
            topKey.append(s)
            topKeyVal.append(score)
        # else:
        #     topKeyVal[topKey.index(s)] += score *10



    cptThreshold = 20
    clauseThreshold = 5

    clauseCountGroup = dict()


    # for iCpt in range(0,min(cptThreshold,len(conceptMap_sorted))):
    for iCpt in range(0, min(cptThreshold, len(conceptMap_filtered))):
    # for iCpt in range(0, min(cptThreshold, len(conceptMap_sorted_tfidf))):
        concept = conceptMap_filtered[iCpt]
        if concept in group_sorted:
            clauseList = group_sorted[concept]
            for iClause in range(0, min(clauseThreshold,len(clauseList))):
                sCpt = clauseList[iClause]
                topKeyVal[topKey.index(sCpt)] *= (1 + 1/(iCpt+1))*(1 + 1/(iClause+1))
                if sCpt in clauseCountGroup:
                    clauseCountGroup[sCpt] +=1
                else:
                    clauseCountGroup[sCpt] = 1

    for keyClause,valClause in clauseCountGroup.items():
        topKeyVal[topKey.index(keyClause)] *= 2






    topKey = [x for (y,x) in sorted(zip(topKeyVal,topKey),reverse=True)]

    predicated+= min(len(topKey),int(keyCount))
    predicatedTotal+=len(topKey)

    for j in range(0, len(goldKey)):
        obj.write(goldKey[j]+"\n")
    obj.write("\n\n\n")

    thisMatched = 0
    thisMatchedDouble = 0
    thisMatchedTriple = 0
    thisMatchedTotal = 0

    for i in range(0,len(topKey)):
        if i==int(keyCount):
            obj.write("\n\n\n")

        pkey=topKey[i]
        # obj.write(pkey + "\n")
        if(isGoldKey(pkey) or isGoldKey(pkey+"s") or (pkey[len(pkey)-1]=='s' and isGoldKey(pkey[:len(pkey)-1]))):
            thisMatchedTotal+=1
            if i < int(keyCount) * 3:
                thisMatchedTriple += 1
                if i<int(keyCount) * 2:
                    thisMatchedDouble += 1
                    if i<int(keyCount):
                        thisMatched += 1
            obj.write("---  "+pkey + "\n")
        else:
            obj.write(pkey + "\n")


    matched+=thisMatched
    matchedDouble += thisMatchedDouble
    matchedTriple += thisMatchedTriple
    matchedTotal+=thisMatchedTotal

def textExtract(doc):
#	print("Currently loading " + doc)
	result = ""
	with open(doc) as d:
		for line in d:
		    result += line
        #print(result)
	return result

#read documents in certain directory
def computLdaOld(fileList):
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

'''knowledge graph base start'''
def countRelOld(str1, str2):
    global jsonRelMap
    result=0.0
    spots1=list()
    spots2=list()
    for key,value in jsonMap.items():
        spot=key
        if spot.lower().find(str1.lower())!=-1:
            spots1.append(value)
        if spot.lower().find(str2.lower())!=-1:
            spots2.append(value)

    for k in range(0,len(spots1)):
        for l in range(0,len(spots2)):
            if jsonRelMap.has_key(spots1[k][0]):
                if jsonRelMap[spots1[k][0]].has_key(spots2[l][0]):
                    result+=jsonRelMap[spots1[k][0]][spots2[l][0]]*float(spots1[k][1])*float(spots2[l][1])
    return result



def computeCptTfIdf():
    global conceptTf,conceptTfIdf,files
    conceptDocs = dict()




    for iDoc in range(0, docCount):
        if JsonDir != "":
            path = JsonDir.rstrip() + files[iDoc].replace(".pos", ".json")
            with open(path.rstrip()) as data_file:
                data = json.load(data_file)
            for iTagMe in range(0, len(data["tagme"])):
                wikiId = str(data["tagme"][iTagMe]["wiki_id"])

                if wikiId not in conceptDocs:
                    docItem = list()
                    docItem.append(files[iDoc])
                    conceptDocs[wikiId] = docItem
                else:
                    if iDoc not in conceptDocs[wikiId]:
                        conceptDocs[wikiId].append(files[iDoc])


    for key,val in conceptTf.items():
        # print(idEntityMap[key],conceptDocs[key])
        conceptTfIdf[key] = float(conceptTf[key])*float(docCount)/float(len(conceptDocs[key]))

    conceptMap_sorted_tfidf = sorted(conceptTfIdf, key=conceptTfIdf.get, reverse=True)

    return conceptMap_sorted_tfidf



def readJsonMap(file):
    global jsonMap,docIndices,document
    spots1 = list()
    spots2 = list()
    # nomatch = 0
    # countIndexmatch = 0
    with open(file) as data_file:
        data = json.load(data_file)
    for i in range(0, len(data["tagme"])):
        wikiId=str(data["tagme"][i]["wiki_id"])
        wikiEntity=str(data["tagme"][i]["wiki_title"])
        idEntityMap[wikiId]=wikiEntity
        conceptScore=0.0



        for j in range(0,len(data["tagme"][i]["annotations"])):
            spot=str(data["tagme"][i]["annotations"][j]["spot"])
            score= str(data["tagme"][i]["annotations"][j]["score"])
            jsonMap[spot]=(wikiId,wikiEntity,score)
            begin=int(data["tagme"][i]["annotations"][j]["begin"])
            end = int(data["tagme"][i]["annotations"][j]["end"])




            for l in range(0,len(docIndices)):
                if docIndices[l][0] in range(begin-20,begin+20):
                    # countIndexmatch += 1
                    # print(spot,begin)
                    # tokenSpot = spot.split(" ")
                    matches = [(m.group(0), (m.start(), m.end())) for m in re.finditer(r'\S+', spot)]
                    tokenSpot = matches[0]
                    # print(matches)
                    flag = False
                    for deviation in range(0,min(20,len(document) - l)):
                        if document[l + deviation].lower() == tokenSpot[0].lower() and flag == False:
                            # print(deviation)
                            docIndices[l + deviation] = (docIndices[l][0],docIndices[l][1])
                            tagmeTagList[l + deviation] = wikiId
                            # if len(matches)>1 and document[l + deviation+1].lower() != matches[1][0].lower():
                                # nomatch += 1
                            flag = True
                    for deviation in range(0, max(-20,-l),-1):
                        if document[l + deviation].lower() == tokenSpot[0].lower() and flag == False:
                            # print(deviation)
                            docIndices[l + deviation] = (docIndices[l][0], docIndices[l][1] + deviation)
                            tagmeTagList[l + deviation] = wikiId
                            # if len(matches)>1 and document[l + deviation+1].lower() != matches[1][0].lower():
                                # nomatch += 1
                            flag = True
                    # if flag==False:
                        # nomatch += 1

    # print(nomatch)
    # print("countIndexMatch",countIndexmatch)
    #
    # print(len(tagmeTagList),len(docIndices),len(jsonMap))
                    # print(document[l-4],document[l-3],document[l-2],document[l-1],document[l],document[l+1],document[l+2],document[l+3],document[l+4])
            # for k in range(begin,end+1):
            #     tagmeTags[k]=wikiId
            #     print(begin,end,spot)


def readRelMap(file):
    global jsonRelMap
    with open(file) as data_file:
        data = json.load(data_file)
    for i in range(0, len(data["relatedness"])):
        src_wiki_id = str(data["relatedness"][i]["src_wiki_id"])
        dst_wiki_jd = str(data["relatedness"][i]["dst_wiki_id"])
        relatedness = float(data["relatedness"][i]["score"])
        if not jsonRelMap.has_key(src_wiki_id):
            jsonRelMap[src_wiki_id]=dict()
        jsonRelMap[src_wiki_id][dst_wiki_jd] = relatedness

#
def getBoostOld(topKey, s):
    score_boost = len(topKey) * 0.1
    for i in range(0,len(topKey)):
        tokenskey=topKey[i].split(" ")
        for ikey in range(0, len(tokenskey)):
            tokens = s.split(" ")
            for istr in range(0, len(tokens)):
                relation = countRel(tokenskey[ikey], tokens[istr])
                score_boost += relation
    return score_boost
'''end'''


def keyTagRate():
    global matched
    for i in range(0, len(goldKey)):
        flag = False
        for j,val in jsonMap.items():
            if goldKey[i]==j or goldKey[i]==(j + "s") or (j[len(j) - 1] == 's' and goldKey[i]==(j[:len(j) - 1])):
                flag = True
        if flag:
            matched += 1




def main(arg):
    '''
    :param arg:
    :return:
    '''
    print "Reading params..."
    #reading params
    global  fileList, goldKeyList,fileDir,goldKeyDir,outputDir,keyCount,windowSize,matched,matchedDouble,matchedTriple,matchedTotal,predicated,predicatedTotal,totalKey,ldaModel,JsonDir,JsonRelDir
    fileList, goldKeyList, fileDir, goldKeyDir, outputDir, keyCount, windowSize,JsonDir,JsonRelDir=readParams(arg)

    global  files
    #read document
    readFiles(1,fileList)
    # ldaModel = computLda(files)
    #read gold key file list
    if goldKeyList!="":
        readFiles(2,goldKeyList)

    for i in range(0,docCount):
    # for i in range(0, 1):
        clean()
        txt=""
        txt=fileDir+files[i]

        print "Processing ",i, files[i], " ..."
        readTxtFiles(1,txt.rstrip(),files[i])
        #readTxtFiles(1,(fileDir+files[0]).rstrip(),files[0])
        key=""
        if goldKeyDir!="" and i<len(keyList):
            key=goldKeyDir+keyList[i]
            readGoldKey(key.rstrip(),i)

        if JsonDir!="":
            path=JsonDir.rstrip()+files[i].replace(".pos",".json")
            readJsonMap(path.rstrip())

        if JsonRelDir != "":
            path = JsonRelDir.rstrip() + files[i].replace(".pos",".json")
            readRelMap(path.rstrip())

        extractConcept(files[i])
        buildGraph()
        singleRank()

        #score each candidate phrase and output top-scoring phrases
        score(files[i],i)

        # for key,val in tagmeTagList.items():
        #     print(document[key],idEntityMap[val])

    objresult = open((outputDir+"0_result").rstrip(), 'w')


    print "---"*30

    p=(100.0*matched)/(1.0*predicated)
    r=(100.0*matched) / (1.0*totalKey)
    f= (2.0*p*r)/(p+r)
    pt=(100.0*matchedTotal)/(1.0*predicatedTotal)

    print "matched= ",matched
    print "predicated= ",predicated
    print "totalKey= ",totalKey
    print "\n"

    print "Recall= ", r
    print "Precision= ", p
    print "F-Score= ", f
    print "\n"


    print "matchedDouble= ", matchedDouble
    print "matchedTriple= ", matchedTriple
    print "matchedTotal= ", matchedTotal
    print "predicatedTotal= ", predicatedTotal
    print "PrecisionTotal= ",pt

    print "done"

    objresult.write("matched= "+str(matched)+"\n")
    objresult.write("predicated= "+str(predicated)+"\n")
    objresult.write("totalKey= "+str(totalKey)+"\n\n")
    objresult.write("Recall= "+str(r)+"\n")
    objresult.write("Precision= "+str(p)+"\n")
    objresult.write("F-Score= "+str(f)+"\n\n")
    objresult.write("matchedDouble= " + str(matchedDouble) + "\n")
    objresult.write("matchedTriple= " + str(matchedTriple) + "\n")
    objresult.write("matchedTotal= " + str(matchedTotal) + "\n")
    objresult.write("predicatedTotal= " + str(predicatedTotal) + "\n")
    objresult.write("PrecisionTotal= "+str(pt)+"\n")



def extractConcept(fileIn):
    global JsonDir,JsonRelDir,outputDir,conceptTf

    conceptMap = dict()


    if JsonRelDir != "":
        pathRel = JsonRelDir.rstrip() + fileIn.replace(".pos",".json")
        with open(pathRel.rstrip()) as rel_file:
            dataRel = json.load(rel_file)



    if JsonDir!="":
        path=JsonDir.rstrip()+fileIn.replace(".pos",".json")
        with open(path.rstrip()) as data_file:
            # data_file = data_file.encode('utf-8')
            data = json.load(data_file)
        for l in range(0, len(data["tagme"])):
            wikiTitle = str(data["tagme"][l]["wiki_title"])
            wikiId=str(data["tagme"][l]["wiki_id"])
            conceptCount = 0
            relatedScore = 0.0
            for j in range(0, len(data["tagme"][l]["annotations"])):
                # score = float(data["tagme"][l]["annotations"][j]["score"])
                conceptCount += 1
            #
            # for k in range(0, len(dataRel["relatedness"])):
            #     src_wiki_id = str(dataRel["relatedness"][k]["src_wiki_id"])
            #     dst_wiki_jd = str(dataRel["relatedness"][k]["dst_wiki_id"])
            #     relatedness = float(dataRel["relatedness"][k]["score"])
            #     if wikiId == src_wiki_id:
            #         relatedScore += relatedness

            # conceptMap[wikiTitle] = conceptCount * relatedScore
            # conceptMap[wikiTitle] = conceptCount
            conceptMap[wikiId] = conceptCount

    conceptMap_sorted = sorted(conceptMap, key=conceptMap.get, reverse=True)

    for key,val in conceptMap.items():
        Tf = val
        if Tf < 1:
            Tf = 0
        conceptTf[key] = float(Tf)/float(len(conceptMap))




    file = (outputDir + fileIn).rstrip().replace(".pos",".cpt")
    # obj = open(file, 'w')
    # for key in conceptMap_sorted:
    #     obj.write(key+"\t\t"+str(conceptMap[key])+"\n")
        # print key
    return conceptMap_sorted





if __name__ == '__main__':
    main('conf.txt')
