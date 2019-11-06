import os
import sys
import json
import glob
import math
import ntpath
import requests
from tqdm import tqdm
import pandas as pd
from ast import literal_eval
from scipy.stats import entropy
from SPARQLWrapper import SPARQLWrapper, JSON

#sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql = SPARQLWrapper("http://dsbox02.isi.edu:8888/bigdata/namespace/wdq/sparql")

requestId = str(sys.argv[1])

# Returns a dictionary with key=QNode and value=list of properties
def getKeyPropertiesMapping(QNodes):
    headers = {'Content-Type': 'application/json'}
    data = json.dumps(QNodes)
    responseText = requests.post('http://minds03.isi.edu:4444/get_properties', headers=headers, data=data).text
    return literal_eval(responseText)


def getCandidatesDataLabel(Labels):
    with open('CandidateData.json') as fp:
        for label in Labels:
            pass


def get_qnodes(search_term):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "format": "json",
        "language": "en",
        "limit": "max"
    }
    params["srsearch"] = search_term
    r = requests.get(url, params=params)
    if r.ok:
        res = r.json()
        newres = res.get('query', [])
        qnodes = [x['title'] for x in newres.get("search", [])]
    return qnodes


# QNodes has a set of candidates for each row. Could also be an empty list
def getCandidatesDF(fileName, columnName):
    print("Getting candidates df..")
    df = pd.read_csv(fileName)
    df['QNodes'] = df[columnName].apply(get_qnodes)
    df['Row'] = df.index + 1
    df['Col'] = df.columns.get_loc(columnName)
    df['File'] = fileName.split('.csv')[0].split('/')[-1]
    return df[['QNodes', 'Row', 'Col', 'File']]


# Get all Candidates in QNodes column and query redis for their properties
# mapping is a dictionary QNode -> list of properties
def getAllQNodes(df):
    # print("Getting all QNodes..")
    collectQNodes = df['QNodes'].apply(pd.Series).stack().drop_duplicates().tolist()
    return collectQNodes


# Populate allProps with union of all properties for all QNodes in column QNodes
def getAllProperties(collectQNodes, nodePropMap):
    # print("Getting all properties..")
    allProps = []
    for QNode in collectQNodes:
        allProps += nodePropMap[QNode]
    allProps = set(allProps)
    return allProps


# Attach a column with a list of list having the property list for each QNode candidate for the row
def getRowProperties(QNodes, nodePropMap):
    properties = [nodePropMap[QNode] for QNode in QNodes]
    return properties


def attachAllProperties(df, nodePropMap):
    # print("attaching all properties..")
    df['QNodesProperties'] = df['QNodes'].apply(getRowProperties, args=(nodePropMap,))
    return df


# Get the union of all properties within a given row. Attach it to the dataframe
def union(QNodesProperties):
    return list(set().union(*QNodesProperties))


temp = []


def getUnion(df):
    # print('Getting union of each row...')
    df['Union'] = df['QNodesProperties'].apply(union)
    return df


# Get the intersection of the Union column, ignoring empty unions
def getGlobalIntersection(df, column, threshold=1):
    # print('Getting intersection of all unions...')
    if threshold == 1:
        collect = filter(None, df[column].tolist()[1:])
        globalIntersection = set(df[column][0]).intersection(*collect)
    else:
        total = 0
        collect = filter(None, df[column].tolist())
        occurenceCount = {}
        for row in collect:
            if row:
                total += 1
                for prop in row:
                    occurenceCount[prop] = occurenceCount.get(prop, 0) + 1
        globalIntersection = set()
        for prop in occurenceCount:
            if occurenceCount[prop] / total > threshold:
                globalIntersection.add(prop)
    return globalIntersection


# Select 1 candidate QNode from all possible for a row based on how similar it is to the globalIntersection
def attachIntersectionSelectedCandidate(row, globalIntersection):
    '''
    If the QNodes list was empty, there were no identified candidates.
    Leave result as blank
    '''
    if row['QNodes'] == []:
        return ""

    # To find similarity between a candidate and the globalIntersection:
    # Current metric: Best candidate will have least elements in the set difference A - B
    # A: globalIntersection
    # B: set of properties for the candidate
    scores = [len(list(globalIntersection.difference(set(row['QNodesProperties'][candidate])))) for candidate in
              range(len(row['QNodes']))]

    # Experiment Picker
    selectedQIndex = scores.index(min(scores))

    # Experiment default
    # selectedQIndex = 0

    # Experiment Picker with selected score difference >= 5
    if len(scores) > 1:
        biggerQIndex = scores.index(sorted(scores)[1])

        if selectedQIndex != 0:
            if scores[biggerQIndex] - scores[selectedQIndex] >= 5:  # confidence of candidate picker
                pass
            else:
                selectedQIndex = 0

    selectedQNodePropCount = {}
    selectedProp = row['QNodesProperties'][selectedQIndex]
    for Prop in selectedProp:
        selectedQNodePropCount[Prop] = selectedQNodePropCount.get(Prop, 0) + 1
    return row['QNodes'][selectedQIndex]


def getSelectedQNodeEntropy(selectedQNodePropCount):
    return entropy(list(selectedQNodePropCount.values()))


def attachTfIdfCandidate(df):
    # Need to identify tf and idf for each property
    # Input: df with columns 'QNodesProperties' and 'Union'

    totalRows = df['QNodesProperties'].count()
    totalCandidates = 0
    tf = {}
    idf = {}
    idf2 = {}

    for _, j in df.iterrows():
        for prop in j['Union']:
            tf[prop] = tf.get(prop, 0) + 1

    #     for _,j in df.iterrows():
    #         for candidate in j['QNodesProperties']:
    #             totalCandidates += 1
    #             for prop in candidate:
    #                 idf[prop] = idf.get(prop, 0) + 1
    for _, j in df.iterrows():
        currentIdf = {}
        currentCount = 0
        for candidate in j['QNodesProperties']:
            currentCount += 1
            for prop in candidate:
                currentIdf[prop] = currentIdf.get(prop, 0) + 1
        for prop in currentIdf:
            currentIdf[prop] = log

    tf.update((x, y / totalRows) for x, y in tf.items())
    idf.update((x, math.log(totalCandidates / y, 10)) for x, y in idf.items())

    def scoreQNode(propertyList):
        score = [tf[prop] * idf[prop] for prop in propertyList if prop in tf and prop in idf]
        return sum(score)

    df['result'] = ''
    for index, row in df.iterrows():
        scores = [scoreQNode(row['QNodesProperties'][candidate]) for candidate in range(len(row['QNodes']))]
        if scores:
            selectedQIndex = scores.index(max(scores))
            df.loc[index, 'result'] = row['QNodes'][selectedQIndex]
    return df

def processIntersect(fileName):
    # print(fileName)
    path = "ServiceOutput/" + requestId + "/"
    df = pd.read_csv(path + fileName, names=["File", "Col", "Row", "QNodes"], converters={"QNodes": literal_eval})
    collectQNodes = getAllQNodes(df)
    nodePropMap = getKeyPropertiesMapping(collectQNodes)
    mapping = nodePropMap
    df = attachAllProperties(df, nodePropMap)
    df = getUnion(df)
    globalIntersection = getGlobalIntersection(df, 'Union', 0.8)
    df['result'] = df.apply(attachIntersectionSelectedCandidate, args=(globalIntersection,), axis=1)
    # print(resultFolder+'Output_final_'+file.split('.')[0]+".csv")
    df[['File', 'Col', 'Row', 'result']].to_csv(path + 'Output_' + fileName.split('.')[0] + ".csv",
                                                header=False, index=False)
path = "ServiceOutput/" + requestId + "/"
file_list =  [f for f in os.listdir(path) if f.endswith('.csv') and f.startswith('final_')]
for file in tqdm(file_list):
    processIntersect(ntpath.basename(file))
