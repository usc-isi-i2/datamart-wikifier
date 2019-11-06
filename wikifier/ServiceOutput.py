import os
import os.path
import sys
import json
import glob
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
from ast import literal_eval
import requests

def getTopKQNodes(qnodes, K):
    return [qnode for qnode in qnodes if qnode not in blackList][:K]

def attach_data(qnodes, data_map):
    return [data_map[qnode] for qnode in qnodes]

def top_k_dict(qnodes, labelsMap, descriptionsMap):
    return_dict = []
    row_top_k = zip(qnodes, [labelsMap[qnode] for qnode in qnodes], [descriptionsMap[qnode] for qnode in qnodes])
    for qnode, label, desc in row_top_k:
        return_dict.append({'qnode':qnode, 'label':label, 'description':desc})
    return return_dict
    
def get_name(qids, queryProperty):
    """
    Get labels of list of qnodes (max 50)
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action":"wbgetentities",
        "format":"json",
        "props":queryProperty,
        "languages":"en",
        "limit":"max",
    }
    params["ids"] = "|".join(qids)
    r = requests.get(url, params=params)
    name = {}
    if r.ok:
        res = r.json()
        for qid in qids:
            name[qid] = res["entities"][qid][queryProperty].get("en",{}).get("value","")
    return name

def get_names(qids, queryProperty):
    """
    Get labels for lists > 50 by sending requests in batches.
    """
    names = {}
    last = 0
    for i in tqdm(range(50, len(qids), 50)):
        names.update(get_name(qids[i-50:i], queryProperty))
        last = i
    names.update(get_name(qids[last:len(qids)], queryProperty))
    return names

def get_column_json(value, Qnode=None, top_k=None):
    ret = {}
    ret['value'] = value
    if Qnode:
        ret['qnode'] = Qnode
    if top_k:
        ret['top_k'] = top_k
    return ret

def formatExcel(fileName, wikifyPercentage, header, K):
    '''
    An extra column with wikidata QNodes is added for every column that we want to wikify
    '''
    # Get Wikidata QNodes for each column
    with open(wikidataOutputPath + 'Files.txt') as fp:
        files = fp.readlines()
    columns = [combo.strip('\n').split(',')[1].replace('$',',') for combo in files]
    #wk = [wikidataOutputPath + 'Output_final_original_' + column + '.csv' for column in columns if os.path.exists('Output_final_original_' + column + '.csv')]
    # Get all required column names
    columnsList = columns
    #for file in wk:
        #columnsList.append(file.split('.')[0].split('_')[-1])
    #for column in columns:
    #    columnsList.append(column)

    # Get all column names from original file 
    orig = pd.read_csv(datasetPath + fileName + '.csv')
    origCols = list(orig)

    # For each required column, attach the wikidata(_WK) column to the original file
    # only if more than wikifyPercentage have been wikified
    usefulCols = []
    for column in columnsList:
        if os.path.exists(wikidataOutputPath + 'Output_final_' + fileName + '_' + column + '.csv'):
            df = pd.read_csv(wikidataOutputPath + 'Output_final_' + fileName + '_' + column + '.csv',
                             names=['File', 'Col', 'Row', 'Wikidata'])
                
            if df['Wikidata'].count() / df['Row'].count() > wikifyPercentage:
                orig[column + '_WK'] = df['Wikidata']
                if K:
                    candidates = pd.read_csv(wikidataOutputPath + 'final_' + fileName + '_' + column + '.csv',
                                 names=['File', 'Col', 'Row', 'QNodes'], converters={"QNodes": literal_eval})
                    candidates['QNodes'] = candidates.apply(lambda x: getTopKQNodes(x.QNodes, K), axis=1)
                    allQNodes = candidates['QNodes'].tolist()
                    allQNodes = [item for sublist in allQNodes for item in sublist]
                    allQNodes = list(set(allQNodes))
                    try:
                        allQNodes.remove('')
                    except:
                        pass
                    allLabels = get_names(allQNodes, 'labels') 
                    allDesc = get_names(allQNodes, 'descriptions')
                    if isReturnJSON:
                        orig[column + '_top_k_dict'] = candidates.apply(lambda x: top_k_dict(x.QNodes, allLabels, allDesc), axis=1)
                        orig[column] = orig.apply(lambda x: get_column_json(x[column], x[column+'_WK'], x[column + '_top_k_dict']), axis=1)
                    else:
                        orig[column + '_labels'] = candidates.apply(lambda x: attach_data(x.QNodes, allLabels), axis=1)                                                                                            
                        orig[column + '_descriptions'] = candidates.apply(lambda x: attach_data(x.QNodes, allDesc), axis=1)  
                        orig[column + '_QNodes'] = candidates['QNodes']
                else:
                    if isReturnJSON:
                        orig[column] = orig.apply(lambda x: get_column_json(x[column], x[column+'_WK']), axis=1)
                usefulCols.append(column)

    if isReturnJSON:
        finalColumns = []
        for column in origCols:
            if column not in usefulCols:
                orig[column] = orig.apply(lambda x: get_column_json(x[column]), axis=1)
            finalColumns.append(column)
        json.dump(orig[finalColumns].to_json(orient='records'), open(outputPath + fileName + '.json','w'))
    else:
        # Ensure the order of columns is correct
        finalColumns = origCols
        for column in origCols:
            if column in usefulCols:
                finalColumns.append(column + '_WK')
                if K:
                    finalColumns.append(column + '_QNodes')
                    finalColumns.append(column + '_labels')
                    finalColumns.append(column + '_descriptions')

        # Write Output
        orig[finalColumns].to_csv(outputPath + fileName + '.csv', index=False)

def formatISWC(fileName, wikifyPercentage, header):
    '''
    Generate the result in the ISWC format (file, col, row, QNode)
    '''
    with open(wikidataOutputPath + 'Files.txt') as fp:
        files = fp.readlines()
    columns = [combo.strip().split(',')[1].replace('$',',') for combo in files]
    wk = [wikidataOutputPath + 'Output_final_original_' + column + '.csv' for column in columns]
    wkDF = [pd.read_csv(wk[i], names=['File', 'Col', 'Row', 'Wikidata']) for i in range(len(wk)) if os.path.exists(wk[i])]

    #remove dataframe if less than wikifyPercentage have been wikified
    wkDF = [df for df in wkDF if df['Wikidata'].count() / df['Row'].count() > wikifyPercentage]

    # Get all required column names
    columnsList = columns
    #for file in wk:
    #    columnsList.append(file.split('.')[0].split('_')[-1])

    # Get all column names from original file 
    orig = pd.read_csv(datasetPath + fileName + '.csv')
    origCols = list(orig)

    # concat all the columns dataframe
    result = pd.concat(wkDF)
    if not header:
        result['Row'] = result['Row'] - 1
    result[['Col','Row','Wikidata']].to_csv(outputPath + fileName + '.csv', index=False, header=None)

if __name__ == '__main__':
    requestId = str(sys.argv[1])
    wikifyPercentage = float(sys.argv[2])
    outputFormat = str(sys.argv[3])
    header = sys.argv[4] == 'True'
    
    # if non 0, return top K qnodes
    K = int(sys.argv[5])
    isBlackList = sys.argv[6] == 'True'
    isReturnJSON = sys.argv[7] == 'True'

    datasetPath = 'ServiceInput/' + requestId + '/'
    outputPath = 'ServiceOutput/' + requestId + '/'
    wikidataOutputPath = outputPath

    if isBlackList:
        blackList = json.load(open(outputPath + 'blacklist.json'))
    else:
        blackList = []

    fileName = 'original'

    if outputFormat == 'ISWC':
        formatISWC(fileName, wikifyPercentage, header)
    elif outputFormat == 'Excel':
        formatExcel(fileName, wikifyPercentage, header, K)
