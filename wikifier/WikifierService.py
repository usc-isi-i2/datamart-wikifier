import pandas as pd
import numpy as np
import requests
import os
import sys
import json
import csv
from multiprocessing import Pool
import glob
import time
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON
import re
import math

#sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparqldb = SPARQLWrapper("http://dbpedia.org/sparql")
#sparql = SPARQLWrapper("http://sitaware.isi.edu:8080/bigdata/namespace/wdq/sparql")
sparql = SPARQLWrapper("http://dsbox02.isi.edu:8888/bigdata/namespace/wdq/sparql")
SuperClassFile = open("SuperClass.json", "r")
SuperClassDict = json.loads(SuperClassFile.read())
CandidateFile = open("CandidateIndex100.json", "r")
CandidateDict = json.loads(CandidateFile.read())
InstanceFile = open("InstanceOf100.json", "r")
InstanceDict = json.loads(InstanceFile.read())
NamesFile = open("Names.json", "r")
NamesDict = json.loads(NamesFile.read())
SuperClassFile.close()
CandidateFile.close()
InstanceFile.close()
NamesFile.close()


def to_csv(df, name):
    global self_path
    file_path = os.path.join(outpath, name)
    df.to_csv(file_path)


def get_instances(qids):
    """
    Gets instance of proprety of all qnodes in list. Returns dict of qnode:instance_of
    """
    instances = {}
    qs = qids[:]
    qids = " ".join(["(wd:{})".format(q) for q in qids])
    temp = qs[:]
    for q in qs:
        if q in InstanceDict:
            instances[q] = InstanceDict[q]
            temp.remove(q)
    qs = temp[:]
    if len(qs) == 0:
        return instances
    qstr = " ".join(["(wd:{})".format(q) for q in qs])
    sparql.setQuery(
        "select distinct ?item ?class where {{ VALUES (?item) {{ {} }} ?item wdt:P31|wdt:P106 ?class .}}".format(qstr))
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    for result in results["results"]["bindings"]:
        qid = result['item']['value'].split("/")[-1]
        cls = result['class']['value'].split("/")[-1]
        if cls in bannedset:
            continue
        if qid in instances:
            instances[qid].append(cls)
            InstanceDict[qid].append(cls)
        else:
            instances[qid] = [cls]
            InstanceDict[qid] = [cls]
        if qid in qs:
            qs.remove(qid)
    for q in qs:
        InstanceDict[q] = []
    jsoninst = json.dumps(InstanceDict)
    f = open("InstanceOf100.json", "w")
    f.write(jsoninst)
    f.close()
    return instances


def get_special_search_qnodes(search_term):
    """
    Searches for the search_term and returns a list of candidates from the wikidata search api
    """
    if search_term == '':
        return []
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "format": "json",
        "language": "en",
        "limit": "max"
    }
    params["srsearch"] = search_term
    params["srlimit"] = candpercell
    r = requests.get(url, params=params)
    num_results = 5
    qnodes = []
    if r.ok:
        res = r.json()
        newres = res.get('query', [])
        qnodes = []
        noresult = 1
        # qnodes = [x['title'] for x in newres.get("search",[])]
        try:
            for x in newres.get("search", []):
                noresult = 0
                num_results -= 1
                qnodes.append(x['title'])
                if num_results == 0:
                    break
            if noresult == 1:
                if '-' in search_term:
                    terms = search_term.split('-')
                    for term in terms:
                        qnodes += get_special_search_qnodes(term)
                sterm = re.sub(r" ?\([^)]+\)", "", search_term)
                if sterm != search_term:
                    qnodes = get_special_search_qnodes(sterm)
        except:
            pass
    return (qnodes)

def getElasticQNodes(label):
    label = re.sub(r"\(.+\)","",label)
    query = json.dumps({"query":{"bool":{"filter":[{"terms":{"namespace":[120],"boost":1}}],"must_not":[{"match_phrase":{"descriptions.en.plain":"Wikipedia disambiguation page"}}],"should":[{"query_string":{"query":label,"fields":["all^0.5","all.plain^1.0"],"use_dis_max":True,"tie_breaker":0,"default_operator":"and","auto_generate_phrase_queries":True,"max_determinized_states":10000,"allow_leading_wildcard":True,"enable_position_increments":True,"fuzziness":"AUTO","fuzzy_prefix_length":2,"fuzzy_max_expansions":50,"phrase_slop":0,"rewrite":"top_terms_boost_1024","escape":False,"split_on_whitespace":True,"boost":1}},{"multi_match":{"query":label,"fields":["all_near_match^2.0"],"type":"best_fields","operator":"OR","slop":0,"prefix_length":0,"max_expansions":50,"lenient":False,"zero_terms_query":"NONE","boost":1}}],"disable_coord":False,"adjust_pure_negative":True,"minimum_should_match":"1","boost":1}},"_source":{"includes":["namespace","title","namespace_text","wiki","labels.en","descriptions.en","incoming_links"],"excludes":[]},"stored_fields":"text.word_count","rescore":[{"window_size":8192,"query":{"rescore_query":{"function_score":{"query":{"match_all":{"boost":1}},"functions":[{"filter":{"match_all":{"boost":1}},"field_value_factor":{"field":"incoming_links","factor":1,"missing":0,"modifier":"log2p"}},{"filter":{"terms":{"namespace":[120],"boost":1}},"weight":0.2}],"score_mode":"multiply","max_boost":3.4028235e+38,"boost":1}},"query_weight":1,"rescore_query_weight":1,"score_mode":"multiply"}}],"stats":["suggest","full_text","full_text_querystring"],"size":5})
    HEADERS = {'Content-Type': 'application/json'}
    uri = "http://kg2018a.isi.edu:9200/my_wiki_content_first/_search"
    r = requests.get(uri,headers=HEADERS, data=query).json()
    try:
        return [i['_id'] for i in r['hits']['hits']]
    except:
        return []

def get_all_qnodes():
    """
    Gets qnodes of all items
    """
    global self_items
    global self_qnodes
    self_qnodes = {}
    temp_items = []
    for item in self_items:
        if type(item) == float:
            item = str(item)
        if str(item) == 'nan':
            item = ''
        temp_items.append(item)
        if item in CandidateDict:
            self_qnodes[item] = CandidateDict[item][:candpercell]
        #            if len(self_qnodes[item])==0:
        #                self_qnodes[item]=get_special_search_qnodes(item)
        #                CandidateDict[item] = self_qnodes[item]
        else:
            self_qnodes[item] = getElasticQNodes(item)
            #self_qnodes[item] = get_special_search_qnodes(item)
            CandidateDict[item] = self_qnodes[item]
    self_items = temp_items[:]
    jsoncand = json.dumps(CandidateDict)
    f = open("CandidateIndex100.json", "w")
    f.write(jsoncand)
    f.close()
    return self_qnodes


def get_wiki_df():
    """
    Construct the wikified df. Wikified df is the table that contains all classes as columns and
    candidates as rows with 1 if a candidate is related to a class and 0 if it is not
    """
    global self_wiki
    global self_items
    global self_qnodes
    self_wiki = pd.DataFrame()
    item_done = {}
    SuperClassList = []
    for item in self_items[:100]:
        if item not in item_done:
            item_done[item] = 'done'
            instances = get_instances(self_qnodes[item])
            for q in self_qnodes[item]:
                s = pd.Series()
                s["items"] = item
                s.name = q
                if q not in instances:
                    continue
                related = set()
                for i in instances[q]:
                    if i in SuperClassDict:
                        SuperClassList = SuperClassDict[i]
                    related.update(set(SuperClassList))
                for r in related:
                    s[r] = True
                self_wiki = self_wiki.append(s)
    self_wiki.index.name = "qnode"
    self_wiki = self_wiki.reset_index()
    try:
        self_wiki = self_wiki.set_index(['items', 'qnode'])
    except:
        print("Empty df")
    self_wiki = self_wiki.fillna(0)
    return self_wiki


def chosen_class():
    global self_path
    global self_items
    global self_qnodes
    global self_fileid
    global self_column
    global self_column_name
    global chosen_class_df
    global columnClass
    chosen_class_df = pd.DataFrame()
    item_done = {}
    SuperClassList = []
    self_entities = []
    colname = self_column_name.replace('/', '')
    if not phaseTest or columnClass[colname]=='': 
        name = self_fileid + '_' + colname + '_candidates.csv'
        candidate_df = pd.read_csv(outpath + name)
        column_qnode = candidate_df['class'].tolist()[0]
    else:
        column_qnode=columnClass[colname]
    name = self_fileid + '_' + colname + '_wikified.csv'
    for item in self_items:
        if item not in item_done:
            item_done[item] = 'Done'
            instances = get_instances(self_qnodes[item])
            for q in self_qnodes[item]:
                s = pd.Series()
                s['items'] = item
                s.name = q
                if q not in instances:
                    continue
                related = set()
                for i in instances[q]:
                    if i in SuperClassDict:
                        SuperClassList = SuperClassDict[i]
                    related.update(set(SuperClassList))
                if column_qnode in related:
                    s[column_qnode] = True
                chosen_class_df = chosen_class_df.append(s)
    chosen_class_df.index.name = "qnode"
    chosen_class_df = chosen_class_df.reset_index()
    chosen_class_df = chosen_class_df.set_index(['items', 'qnode'])
    chosen_class_df = chosen_class_df.fillna(0)
    return chosen_class_df


def get_histogram():
    """
    Caluclate subtotals and histogram
    """
    global self_subtotals
    global self_wiki
    global self_his
    try:
        self_subtotals = self_wiki.groupby('items').sum()
    except:
        print("wiki df empty")
        return
    self_his = pd.DataFrame()
    cols = ["0", "1", "2", "3", ">4"]
    for i in range(len(self_subtotals.columns)):
        counts, bins = np.histogram(self_subtotals.iloc[:, i], bins=[0, 1, 2, 3, 4, float('inf')])
        self_his = self_his.append(pd.Series(counts, name=self_wiki.columns[i], index=cols))
    self_his = self_his.sort_values(['1'], ascending=False)
    self_his = self_his / len(self_subtotals)
    qids = self_his.index.tolist()
    names = get_names(qids)
    # names = [names[q] for q in qids]
    namelist = []
    for q in qids:
        if q in names:
            namelist.append(names[q])
        else:
            namelist.append(" ")
    self_his["name"] = namelist
    return self_his


def get_name(qids):
    """
    Get labels of list of qnodes (max 50)
    """
    name = {}
    qs = qids[:]
    tempqs = qs[:]
    for q in qs:
        if q in NamesDict:
            name[q] = NamesDict[q]
            if q in tempqs:
                tempqs.remove(q)
    qs = tempqs[:]
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "format": "json",
        "props": "labels",
        "languages": "en",
        "limit": "max"
    }
    params["ids"] = "|".join(qs)
    r = requests.get(url, params=params)
    if r.ok:
        res = r.json()
        if 'entities' in res:
            for qid in qs:
                try:
                    name[qid] = res["entities"][qid]["labels"].get("en", {}).get("value", "")
                    NamesDict[qid] = name[qid]
                except:
                    pass
    jsonname = json.dumps(NamesDict)
    f = open("Names.json", "w")
    f.write(jsonname)
    f.close()

    return name


def get_names(qids):
    """
    Get labels for lists > 50 by sending requests in batches.
    """
    names = {}
    last = 0
    print("Retreiving names")
    for i in range(50, len(qids), 50):
        names.update(get_name(qids[i - 50:i]))
        last = i
    names.update(get_name(qids[last:len(qids)]))
    return names


def get_result():
    """
    Get the final results. To choose the class.
    """
    global self_result
    global self_his
    self_result = self_his[['1', 'name']][:10]
    self_result = pd.DataFrame(self_result)
    self_result = self_result.rename({'1': 'confidence'}, axis='columns')
    self_result['confidence'] *= 100
    self_result.index.name = "class"
    return self_result


def final_qnodes():
    """
    After choosing a class, Get the correct qnodes from the candidates that are an instance of that class. Do
    a special search with class name appended to the label if you can not find any candidates that are related
    to that class for a particular label
    """
    global self_path
    global self_items
    global self_qnodes
    global self_fileid
    global self_column
    global self_column_name
    global columnClass
    self_entities = []
    colname = self_column_name.replace('/', '')
    if not phaseTest or columnClass[colname]=='':
        name = self_fileid + '_' + colname + '_candidates.csv'
        candidate_df = pd.read_csv(outpath + name)
        column_qnode = candidate_df['class'].tolist()[0]
    else:
        column_qnode=columnClass[colname]

    name = self_fileid + '_' + colname + '_chosenclass.csv'
    wiki_df = pd.read_csv(outpath + name)
    result_list = []
    row = 1
    name = get_name([column_qnode])
    for item in self_items:
        if type(item) == float and math.isnan(item):
            result_list.append([self_fileid, self_column, row, []])
            continue
        item_df = wiki_df[wiki_df['items'] == item]
        possible_df = item_df[item_df[column_qnode] == 1]
        if possible_df.empty:
            # result_list.append([self.fileid,self.column,row,[]])
            # Use the following commented out code if you want to use the wikidata's special search to map items that do not
            # have candidates that are an instance of the chosen class.
            if column_qnode in name:
                search_str = item + ' ' + name[column_qnode]
            else:
                search_str = item
            if search_str in CandidateDict:
                possible_qnodes = CandidateDict[search_str]
            else:
                possible_qnodes = getElasticQNodes(search_str)
                #possible_qnodes = get_special_search_qnodes(search_str) 
                CandidateDict[search_str] = possible_qnodes
            # print(search_str)
            # print(possible_qnodes)
            if len(possible_qnodes) == 0:
                if len(self_qnodes[item]) == 0:
                    result_list.append([self_fileid, self_column, row, []])
                else:
                    self_entities.append(self_qnodes[item][0])
                    result_list.append([self_fileid, self_column, row, [self_qnodes[item][0]]])
                    # result_list.append([self_fileid,self_column,row,[]])
            else:
                self_entities.append(possible_qnodes[0])
                result_list.append([self_fileid, self_column, row, [possible_qnodes[0]]])
        else:
            self_entities.append(possible_df.loc[possible_df.index[0], 'qnode'])
            result_list.append([self_fileid, self_column, row, possible_df['qnode'].tolist()])
        row += 1
        # print(result_list)
    filep = 'final_' + self_fileid + '_' + colname + '.csv'
    with open(outpath + filep, 'w', newline='') as myfile:
        for result_row in result_list:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(result_row)
    jsoncand = json.dumps(CandidateDict)
    f = open("CandidateIndex100.json", "w")
    f.write(jsoncand)
    f.close()
    return self_entities


def wikify(items, path, filename, column, column_name):
    """
    Main function to run everything together.
    """
    global self_wiki
    global self_subtotals
    global self_his
    global self_result
    global self_items
    global self_path
    global self_fileid
    global self_column
    global self_column_name
    global error
    global columnClass
    self_items = items
    self_path = path
    self_fileid = filename
    self_column = column
    self_column_name = column_name
    colname = self_column_name.replace('/', '')
    print("Retrieving all qnodes")
    get_all_qnodes()
    if not phaseTest or columnClass[colname]=='':
        print("Building wikified data")
        get_wiki_df()
        name = self_fileid + '_' + colname + '_wikified.csv'
        to_csv(self_wiki, name)
        print("Calculating histogram")
        get_histogram()
        name = self_fileid + '_' + colname + '_subtotals.csv'
        try: self_subtotals;to_csv(self_subtotals, name)
        except:
            #error
            return
        name = self_fileid + '_' + colname + '_histogram.csv'
        to_csv(self_his, name)
        print("Result")
        print(get_result())
        name = self_fileid + '_' + colname + '_candidates.csv'
        to_csv(self_result, name)
        # print("Wikimap")
        # self.build_wiki_json()
        # json.dump(self.wiki_map, open(os.path.join(self.path, "wiki_map.json"), 'w+'))
    print('Building Wikified data for chosen class')
    chosen_class()
    name = self_fileid + '_' + colname + '_chosenclass.csv'
    to_csv(chosen_class_df, name)
    print('Retrieving entities of items')
    final_qnodes()


def start(combo):
    file = combo[0]
    df = pd.read_csv(path + '/' + file)
    column_name = combo[1]
    column_name = column_name.replace('$', ',')
    filename = file.replace('.csv', '')
    items = df[column_name].tolist()
    newitems = []
    for i in items:
        if type(i) != float and type(i) != int:
            i = i.replace('?', '*')
        newitems.append(i)
    items = newitems[:]
    headers = list(df)
    for i in range(len(headers)):
        if headers[i] == column_name:
            column = i
            break
    #print(items)
    wikify(items, path, filename, column, column_name)


idname = str(sys.argv[1])
print(idname)
phase = str(sys.argv[2])
phaseTest = False
if phase == 'test':
    phaseTest = True
    columnClassFile = open('ServiceOutput/' + idname + '/columnClass.json','r')
    columnClass = json.load(columnClassFile)
    columnClassFile.close()
candpercell = 5  # Number of candidates you want to consider
bannedset = {'Q101352', 'Q4167410', 'Q58494026', 'Q11266439', 'Q4167836'}
path = 'ServiceInput/' + idname
outpath = 'ServiceOutput/' + idname + '/'  # Path to all your outputs
file = open('ServiceOutput/' + idname + '/' + 'Files.txt', 'r')
f = file.readlines()
all_list = []
for combo in f:
    combo = combo.strip('\n')
    NameCol = combo.split(',')
    all_list.append(NameCol)
for i in all_list:
    start(i)
