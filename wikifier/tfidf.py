import json
import requests
import pandas as pd
from ast import literal_eval
import os
import subprocess
from shutil import copyfile
import pickle
import regex as re
from multiprocessing import Pool
import random
import sys
import math
import itertools
from wikidata2dbpedia import convertURI

from SPARQLWrapper import SPARQLWrapper, JSON
#sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparqldb = SPARQLWrapper("http://dbpedia.org/sparql")
#sparql = SPARQLWrapper("http://sitaware.isi.edu:8080/bigdata/namespace/wdq/sparql")
#sparql = SPARQLWrapper("http://kg2018a.isi.edu:8888/bigdata/namespace/wdq/sparql")
sparql = SPARQLWrapper("http://dsbox02.isi.edu:8888/bigdata/namespace/wdq/sparql")

dataPath = 'input/'
inferPath = 'input/'
mappingFile = 'input/mapping.pkl' # mapping of features to index
CandidateFile = open("CandidateIndex100.json", "r")
CandidateDict = json.loads(CandidateFile.read())
candidate_data = CandidateDict
dbpedia_typeof = 'TypeOf.json'
urimap = 'URICache.json'

try:
    with open(urimap,'r') as fp:
        URIMapCache = json.loads(json.load(fp))
except:
    URIMapCache = {}
    
try:
    with open(dbpedia_typeof,'r') as fp:
        typeOf = json.load(fp)
except:
    typeOf = {}

requestId = str(sys.argv[1])

candpercell = ranks = 100
mask_n = 10
confidence = int(sys.argv[2]) # default = 300
challenge = False
multi = False
debug_weights = True # tf, idf, tf*idf, feature
debug_scores = False # confidence score tf*idf*v for the top candidate

use_wikidata_class = sys.argv[3]=='True' # use class as features   default=True
#use_dbpedia_class = sys.argv[4]=='True' # use class as features   default=True
use_dbpedia_class = False
use_wikidata_props = sys.argv[5]=='True' # default = True
use_dbpedia_props = False # use property as features
use_tf = str(sys.argv[6]) # default 'pedro'
use_df = str(sys.argv[7]) # default 'jay'
norm = False # normalize i.e. divide by number of 1s in the feature vector


def get_all_qnodes(items):
    """
    Gets qnodes of all items
    """
    self_qnodes = {}
    temp_items = []
    for item in items:
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
            CandidateDict[item] = self_qnodes[item]
    items = temp_items[:]
    jsoncand = json.dumps(CandidateDict)
    f = open("CandidateIndex100.json", "w")
    f.write(jsoncand)
    f.close()
    return self_qnodes


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

def get_properties(qids):
    # perhaps cache this too later
    headers = {'Content-Type': 'application/json'}
    data = json.dumps(qids)
    responseText = requests.post('http://minds03.isi.edu:4444/get_properties', headers=headers, data=data).text
    # print(responseText)
    try:
        return literal_eval(responseText)
    except:
        print(responseText)
        exit()
    # dict {"Q160907": ["P3241", "P735", ... "P691", "P570"], "Q..."}


def cache_wikidata_instances(qids, batch=41):
    '''try:
        #a = 2/0 # comment to use cached instances
        instances = pickle.load(open(directory+'instances.pkl','rb'))
        qids_new = []
        for qid in qids:
            if qid not in instances.keys():
                qids_new.append(qid)
        print("Instances: old qids =",len(qids),"new qids =",len(qids_new))
        if len(qids) > 4*len(qids_new): # if less than 1/4th remain, assume they won't be found
            print("no sparqling needed")
            return
        qids = qids_new
    except:
        instances = {}'''
    

    try:
        instances = json.load(open('InstanceOf100.json'))
        qids_new = []
        for qid in qids:
            if qid not in instances:
                qids_new.append(qid)
        qids = qids_new
    except:
        instances = {}   


    for i in range(int(len(qids) / batch) + 1):
        a = i * batch
        b = min((i + 1) * batch, len(qids))
        try:
            instances_temp = download_instances(qids[a:b])
        except:
            print("sparql download failed")
            instances_temp = dict((q, []) for q in qids[a:b])
        instances.update(instances_temp)
    json.dump(instances,open('InstanceOf100.json','w'))
    return instances


def download_instances(qids):
    """
    Gets the instance_of proprety of all qnodes in list. Returns dict of qnode:instance_of
    """
    qids = " ".join(["(wd:{})".format(q) for q in qids])
    sparql.setQuery(
        "select distinct ?item ?class where {{ VALUES (?item) {{ {} }} ?item wdt:P31 ?class .}}".format(qids))
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    instances = {}
    for result in results["results"]["bindings"]:
        qid = result['item']['value'].split("/")[-1]
        cls = result['class']['value'].split("/")[-1]
        if qid in instances:
            instances[qid].append(cls)
        else:
            instances[qid] = [cls]
    return instances  # dict


# select distinct ?x where {{ <http://dbpedia.org/resource/Vatican_City> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?x . ?x <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/2002/07/owl#Class> .}}

def IsInstanceOf(uri):
    print("Getting from isInstance")
    sparqldb.setQuery(
        "select distinct ?x where {{ <{}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?x . ?x <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/2002/07/owl#Class> .}}".format(
            uri))
    sparqldb.setReturnFormat(JSON)
    results = sparqldb.query().convert()
    instances = set()
    for result in results["results"]["bindings"]:
        dbp = result['x']['value']
        instances.add(dbp)

    if len(instances) == 0:
        sparqldb.setQuery(
            "select distinct ?x where {{ <{}> <http://dbpedia.org/ontology/wikiPageRedirects> ?x . }}".format(uri))
        sparqldb.setReturnFormat(JSON)
        results = sparqldb.query().convert()
        for result in results["results"]["bindings"]:
            dbp = result['x']['value']
            return IsInstanceOf(dbp)
        return []
    else:
        return list(instances)


def get_wikidata_instances(cand, instances):
    try:
        return instances[cand]
    except:
        return []


def get_dbpedia_instances(cand):
    try:
        if cand not in URIMapCache:
            URIMapCache[cand] = convertURI().convert(cand)
        uri = URIMapCache[cand]
        return typeOf[uri]
        try:
            return typeOf[uri]
        except:
            typeOf[uri] = IsInstanceOf(uri)
            return typeOf[uri]
    except:
        return []

def read_dataset(filename, column, dataPath=dataPath, trim=True):
    # just read
    # maybe trim out unnecessary columns
    df = pd.read_csv(filename + ".csv")
    df.sample(3)
    if trim == False:
        return df, range(len(df.columns))
    df1 = df[[column]]
    return df1, [list(df.head()).index(column)]

'''def get_super(dir, uris):
    with open('rev_DBClassesClosure.json') as json_file: # all candidates for each cell
        super_data = json.load(json_file)
    supers = dict((k, super_data[k]) for k in uris if k in super_data)
    return supers'''

def read_candidates(df):
    # only for one column
    cand_CEA = []
    qids = []
    for i, row in df.iterrows():
        cand_CEA.append([])
        for j, cell in enumerate(row):
            #cand_CEA[i].append([])
            try:
                cell2 = ''
                for c in cell:
                    if c=='?':
                        cell2+='*'
                    else:
                        cell2+=c
                cand_list = candidate_data[cell2][:ranks]
            except:
                cand_list = []
                #print(cell)
                #continue
            cand_CEA[i].append(cand_list)
            qids.extend(cand_list)
    return cand_CEA,list(set(qids))


def vectorize(dictionary):  # inputs a feat_map
    # print(list(dictionary.keys())[:5], dictionary[list(dictionary.keys())[0]])

    values = list(dictionary.values())
    values = list(itertools.chain(*values))
    values = list(set(values))
    # list of features, eg. all class labels
    dim = len(values)
    print(dim)
    # print(values[:5])

    mapping = dict((v, i) for i, v in enumerate(values))  # {feature1: 0, feature2: 1, ...}
    # what about multiple feature types, eg. properties?
    pickle.dump(mapping, open(mappingFile, 'wb'))

    output = {}  # a dict mapping each qid to a vector
    for qid in dictionary:
        output[qid] = [0] * len(values)
        for feature in dictionary[qid]:
            output[qid][mapping[feature]] = 1
    # return output, mapping, dim
    return output, values
    # mapping = [i[0] for i in sorted(mapping.items(),key=operator.itemgetter(1)) ] # list of feature names sorted
    # wait that's just values


def feat_dbpedia_class(qids):
    # each DBPedia class is a feature
    # note: not using dbpedia ontology
    print("feat_dbpedia_class:")
    feat_maps = dict((q, get_dbpedia_instances(q)) for q in qids)
    return vectorize(feat_maps)


def feat_wikidata_class(qids, instances):
    print("feat_wikidata_class:")
    feat_maps = dict((q, get_wikidata_instances(q, instances)) for q in qids)
    return vectorize(feat_maps)


def feat_wikidata_properties(qids, props):
    # each property is a feature
    print("feat_wikidata_properties:")
    # feat_maps = dict((q, props[q]) for q in qids if q in props else (q,[]) )
    feat_maps = {}
    for q in qids:
        try:
            feat_maps[q] = props[q]
        except:
            feat_maps[q] = []
    return vectorize(feat_maps)


def feat_prop_value(qids):
    print("feat_prop_value:")
    # each property-value pair is a feature


def tf_jay(feats, n):
    # occurrence over any candidate
    # occurrence over 1st candidate
    tfs = [0] * n
    for i, row in enumerate(feats):
        for j, cell in enumerate(row):
            for f in range(n):
                for k, cand in enumerate(cell):
                    if cand[f] == 1:
                        tfs[f] += 1
                        break  # it's good as long as any one candidate has that feature
    return tfs


def tf_pedro(feats, n):
    # occurrence over 1st candidate
    tfs = [0] * n
    for i, row in enumerate(feats):
        for j, cell in enumerate(row):
            for k, cand in enumerate(cell):
                for f, v in enumerate(cand):
                    if v == 1:
                        tfs[f] += 1
                break  # just the 1st candidate should have that feature
    return tfs


def df_jay(feats, n):
    # occurrence over table
    dfs = [0] * n
    for i, row in enumerate(feats):
        for j, cell in enumerate(row):
            for k, cand in enumerate(cell):
                for f, v in enumerate(cand):
                    if v == 1:
                        dfs[f] += 1
    return dfs


'''def df_pedro(feats, n):
    # occurrence over wikidata'''


def make_features(cand_CEA, qids, instances, props):
    # call other functions that make individual feature maps/dicts
    # ASSUME features depend only on qid, not their position in table
    # then for each cell, stack them into a single vector
    print("Make features")
    FS = []  # a list of dicts, each mapping qids to a vector
    MAPPING = []  # list of all feature names
    if use_wikidata_class:
        fs, mapping = feat_wikidata_class(qids, instances)
        FS.append(fs)
        MAPPING.extend(mapping)
    if use_wikidata_props:
        fs, mapping = feat_wikidata_properties(qids, props)
        FS.append(fs)
        MAPPING.extend(mapping)
    if use_dbpedia_class:
        fs, mapping = feat_dbpedia_class(qids)
        FS.append(fs)
        MAPPING.extend(mapping)
    # fs.append(feat_prop_value(qids))

    DIM = len(MAPPING)
    # print(MAPPING)
    print(DIM)
    # return

    feats = []
    # DIM = -1
    for i, row in enumerate(cand_CEA):
        feats.append([])
        for j, cell in enumerate(row):
            feats[i].append([])
            for k, cand in enumerate(cell):
                feats[i][j].append([])
                for f in FS:
                    try:
                        feats[i][j][k].extend(f[cand])
                    except:
                        feats[i][j][k].extend([0] * DIM)
                        # print(i,j,k,cand)
                # DIM = len(feats[i][j][k])

    print("make features:", len(feats), len(feats[0]), len(feats[0][0]), DIM)  # 110, 1, 5, 15
    # n_rows, n_cols, n_candidates per cell, n_features
    return feats, MAPPING  # a 3d array of vectors, n=dim of vectors


def tf(feats, n):
    # call other tf functions as chosen
    # return n dim vector
    if use_tf == 'jay':
        return tf_jay(feats, n)
    elif use_tf == 'pedro':
        return tf_pedro(feats, n)
    else:
        return [1] * n


def idf(feats, n):
    # call other idf functions as chosen
    # return n dim vector
    if use_df == 'jay':
        df = df_jay(feats, n)
    elif use_df == 'pedro':
        df = df_pedro(feats, n)
    else:
        # df = [1]*n
        return [1] * n
    N = 0
    for row in feats:
        for cell in row:
            N += len(cell)
    idf = [math.log(N / df1) for df1 in df]
    # other idf approaches maybe
    return idf


def score(tf, idf, feats, mask):
    # score each allotment using tf and idf over their features
    scores = []  # 3d vector of scalar score for each candidate
    for i, row in enumerate(feats):
        scores.append([])
        for j, cell in enumerate(row):
            scores[i].append([])
            for k, cand in enumerate(cell):
                s = 0
                mod = 0
                for f, v in enumerate(cand):
                    s += tf[f] * idf[f] * v * mask[f]
                    mod += v
                if norm:
                    try:
                        scores[i][j].append(s / mod)
                    except:
                        # division by 0
                        scores[i][j].append(0)
                else:
                    scores[i][j].append(s)
    return scores


def pick(scores, cand_CEA):
    # pick the top one, or maybe even none if low confidence?
    picks = []  # 2d vector of one pick per cell, else -
    for i, row in enumerate(scores):
        picks.append([])
        for j, cell in enumerate(row):
            best = -1
            for k, cand in enumerate(cell):
                if best == -1 or cell[best] < cand:
                    best = k
            if best == -1 or cell[best] < confidence:
                picks[i].append('-')
            else:
                picks[i].append(cand_CEA[i][j][best])
                if debug_scores:
                    print(i, '\t', j, '\t', cand_CEA[i][j][best], '\t', round(cell[best], 2))
    return picks

def make_sub(picks, filename, column, cols, inferPath):
    print(filename, inferPath)
    f = open('ServiceOutput/'+ filename.split('/')[1] + '/Output_final_original_'+column+'.csv','w')
    for i,row in enumerate(picks):
        for j,cell in enumerate(row):
            if cell == '-':
                cell = ''
            f.write(filename.split('/')[-1]+','+str(cols[j])+','+str(i+1)+','+cell+'\n')
    f.close()


def per_file(filename, column, dataPath=dataPath):
    df, cols = read_dataset("ServiceInput/"+filename, column)
    assert len(cols) == 1, "multiple columns?"
    labels = df.iloc[:, 0].tolist()
    print("Getting candidates...")
    allCandidates = get_all_qnodes(labels)
    allQNodes = list(set([QNode for candidate in allCandidates.values() for QNode in candidate]))
    print("Getting props...")
    props = get_properties(allQNodes)
    print("Getting qids")
    cand_CEA, qids = read_candidates(df)
    if use_wikidata_class:
        wd_instances = cache_wikidata_instances(qids)
    # print(list(wd_instances.keys())[:10])
    else:
        wd_instances = []
    feats, mapping = make_features(cand_CEA, qids, wd_instances, props)
    # feats: same format as cand_CEA. 3d array of feature vectors
    # mapping: list of all feature names
    # n: scalar
    n = len(mapping)
    tfs = tf(feats, n)  # n dim vector
    idfs = idf(feats, n)  # n dim vector

    # display weights of each feature, maintain mappings as well
    '''if debug_weights:
        mapping = pickle.load(open(mappingFile, 'rb'))
        for m in mapping:
            print(tfs[mapping[m]],' ',round(idfs[mapping[m]],2),' ',round(tfs[mapping[m]]*idfs[mapping[m]],2),' ',m)'''
    print(len(mapping), len(tfs))

    if debug_weights:
        mask = [0] * n
        # can rearrange these lists now that their work is done (we only need scores)
        tfidf = {}
        for i, m in enumerate(mapping):
            try:
                m = prop_labels[m]
            except:
                _ = True
            tfidf[m] = (round(tfs[i], 2), round(idfs[i], 2), round(tfs[i] * idfs[i], 2), i)
        sorted_tfidf = sorted(tfidf.items(), key=lambda a: a[1][2], reverse=True)
        # a[1][2] for tfidf, a[1][0] for tf, a[1][1] for idf
        for item in sorted_tfidf[:mask_n]:
            print(item)
            mask[item[1][3]] = 1
        # print(sorted_tfidf)
        '''for i,m in enumerate(mapping):
            print(tfs[i],'\t',round(idfs[i],2),'\t',round(tfs[i]*idfs[i],2),'\t',m)'''

    scores = score(tfs, idfs, feats, mask)
    # scores = score(tfs, idfs, feats) # cand_CEA like 3d vector of a scalar score per cell-candidate allotment
    # print(scores)

    picks = pick(scores, cand_CEA)  # pick the best allotment
    make_sub(picks, "ServiceOutput/" + filename, column, cols, inferPath=inferPath)
    json.dump(json.dumps(URIMapCache),open('URICache.json','w'))
    json.dump(typeOf, open('TypeOf.json','w'))

inputPath = "ServiceInput/" + requestId + "/"
outputPath = "ServiceOutput/" + requestId + "/"

file = open(outputPath + 'Files.txt', 'r')
f = file.readlines()
for combo in f:
    print("tfidf: Processing " + combo)
    combo = combo.strip()
    NameCol = combo.split(',')
    print(NameCol[1])
    per_file(requestId+'/'+NameCol[0][:-4], NameCol[1])
