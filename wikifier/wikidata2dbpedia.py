import pandas as pd
import glob
import re
import os	
import json
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON
from multiprocessing import Pool

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparqldb = SPARQLWrapper("http://dbpedia.org/sparql")

class convertURI():
    def __init__(self):
        try:
            with open('URICache.json', 'r') as fp:
                self.URIMapCache = json.loads(json.load(fp))
        except:
            self.URIMapCache = {}

    def getDBPediaFromWikiData(self, QNode):
        try:
            sparql.setQuery("""
            SELECT ?article WHERE {{
                OPTIONAL {{
                ?article schema:about wd:{} .
                ?article schema:inLanguage "en" .
                FILTER (SUBSTR(str(?article), 1, 25) = "https://en.wikipedia.org/")
                }}
            }} 
            """.format(QNode))
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            wiki = results['results']['bindings'][0]['article']['value'].split('/')[-1]
            wiki = re.sub(r'([()])', r'\\\1', wiki)
            sparqldb.setQuery("""
            select ?dbpedia where {{?dbpedia foaf:isPrimaryTopicOf wikipedia-en:{}}} LIMIT 1
            """.format(wiki))
            sparqldb.setReturnFormat(JSON)
            results = sparqldb.query().convert()
            out = results['results']['bindings'][0]['dbpedia']['value']
            return out
        except:
            return

    def wiki_to_db(self, qnode):
        sparqldb.setQuery("select ?x where {?x <http://www.w3.org/2002/07/owl#sameAs> <http://www.wikidata.org/entity/"+qnode+">}")
        sparqldb.setReturnFormat(JSON)
        results = sparqldb.query().convert()
        dbp = None
        for result in results["results"]["bindings"]:
            dbp = result['x']['value']
        return(dbp)

    def toDB(self, QNode):
        if QNode in self.URIMapCache:
            return self.URIMapCache[QNode]
        result = [self.getDBPediaFromWikiData(QNode), self.wiki_to_db(QNode)]
        ret = result[0] or result[1]
        self.URIMapCache[QNode] = ret
        return ret

    def processFile(self, fileName):
        '''Convert entire file from Wikidata to DBPedia'''
        df = pd.read_csv("Results/Wikidata Output/"+fileName, names=["FileName", "Col", "Row", "Result"])
        df = df[~df['Result'].isnull()]
        df['Result'] = df['Result'].apply(self.toDB)
        df = df[~df['Result'].isnull()]
        df[['FileName','Col','Row','Result']].to_csv('Results/DBPedia Output/Result_' + fileName.split('.')[0].split('/')[-1]+'.csv',header=None, index=False)
        json.dump(json.dumps(self.URIMapCache),open('URICache.json','w'))

    def convert(self, QNode):
        '''Convert QNode to URI'''
        return self.toDB(QNode)

if __name__ == '__main__':
    c = convert()
    file_list = [f for f in os.listdir("Results/Wikidata Output/") if f.endswith('.csv')]
    totalFiles = len(file_list)
    print("Processing " + str(totalFiles) + " files...")

    for file in tqdm(file_list):
        c.processFile(file)