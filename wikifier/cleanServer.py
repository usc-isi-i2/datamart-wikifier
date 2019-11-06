import io
import os
import re
import csv
import sys
import json
import uuid
import glob
import shutil
import ntpath
import requests
import subprocess
from flask import Flask
from flask import request
from flask import make_response

app = Flask(__name__)

@app.route('/')
def csvWikifier():
        return 'Wikifies an Excel file'

def make_folders(inputPath, outputPath):
    # Create separate input and output folders for every request
    try:
        os.mkdir(inputPath)
        os.mkdir(outputPath)
    except:
        app.logger.error("Couldn't create folders for requestid " + requestId)
        sys.exit(0)
    
def get_columns():
    # Get columns to work on if specified
    columns = request.form.get('columns')
    if columns:
        # columns param was specified
        columns = json.loads(columns)['names']
        columns = list(map(str, columns))
    else:
        # No columns param in request
        app.logger.info("No param columns. Running on all columns..")
        columns = []
    return columns

def get_black_list():
    blackList = request.form.get('blackList')
    if blackList:
        blackList = json.loads(blackList)['QNodes']
        with open(outputPath + 'blacklist.json','w') as fp:
            json.dump(blackList, fp)
        blackList = "True"
    else:
        blackList = "False"
    return blackList

@app.route('/wikify', methods=['POST'])
def upload():
    # Create original.csv with contents from request csv


    requestId = uuid.uuid4().hex
    inputPath = 'ServiceInput/' + requestId + '/'
    outputPath = 'ServiceOutput/' + requestId + '/'

    make_folders(inputPath, outputPath)

    # Get the input file from the request
    with open(inputPath + 'original.csv','wb') as fp:
        fp.write(request.files['file'].read())

    # Get columns to work on if specified
    columns = get_columns()

    blackList = get_black_list()

    # Get wikifyPercentage from request. Column will be in result only if wikifyPercentage of rows have been wikified
    # Default value 0.5
    wikifyPercentage = request.form.get('wikifyPercentage')
    if not wikifyPercentage:
        wikifyPercentage = "0"

    # Get output format from request Excel/ISWC
    # Default value Excel
    formatType = request.form.get('format')
    if not formatType:
        formatType = "Excel"

    retType = request.form.get('retType')
    if not retType:
        retType = 'CSV'

    header = request.form.get('header')
    if not header:
        header = "True"

    K = request.form.get('K')
    if not K:
        K = "0"    

    # Run the pipeline
    approach = request.form.get('approach')
    
    phase = request.form.get('phase')
    if not phase:
        phase = 'train'
    if phase == 'test':
        columnClass = request.form.get('columnClass')
        if not columnClass:                                                                                         phase = 'train'
        else:                                                                                          
             columnClass = json.loads(columnClass)['names']                                            
             columnClass = list(map(str, columnClass))                                                 
             if len(columns)!=len(columnClass):                                                        
                 phase = 'train'
    if phase == 'test':                                                                                
        #get column name: test columnClass mapping                                                     
        classMap = dict(zip(columns, columnClass))                                                     
        json.dump(classMap, open(outputPath+'columnClass.json','w'))

    subprocess.run([sys.executable, "-u","Starter.py", requestId, "original.csv", header] + columns, check=True)
    if not approach:
        subprocess.run([sys.executable, "-u","WikifierService.py", requestId, phase])
        subprocess.run([sys.executable, "-u","picker.py", requestId])
    elif approach=='tfidf':
        confidence = request.form.get('confidence')
        if not confidence:
            confidence = "0"
        use_wikidata_class = request.form.get('use_wikidata_class')
        if not use_wikidata_class:
            use_wikidata_class = "True"
        use_dbpedia_class = request.form.get('use_dbpedia_class')
        if not use_dbpedia_class:
            use_dbpedia_class = "True"
        use_wikidata_props = request.form.get('use_wikidata_props')
        if not use_wikidata_props:
            use_wikidata_props = "True"
        use_tf = request.form.get('use_tf')
        if not use_tf:
            use_tf = "pedro"
        use_df = request.form.get('use_df')
        if not use_df:
            use_df = "jay"
        subprocess.run([sys.executable, "-u", "tfidf.py", requestId, confidence, use_wikidata_class, use_dbpedia_class, use_wikidata_props, use_tf, use_df])
        #print(["tfidf.py", requestId, confidence, use_wikidata_class, use_dbpedia_class, use_wikidata_props, use_tf, use_df])
    subprocess.run([sys.executable, "-u", "ServiceOutput.py", requestId, wikifyPercentage, formatType, header, K, blackList])

    if retType == 'CSV':
        stringIO = io.StringIO()
        writeCSV = csv.writer(stringIO)
        with open(outputPath + 'original.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            writeCSV.writerows(readCSV)
        data = stringIO.getvalue()
    elif retType == 'JSON':
        with open(outputPath + 'original.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
        data = json.loads(json.dumps(rows))
 
    if not approach:
        processedColumns = glob.glob(outputPath+'original_*_chosenclass.csv')
        chosenClass = {}
        for col in processedColumns:
            with open(col) as fp:
                chosenClass[re.match("original_(.*?)_chosenclass.csv",ntpath.basename(col)).group(1)] = fp.readline().strip('\n').split(',')[-1]
        output = {"data":data, "class":chosenClass}
    else:
        output = {"data":data}
    return json.dumps(output)
