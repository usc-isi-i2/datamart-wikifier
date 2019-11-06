# Accessing the Wikifier endpoint

Wikifier service is running on http://dsbox02.isi.edu:8396

## Basic Command:

Call from CURL:

```
curl -F 'file=@someFile.csv;type=text/csv' http://localhost:8396/wikify > result.csv
```

Call from requests:
```
resp = requests.post(
    'http://localhost:8396/wikify',
    data = {'columns': '{"names":["selected column"]}'},
    files={'file': ('somefile.csv', open('somefile.csv', 'rb'), 
            'text/csv', {'Expires': '0'})}

with open('result.csv', 'wb') as f:
    f.write(resp.content)
)
```

- someFile.csv is the file that we want to wikify
- result will be saved to result.csv

Optional:

1. Specify columns
   Default: Runs on all columns that do not have mostly numbers

   ```
   curl -F 'file=@someFile.csv;type=text/csv' -F 'columns={"names":["Column1"]}' http://localhost:8396/wikify > result.csv
   ```
   
2. Specify wikifyPercentage
   Default: 0.5 - A wikified column is added to result file only if 50% of the rows have been wikified

   ```
   curl -F 'file=@someFile.csv;type=text/csv' -F 'wikifyPercentage=0.7' http://localhost:8396/wikify > result.csv
   ```
   
3. Specify output format (Excel/ISWC)
   Default: Excel - Each wikified column is appended to the right of original columns

   ```
   curl -F 'file=@someFile.csv;type=text/csv' -F 'format=ISWC' http://localhost:8396/wikify > result.csv
   ```