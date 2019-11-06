import sys
import pandas as pd

def toomanynum(string):
    numcount=0
    if type(string)==str:
        for i in string:
            if i.isdigit():
                numcount+=1
#        print(float(numcount)/len(string))
        if float(numcount)/len(string) > 0.6:
            return True
    return False


def is_number(n):
    try:
        float(n)   # Type-casting the string to `float`.
                   # If string is not a valid `float`, 
                   # it'll raise `ValueError` exception
    except ValueError:
        return False
    return True

    

idname=str(sys.argv[1])
filename=str(sys.argv[2])
header = sys.argv[3] == 'True'
columns=[]
if header:
    df=pd.read_csv('ServiceInput/'+idname+'/'+filename)
else:
    df=pd.read_csv('ServiceInput/'+idname+'/'+filename, header=None)
f=open('ServiceOutput/'+idname+'/Files.txt','w')

# Do not read in columns from from request if there is no header
if len(sys.argv)>4 and header:
    for i in range (4,len(sys.argv)):
        columns.append(str(sys.argv[i]))
    for col in columns:
        string=filename+','+col.replace(',','$')+'\n'
        f.write(string)

else:
    for i in range(len(df.columns)):
        string=filename+','+str(df.columns[i]).replace(',','$')+'\n'
        count=0
        lis=df[df.columns[i]].tolist()
        for i in lis:
            if type(i)==int or type(i)==float or is_number(i) or i==' ' or i=='-' or toomanynum(i):
                count+=1
        if count/len(lis) <0.8:
            f.write(string)
if not header:
    df.to_csv('ServiceInput/'+idname+'/'+filename, index=False)
