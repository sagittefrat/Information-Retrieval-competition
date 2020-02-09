import pandas as pd
import io, json
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

a=pd.read_csv('xml2csv.csv',header=None).values
querynums=[]
querytext=[]
for i in range(0,250,2):
	print a[i+1][0]
	querynums.append(a[i][0])
	querytext.append(a[i+1][0])


QUERIES=dict(zip(querynums, querytext))
with io.open('QUERIES.json', 'w', encoding='utf8') as outfile:
	str_ = json.dumps(QUERIES,
				  indent=4, sort_keys=True,
				  separators=( ',',': '), ensure_ascii=False)
	outfile.write(to_unicode(str_))