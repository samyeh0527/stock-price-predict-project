import pandas as pd
import requests
import xlwings as xw
import datetime
today = str(datetime.date.today()).replace('-', '')


url='https://www.twse.com.tw/exchangeReport/MI_INDEX?response=csv&date=' + today + '&type=ALL'
res = requests.get(url)
data = res.text
data.split('\n')

print(data)
type(data.split('\n'))
print(data.split('\n')[-973])

for da in data.split('\n'):
    if len(da.split('","')) == 16 and da.split('","')[0][0] != '=':
        print(da.split('","'))

cleaned_data = []
for da in data.split('\n'):
    if len(da.split('","')) == 16 and da.split('","')[0][0] != '=':
        cleaned_data.append([ele.replace('",\r','').replace('"','') 
                             for ele in da.split('","')])
df = pd.DataFrame(cleaned_data, columns = cleaned_data[0])
df = df.set_index('證券代號')[1:]
xw.view(df)
