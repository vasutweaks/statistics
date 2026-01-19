# https://github.com/obedjunias/Cricket-Statistics/blob/master/Batting-Statistics/Notebooks/Test_BStats.ipynb
from urllib import request
from bs4 import BeautifulSoup
import pandas as pd
import re

def getTestdata(url):
    link = url
    html_page = request.urlopen(link);
    soup = BeautifulSoup(html_page,'lxml')
    table = soup.find_all('table',class_="engineTable")
    data = table[2].get_text()
    #Rough cleaning to store as an excel file
    clean_data = re.sub('(\\n)',',',data)
    clean_data = re.sub('(,,,,|,,,|,,)',' \n ',clean_data)
    clean_data = re.sub(',',' \t ',clean_data)
    with open('TestStats.xslx','a') as f:
        f.write(clean_data)
    print("Successfully got the data.")

#Url
url = "http://stats.espncricinfo.com/ci/engine/stats/index.html?class=1;page={};template=results;type=batting"

#The above url contains 61 pages of data.
# Using only 2 pages for testing 
# for i in range(1,3):
    # getTestdata(url.format(i))

df = pd.read_csv('TestStats.xslx',sep='\t')
print(df.info())
