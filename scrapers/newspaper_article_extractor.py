from newspaper import Article
from newspaper import Config
import pandas as pd
import numpy as np


data = pd.read_csv('ukrinform_urls_and_dates.csv')

user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36 Edg/106.0.1370.34'
config = Config()
config.browser_user_agent = user_agent
#urls=['https://ukranews.com/en/news/886758-russian-troops-tried-to-attack-in-areas-of-13-settlements-in-east-and-south-general-staff']
date_list = []
content_list = []
title_list = []
url_list = []
count=0

for url in data['urls']:
        try:
            article = Article(url, config=config)
            article.download()
            article.parse()
            
            count+=1
            print(count)
            
        except:
            continue
    
        
        content_list.append(article.text)
        title_list.append(article.title)
        url_list.append(article.url)

df = pd.DataFrame()
df['title'] = title_list
df['publish_date'] = data['publish_date']
df['content'] = content_list
df['url'] = url_list
df.to_csv('ukrinform_final.csv')
#print(df['content'])