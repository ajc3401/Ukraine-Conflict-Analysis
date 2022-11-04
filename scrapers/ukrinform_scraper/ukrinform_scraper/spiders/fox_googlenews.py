from pygooglenews import GoogleNews
import datetime
from dateparser import parse as parse_date
import pandas as pd

months_1 = [x for x in range(2, 3, 1)]
days_1 = [x for x in range(20, 31, 1)]

months_2 = [x for x in range(3, 11, 1)]
days_2 = [x for x in range(1, 31, 1)]
gn = GoogleNews(lang = 'en', country = 'US')
url_list = []
date_list = []
for month in months_1:
    for day in days_1:
        
#print(parse_date(start_date_formatted).strftime('%Y-%m-%d'))
        try:
            start_date = datetime.date(2022, month, day)
            end_date = datetime.date(2022, month, day+1)

            start_date_formatted = start_date.strftime('%Y-%m-%d')
            s = gn.search('ukrinform ukraine war', from_ = start_date.strftime('%Y-%m-%d'), to_= end_date.strftime('%Y-%m-%d'))
        
        except:
            continue
        for entry in s['entries']:
            print(entry.link)
            if(entry.link.startswith('https://www.ukrinform.net')):
                #if (entry.link.startswith('https://www.cnn.com/videos')) or (entry.link.startswith('https://www.cnn.com/audio')):
                 #   continue
                #else:
                url_list.append(entry.link)
                date_list.append(entry.published)
for month in months_2:
    for day in days_2:
        
#print(parse_date(start_date_formatted).strftime('%Y-%m-%d'))
        try:
            start_date = datetime.date(2022, month, day)
            end_date = datetime.date(2022, month, day+1)

            start_date_formatted = start_date.strftime('%Y-%m-%d')
            s = gn.search('ukrinform ukraine war', from_ = start_date.strftime('%Y-%m-%d'), to_= end_date.strftime('%Y-%m-%d'))
        
        except:
            continue
        for entry in s['entries']:
            if(entry.link.startswith('https://www.ukrinform.net')):
                #if (entry.link.startswith('https://www.cnn.com/videos')) or (entry.link.startswith('https://www.cnn.com/audio')):
                 #   continue
                #else:
                url_list.append(entry.link)
                date_list.append(entry.published)                    
#print(url_list)
print(len(url_list))
data = pd.DataFrame()
data['urls'] = url_list
data['publish_date'] = date_list
data.to_csv('ukrinform_urls_and_dates.csv')