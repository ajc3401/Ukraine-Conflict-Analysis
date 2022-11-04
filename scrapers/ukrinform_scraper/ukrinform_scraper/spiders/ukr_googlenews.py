from pygooglenews import GoogleNews
import datetime
from dateparser import parse as parse_date
import pandas as pd

months = [x for x in range(2, 3, 1)]
days = [x for x in range(20, 22, 1)]
gn = GoogleNews(lang = 'en', country = 'US')
url_list = []
for month in months:
    for day in days:
        
#print(parse_date(start_date_formatted).strftime('%Y-%m-%d'))
        try:
            start_date = datetime.date(2022, month, day)
            end_date = datetime.date(2022, month, day+1)

            start_date_formatted = start_date.strftime('%Y-%m-%d')
            s = gn.search('cnn ukraine war', from_ = start_date.strftime('%Y-%m-%d'), to_= end_date.strftime('%Y-%m-%d'))
        
        except:
            continue
        for entry in s['entries']:
            print(entry.published)
            if(entry.link.startswith('https://www.cnn.com')):
                if (entry.link.startswith('https://www.cnn.com/videos')) or (entry.link.startswith('https://www.cnn.com/audio')):
                    continue
                else:
                    url_list.append(entry.link)
print(url_list)
print(len(url_list))
data = pd.DataFrame()
data['urls'] = url_list
#data.to_csv('cnn_urls.csv')