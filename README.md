# Ukraine-Conflict-Analysis

![UkraineAnalysisNewspapersandMap](https://user-images.githubusercontent.com/117476344/200758790-9dec9481-ffc5-4a56-9a80-d69cbfbae4ff.png)

## Introduction ##

In this project the coverage of the Ukraine conflict by CNN, Fox News, Reuters, Ukrinform (Ukranian news agency), and TASS (Russian news agency) is examined and compared.

The following analysis was performed:

1. Basic exploration of text length/article distribution
2. Sentiment analysis
3. Identification of keywords using [c-TF-IDF](https://github.com/MaartenGr/cTFIDF)

Following this, both a gradient boosted library XGBoost and a dense neural network were trained to classify an article based on which newspaper it belonged to with over **95 %** accuracy.

## Data used in this study ##

Articles from the five newspapers were scraped from the web using a combination of scrapy, selenium, and [pygooglenews](https://github.com/kotartemiy/pygooglenews).  

In this study we focus on articles spanning from the beginning of the conflict (Feburary 24, 2022) to September 30, 2022.

## Key insights ##

**Sentiment analysis**

Sentiment over time is heavily varied for CNN, Fox News, and Reuters but the sentiment from TASS and Ukrinform accurately reflect the state of the war.

![Sentiment_score_over_time](https://user-images.githubusercontent.com/117476344/200761361-996f92bd-873f-431c-965d-74c5915e761e.png)

**Key words**

Key words from each newspaper are distinguished from each other and give good insight in how each newspaper covers the conflict.

* **Reuters**: Focused on economic impact.
* **CNN** Considers the impact on a global scale.
* **Fox News** Writes mainly on how the war affects the USA and critical of the Biden administration's response
* **TASS** Biased towards Russia by referencing areas of Ukraine with pro-Russian separitists, etc.
* **Ukrinform** Biased towards Ukraine seeing itself as a victim with words like "invasion" and "enemy".

For more analysis, see this [notebook](Analysis.ipynb)

![Key_unigrams_bigrams](https://user-images.githubusercontent.com/117476344/200761916-5098c357-ec91-4914-a5a8-2bcc583c0431.png)



