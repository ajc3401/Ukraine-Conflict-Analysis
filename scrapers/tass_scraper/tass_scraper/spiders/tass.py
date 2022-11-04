import scrapy
from shutil import which 
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from scrapy.selector import Selector
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

class TassSpider(scrapy.Spider):
    name = 'tass'
    allowed_domains = ['tass.com']
    start_urls = ['http://tass.com/']
    def __init__(self):
        chrome_options = Options()
        chrome_path = which("chromedriver.exe")
        # chrome_options.add_argument("--headless") # uncomment if don't want to appreciate the sight of a possessed browser
        driver = webdriver.Chrome(executable_path="../../../chromedriver.exe", options=chrome_options)
        driver.maximize_window()
        driver.get("https://tass.com/military-operation-in-ukraine")

        last_height = driver.execute_script("return document.body.scrollHeight")
        i=0
        while True:
            # Scroll down to bottom
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            # Wait to load page
            time.sleep(7)
            i+=1
    # Calculate new scroll height and compare with last scroll height
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        self.html = driver.page_source
    def parse(self, response):
        resp = Selector(text=self.html)
        results = resp.xpath("//h2/a")
        for result in results:
            link = result.xpath(".//@href").get()
            title = result.xpath(".//text()").get()
            yield response.follow(url=link, callback=self.parse_article, meta={"title": title, "url": link})
    def parse_article(self, response):
        title = response.request.meta['title']
        link = response.request.meta['url']
        #results = resp.xpath(By.XPATH, "(//article[@class='ssrcss-pv1rh6-ArticleWrapper+e1nh2i2l6'])")
        #content = ' '.join(response.xpath('normalize-space(//div[@class="article-body-v2__content__27d7d paywall-article"]/p[starts-with(@data-testid, "paragraph")]/text())').getall())
        #content = ' '.join(response.xpath("normalize-space(//div[@class='article-body-v2__content__27d7d paywall-article']//text())").getall())
        content = ' '.join(response.xpath("//article/div[@class='newsText']//p/text()").getall())
                                
        yield {
            "title": title,
            "url": link,
         #   "byline": ' '.join(authors), # could be multiple authors
            "time": response.xpath("//dateformat/@time").get(),
            "content": content
            } 