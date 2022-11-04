from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from shutil import which
import time

url_list = []
page_url = "https://tass.com/military-operation-in-ukraine"
# set up the driver
chrome_options = Options()
# chrome_options.add_argument("--headless") # uncomment if don't want to appreciate the sight of a possessed browser
driver = webdriver.Chrome(executable_path='./chromedriver.exe', options=chrome_options)
    
driver.get(page_url)

last_height = driver.execute_script("return document.body.scrollHeight")
while True:
    # Scroll down to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # Wait to load page
    time.sleep(5)

    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

urls = driver.find_elements(By.XPATH, "//a[@class='news-content__container']")
for url in urls:
    url_list.append(url.get_attribute("href"))
with open('tass_urls.csv', 'a') as f:
    for i in range(len(url_list)):
        f.write(url_list[i] + "\n")
print(url_list)  