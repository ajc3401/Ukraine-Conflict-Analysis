from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from shutil import which
import time

url_list = []  
for page in range(0, 8000, 20):
    
    page_url = "https://www.reuters.com/site-search/?query=ukraine+war&sort=relevance&offset=" + str(page) +"&date=past_year"
    chrome_path = which("chromedriver")
    # set up the driver
    chrome_options = Options()
    # chrome_options.add_argument("--headless") # uncomment if don't want to appreciate the sight of a possessed browser
    driver = webdriver.Chrome(executable_path='./chromedriver.exe', options=chrome_options)
    driver.get(page_url)
    urls = driver.find_elements(By.XPATH, "//a[@data-testid='Heading']")
    for url in urls:
        url_list.append(url.get_attribute("href"))
with open('reuters_urls.csv', 'a') as f:
    for i in range(len(url_list)):
        f.write(url_list[i] + "\n")
#np.savetxt('kyiv_post_urls.dat', url_list)
#closing the driver
driver.close()