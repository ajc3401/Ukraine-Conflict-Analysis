from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from shutil import which
import time

url_list = []  
for page in range(1, 3, 20):
    
    page_url = "https://www.cnn.com/search?q=ukraine+war&from=0&size=10&page=" + str(page) + "&sort=newest&types=article&section="
    chrome_path = which("chromedriver")
    # set up the driver
    chrome_options = Options()
    # chrome_options.add_argument("--headless") # uncomment if don't want to appreciate the sight of a possessed browser
    driver = webdriver.Chrome(executable_path='./chromedriver.exe', options=chrome_options)
    driver.get(page_url)
    urls = driver.find_elements(By.XPATH, "//div[@class='search__results-list']//a")
    for url in urls:
        url_list.append(url.get_attribute("href"))
print(url_list)
#with open('cnn_urls_2.csv', 'a') as f:
 #   for i in range(len(url_list)):
  #      f.write(url_list[i] + "\n")
#np.savetxt('kyiv_post_urls.dat', url_list)
#closing the driver
driver.close()