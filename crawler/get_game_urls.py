import sys

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select


initial_page = int(sys.argv[1])
final_page = int(sys.argv[2])
txt_filename = sys.argv[3]

chrome_options = Options()
chrome_options.add_argument("--headless")
driver = webdriver.Chrome(options=chrome_options)


base_url = 'https://gameknot.com/list_annotated.pl?u=all&c=0&sb=0&rm=0&rn=0&rx=9999&sr=0&p='

list_games = []

for i in range(initial_page, final_page):
    print("getting links of page %d ..." % i)
    url = base_url + str(i)
    driver.get(url)

    #Links nas posições pares da tabela da página
    even_links = driver.find_elements_by_xpath('//tr[@class="evn_list"]/td[2]/a[2]')

    for even_link in even_links:
        list_games.append(even_link.get_attribute('href'))


    #Links nas posições ímpares da tabela da página
    odd_links = driver.find_elements_by_xpath('//tr[@class="odd_list"]/td[2]/a[2]')
    
    for odd_link in odd_links:
        list_games.append(odd_link.get_attribute('href'))


with open(txt_filename, 'w') as f:
    for item in list_games:
        f.write("%s\n" % item)

f.close()
