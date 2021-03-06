import sys
import re

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager





#Faz download do jogo em .pgn
#input = id da página, que consta na url
def get_pgn(page_id):
    chrome_options = Options()
    chrome_options.add_argument("--headless")

    #Definindo a conexão
    #driver = webdriver.Chrome(options=chrome_options)
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)
    #url
    base_url = 'https://gameknot.com/annotate.pl?id='
    pgn_url = base_url + page_id
    
    #carregando a página
    driver.get(pgn_url)
    
    #Clicando no botão "save/export"
    save_export_button = driver.find_element_by_xpath("//div[@id='anno-footer']/div[@id='anno-links']/a[3]")
    driver.implicitly_wait(10)
    ActionChains(driver).move_to_element(save_export_button).click(save_export_button).perform()

    #Clicando no botão "get pgn"
    get_pgn_button = driver.find_element_by_xpath("//div[@class='popmenu']/a[2]")
    driver.implicitly_wait(10)
    ActionChains(driver).move_to_element(get_pgn_button).click(get_pgn_button).perform()
    
    #Crawleando o .pgn
    pgn_text_DOM = driver.find_element_by_xpath("//tr/td/textarea[@id='pgn_code']")
    pgn_text = pgn_text_DOM.get_attribute('innerHTML')

    driver.quit()
    return pgn_text




#urls dos jogos
game_links = sys.argv[1]
pgn_path = sys.argv[2]


with open(game_links,'r') as f:
    game_urls = f.readlines()

f.close()

game_ids = [re.findall(r'gm=(\d+)$', i)[0] for i in game_urls if re.search(r'gm=(\d+)$', i)]

#game_id = '59540'
for game_id in game_ids:

    pgn_text = get_pgn(game_id)

    pgn_filename = pgn_path + 'pgn_' + game_id + '.txt'
    print('saved' + pgn_filename)
    with open(pgn_filename, 'w') as f:
        f.write("%s" % pgn_text)

    f.close()


