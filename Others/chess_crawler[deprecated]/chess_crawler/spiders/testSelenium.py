from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select



chrome_options = Options()
#chrome_options.add_argument("--headless")
driver = webdriver.Chrome(options=chrome_options)


'''

#Dentro de uma p


driver.get('https://gameknot.com/annotate.pl?id=66392')



save_export = driver.find_element_by_xpath('div[@id="anno-links"]/a[3]')


save_export.click()



'''
'''
driver.get("https://gameknot.com/annotation.pl/ajedrez?gm=66392")


x = driver.find_element_by_xpath("//tr/td/a[1]")


x.click()

y = driver.find_element_by_xpath("//div[@id='anno-links']/a[3]")

y.click()
'''
driver.get("https://gameknot.com/annotate.pl?id=66392")


x = driver.find_element_by_xpath("//div[@id='anno-footer']/div[@id='anno-links']/a[3]")

x.click()


y = driver.find_element_by_xpath("//div[@class='popmenu']/a[2]")

y.click()

z = driver.find_element_by_xpath("//tr/td/textarea[@id='pgn_code']")

lala = z.get_attribute('innerHTML')



#print(type(lala))

#Varrer annotated games e conseguir página de todos os jogos
'''
#driver.get('https://gameknot.com/list_annotated.pl?u=all')
driver.get('https://gameknot.com/list_annotated.pl?u=all&c=0&sb=0&rm=0&rn=0&rx=9999&sr=0&p=4')

#x = driver.find_element_by_xpath('//tr[@class="evn_list"]/td[2]/a[2]/@href')

#for X in x:
    #print(X.get_attribute('href'))


list_games = []

x = driver.find_elements_by_xpath('//tr[@class="evn_list"]/td[2]/a[2]')

for y in x:
    list_games.append(y.get_attribute('href'))


z = driver.find_elements_by_xpath('//tr[@class="odd_list"]/td[2]/a[2]')

for y in z:
    list_games.append(y.get_attribute('href'))


print(list_games)


print(len(list_games))
'''