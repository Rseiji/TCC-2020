# -*- coding: utf-8 -*-
import scrapy
import re
import glob
from scrapy.http import Request
from scrapy import signals, Spider, Item, Field


class ChessSpiderSpider(scrapy.Spider):
    name = 'chess_spider'
    allowed_domains = ['gameknot.com']
    
    saved_pages = glob.glob('/home/*/TCC-2020/saved_files/*')
    start_urls = ['file://' + s for s in saved_pages]

    def parse(self, response):
        
        item = GameItem()

        game_number = re.findall('/saved_files/saved(.*).html', response.url)[0]
        item_comment = [item + '\n' for item in response.css('tr td:nth-child(2)[style*=vertical-align]::text').extract() if item not in '•']        
        item_move = [item for item in response.xpath("//tr/td[1][contains(@style,'vertical-align: top')]/text()").extract() if item not in ('\n', '\n\n', '\xa0', ' ', 'Pages:\xa0')]


        item['game_commentary'] = item_comment
        item['game_movement'] = item_move
        item['game_number'] = game_number

        yield item
        

class GameItem(Item):
    game_commentary = Field()
    game_movement = Field()
    game_number = Field()
