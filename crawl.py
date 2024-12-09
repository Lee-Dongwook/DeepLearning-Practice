import requests
from bs4 import BeautifulSoup

response = requests.get('https://news.naver.com/')

soup = BeautifulSoup(response.text, 'html.parser')

headlines = soup.select('div.cjs_channel_card div.cjs_journal_wrap._item_contents div.cjs_news_tw div.cjs_t')

for headline in headlines:
    print(headline.text.strip())