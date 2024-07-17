import requests
from bs4 import BeautifulSoup
import time

url = "http://www.umeituku.com/bizhitupian/weimeibizhi/"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0'
}

response = requests.get(url, headers=headers)
response.encoding = 'utf-8'
page = BeautifulSoup(response.text, "html.parser")
data = page.find("div", attrs={"class": "TypeList"}).find_all("a")

for hrefs in data:
    href = hrefs.get("href")
    response2 = requests.get(href, headers=headers)
    response2.encoding = "utf-8"
    page2 = BeautifulSoup(response2.text, "html.parser")
    # 检查是否存在<p>标签，然后检查是否存在<a>标签，最后获取<img>标签的src属性
    p = page2.find("p", align="center")
    if p:
        a = p.find("a")
        if a:
            img = a.find("img")
        else:
            img = p.find("img")
        if img:
            src = img.get("src")
            print(src)
            time.sleep(1)
        else:
            continue
    else:
        continue
    response2.close()
response.close()