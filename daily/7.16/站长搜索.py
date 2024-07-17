import requests
from bs4 import BeautifulSoup
import os

# 用户输入关键词和需要抓取的页数
keyword = input("输入关键词: ")
sum_pages = int(input("输入需要抓取的页数: "))

# 存储所有找到的audio网址
audio_urls = []

# 循环抓取每一页的数据
for page in range(1, sum_pages + 1):
    url = f"https://aspx.sc.chinaz.com/query.aspx?keyword={keyword}&issale=&classID=0&navindex=0&page={page}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0"
    }
    response = requests.get(url, headers=headers)
    response.encoding = 'utf-8'

    # 解析网页内容
    soup = BeautifulSoup(response.text, "html.parser")
    content = soup.find_all("div", class_="new_block")
    # print(content)
    for item in content:
        # 正确提取<a>标签并检查是否为广告图片
        title_link = item.find('a', href=True)  # 寻找具有href属性的<a>标签
        if title_link and '图片' not in title_link.text:
            # 构造完整的音频页面URL
            audio_page_link = title_link['href']
            if not audio_page_link.startswith(('http:', 'https:')):
                audio_page_url = f"https:{audio_page_link}"  # 添加协议头
            else:
                audio_page_url = audio_page_link

            # 请求音频页面
            audio_response = requests.get(audio_page_url, headers=headers)
            audio_soup = BeautifulSoup(audio_response.text, "html.parser")

            # 寻找<audio>标签下的<source>标签，并获取src属性
            audio_tag = audio_soup.find('audio')
            if audio_tag:
                source_tag = audio_tag.find('source')
                if source_tag and 'src' in source_tag.attrs:
                    audio_url = source_tag['src']
                    audio_urls.append(audio_url)
    # print(audio_urls)
    response.close()

# 下载并保存音频文件
for index, audio_url in enumerate(audio_urls, start=1):
    if not audio_url.startswith(('http:', 'https:')):
        audio_url = 'https:' + audio_url  # 为audio_url添加https协议头
    filename = f"{keyword}{index:04d}.mp3"
    directory = os.path.join(os.getcwd(), keyword)
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, filename)

    with requests.get(audio_url, stream=True) as audio_response:
        with open(file_path, 'wb') as audio_file:
            for chunk in audio_response.iter_content(chunk_size=8192):
                audio_file.write(chunk)
    print(f"文件已保存: {file_path}")

print("所有音频文件已下载完毕。")