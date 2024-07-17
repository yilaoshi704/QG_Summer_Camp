import os
from bs4 import BeautifulSoup
import requests

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36',
}

# 替换对应链接
response = requests.get("https://stock.zhipianbang.com/sound/search.html?keyword=%E4%BA%A4%E8%B0%88", headers=headers)

content = response.text
soup = BeautifulSoup(content, 'html.parser')
all_h2 = soup.findAll('h2', class_='over_text1 font16')
all_links = soup.findAll('a')
link_box = []
for h2 in all_h2:
    a_tag = h2.find('a')
    if a_tag:
        href = a_tag.get('href')
        link_f = 'https://stock.zhipianbang.com' + href
        link_box.append(link_f)

print("Collected links:", link_box)

i = 1
for link in link_box:
    response_in = requests.get(link)
    content_in = response_in.text
    soup_in = BeautifulSoup(content_in, 'html.parser')
    music_head = soup_in.find(class_='music_head')
    if music_head:
        data_url = music_head.get('data-url')
        print("Downloading:", data_url)
        response_f = requests.get(data_url, headers=headers)
        if response_f.status_code == 200:
            # 指定保存路径
            save_dir = '../test'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            file_path = f'{save_dir}/talk_sound{i}.mp3'
            with open(file_path, 'wb') as file:
                file.write(response_f.content)
            print(f'MP3 file downloaded: {file_path}')
        else:
            print(f"Failed to download file {i}. Status code: {response_f.status_code}")
    i += 1

print('Finished downloading all files.')