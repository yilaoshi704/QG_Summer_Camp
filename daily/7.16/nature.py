from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import time
import os
import requests
from bs4 import BeautifulSoup

# 用户输入关键词和需要抓取的页数
keyword = input("输入关键词: ")
sum_pages = int(input("输入需要抓取的页数: "))

# 存储所有找到的audio网址
audio_urls = []
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0'
}
directory = keyword
if not os.path.exists(directory):
    os.makedirs(directory)  # 如果文件夹不存在，则创建文件夹
# 设置Chrome选项
options = webdriver.ChromeOptions()
# options.add_argument('--headless')  # 注释掉无头模式
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')

# 循环抓取每一页的数据
for a in range(1, sum_pages + 1):
    url = f"https://stock.zhipianbang.com/sound/list-63.html?keyword={keyword}&page={a}"
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.text, "html.parser")
    content = soup.find_all("div", class_="sound_list_item type-name-sound")
    # 遍历找到的所有符合条件的div元素
    for item in content:
        # 提取audio的网址
        audio_url = item.find("div", class_="wavesurfe_content")["data-src"]
        audio_urls.append(audio_url)

download_count = 1  # 初始化下载计数器
for audio_url in audio_urls:
    # 启动Chrome浏览器
    driver = webdriver.Chrome(options=options)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument",
                           {"source": """Object.defineProperty(navigator, 'webdriver',{get: () => undefined})"""})
    driver.get(audio_url)
    time.sleep(5)  # 等待页面加载

    # 处理滑动验证码
    try:
        slider = driver.find_element(By.CLASS_NAME, 'nc_iconfont.btn_slide')
        action = ActionChains(driver)
        action.click_and_hold(slider).perform()
        action.move_by_offset(260, 0).perform()
        action.release().perform()
        time.sleep(1)  # 等待验证完成
        input()

    except Exception as e:
        print(f"滑动验证码处理失败: {e}")
        input("请手动完成滑动验证码，然后按回车继续...")
        # 下载文件
        filename = f"{download_count}.mp3"
        file_path = os.path.join(directory, filename)
        print(f"正在下载到 {file_path}: ")

        try:
            with requests.get(audio_url, stream=True) as audio_response:
                if audio_response.status_code == 200:
                    with open(file_path, 'wb') as audio_file:
                        for chunk in audio_response.iter_content(chunk_size=8192):
                            audio_file.write(chunk)
                    print(f"文件已保存: {file_path}")
                    # 检查文件大小
                    file_size = os.path.getsize(file_path)
                    print(f"文件大小: {file_size} 字节")
                else:
                    print(f"下载失败，状态码：{audio_response.status_code}")
        except Exception as e:
            print(f"下载过程中发生错误: {e}")

        download_count += 1  # 增加下载计数器

    # 关闭浏览器
    driver.quit()

print("所有音频文件已下载完毕。")