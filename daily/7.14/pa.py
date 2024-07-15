import requests
import re
import csv

# s1，定位到2020必看片
domain = "https://www.dy2018.com"
headers = {"User-Agent":
           "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0"}
resp = requests.get(domain, verify=False, headers=headers)  # 去掉安全验证
resp.encoding = 'utf-8'  # 一般网站会告诉你他编码格式，也可以写gbk,如果没有这一句则可能出现乱码
page = resp.text  # 用page表达一下，方便理解

# 三次正则表达式都放到一起写了，也可以写到后面
obj1 = re.compile(r"2024必看热片.*?<ul>(?P<ul>.*?)<ul>", re.S)
obj2 = re.compile(r"<a href='(?P<href>.*?)'", re.S)
obj3 = re.compile(r'<meta name=keywords content="(?P<moviename>.*?)下载">.*?<td '
                  r'style="WORD-WRAP: break-word" bgcolor="#fdfddf"><a href="(?P<dizhi>.*?)"', re.S)
result1 = obj1.finditer(page)
child_href_list = []  # 准备一个字典存所有子页面
print(page)
for it in result1:  # 其实只有一个地方能匹配，所以用search也行，不必循环
    ul = it.group("ul")  # 缩小区间,就算是定位到必看片系列了
    # print(ul)

    # s2,从2020必看片中提取到子页面地址
    result2 = obj2.finditer(ul)  # 使用缩小后的区间，正则表达式找到子页面链接
    for itt in result2:  # 拼接子页面的url地址，域名+子页面地址，有的需要拼接有的不需要
        child_href = domain + itt.group("href")  # 如果有需要可以用.strip("/")把/去掉
        child_href_list.append(child_href)  # 把子页面链接放到列表里
    # print(child_href_list)#，循环往字典添加子页面网站之后可以打印一下看看网址对不对


# s3,进去子页面，拿到迅雷下载链接
f = open("download.csv", mode="w")
csvwritter = csv.writer(f)
for href in child_href_list:
    child_resp = requests.get(href, verify=False)  # 循环进入子页面并去掉安全验证
    child_resp.encoding = 'gbk'  # 修改编码格式
    page2 = child_resp.text  # 用page2表示
    result3 = obj3.search(page2)  # 也是只有一处能匹配，所以search
    print(result3.group("moviename"))
    print(result3.group("dizhi"))
    dic = result3.groupdict()
    dic['moviename'] = dic['moviename'].strip()
    dic['dizhi'] = dic['dizhi'].strip()
    csvwritter.writerow(dic.values())
f.close()