from bs4 import BeautifulSoup
import re


def html2txt(HTML_path, txt_path):
    # 读取HTML文件
    with open(HTML_path, 'r', encoding='utf-8') as html_file:
        html_content = html_file.read()
    # 创建Beautiful Soup对象并解析HTML
    soup = BeautifulSoup(html_content, 'html.parser')

    # 找到所有的<strong>标签
    strong_tags = soup.find_all('strong')

    # 遍历每个<strong>标签并保留其内容
    for strong_tag in strong_tags:
        strong_tag.insert_before('')  # 在<strong>标签前插入换行符

    # 提取文本内容
    text_content = soup.get_text(strip=False)

    # 拆分文本内容为行，并逐行写入到文本文件
    lines = text_content.splitlines()
    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        for line in lines:
            txt_file.write(line.strip() + '\n')
