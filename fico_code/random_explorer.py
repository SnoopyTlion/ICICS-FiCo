# use monkey
import os
import multiprocessing
import threading
import time
import os.path

import unicodedata
import xml.etree.ElementTree as ET
import uiautomator2 as u2
import time
from utils import *


def run_monkey(pkg_name, remaining_time):

    monkey_1 = 'adb shell dumpsys window | findstr mCurrentFocus'
    os.system(monkey_1)
    monkey_2 = f'adb shell monkey -p {pkg_name} --pct-syskeys 0 --throttle 1000 -v {int(remaining_time)*4}'
    os.system(monkey_2)


def get_xml_from_view_tree(device):
    """
    将uiautomator dump的结果转化成xml结构，并且进行一系列的预处理。
    :param device:
    :param convertor:
    :param use_img_convertor:
    :param view_tree:
    :return:
    """
    view_tree = device.dump_hierarchy()
    root_element = ET.fromstring(view_tree)
    nodes_to_delete = root_element.findall(".//node[@package='" + "com.android.systemui" + "']")
    # Remove the found nodes from their parent
    for node in nodes_to_delete:
        if node in root_element:
            root_element.remove(node)
    nodes_to_delete = root_element.findall(".//node[@package='" + "com.baidu.input_mi" + "']")
    # Remove the found nodes from their parent
    for node in nodes_to_delete:
        if node in root_element:
            root_element.remove(node)

    for node in root_element.iter():
        if 'text' in node.attrib:
            node.attrib['text'] = get_normal_chinese(node.attrib['text'])
            if node.attrib['text'] == '':
                if node.attrib['content-desc'] != '' and node.attrib['content-desc'] is not None:
                    node.attrib['text'] = node.attrib['content-desc']

    return root_element

def get_normal_chinese(text: str):
    # 去掉所有特殊字符，空格、换行符、16进制字符。。。
    special_chars = []
    code_names = ['CJK', 'SPACE', 'LATIN', 'DIGIT']
    for char in text:
        # 是英文或者空格，正常字符
        if char.isalpha() or char == ' ':
            continue
        char_name = unicodedata.name(char, "")
        normal = False
        # 把不是中文的字符都算作不正常字符
        for code_name in code_names:
            if code_name in char_name:
                normal = True
            if not normal:
                special_chars.append(char)
    for char in special_chars:
        text = text.replace(char, '')
    return text


def record_log_and_coarse_analysis(device, remaining_time, app_explorer, logger):
    start_time = time.time()

    try:
        time_now = time.time()

        while int(time_now-start_time) <= remaining_time:
            # log
            root = get_xml_from_view_tree(device)
            text_set = set()
            for node in root.iter():
                if 'text' in node.attrib:
                    if node.attrib['text'].strip() == '':
                        continue
                    text_set.add(node.attrib['text'])

            logger.log_cur_page('RANDOM', text_set)

            coarse_analysis(app_explorer, root)

            time.sleep(0.5)
            time_now = time.time()
    except Exception as e:
        print(e)


def coarse_analysis(app_explorer, root_element):
    cur_node = app_explorer.generate_new_node_from_view_tree_xml(root_element)

    for view in cur_node.view_type_dict:
        if cur_node.view_type_dict[view] != ui_type_TEXT:
            cur_node.view_type_dict[view] = ui_type_INDIRECT_INPUT
    app_explorer.try_analysis_privacy(cur_node, mode='force')


def random_run(pkg_name, device, remaining_time, app_explorer, logger):
    print('start random', remaining_time)

    if remaining_time <= 0:
        return

    p1 = threading.Thread(target=run_monkey, args=(pkg_name, remaining_time))
    p1.start()

    record_log_and_coarse_analysis(device, remaining_time, app_explorer, logger)

    p1.join()

    print('end random')


def test():
    while True:
        print('ppp')

def process_test():
    p1 = threading.Thread(target=test)

    p1.start()

    while True:
        print('a')
    p1.join()


if __name__ == '__main__':
    process_test()

