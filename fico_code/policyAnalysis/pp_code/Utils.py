import os


def check_folder(path):
    if os.path.isdir(path):
        return True
    else:
        os.makedirs(path)
        return False


def count_characters(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    english_count = sum(1 for char in content if char.isalpha() and char.isascii())
    chinese_count = sum(1 for char in content if '\u4e00' <= char <= '\u9fff')

    English_flag = True
    if english_count > chinese_count:
        print("英文字符数量较多")
    elif english_count < chinese_count:
        print("中文字符数量较多")
        English_flag = False
    else:
        print("英文字符和中文字符数量相同")
    return English_flag


