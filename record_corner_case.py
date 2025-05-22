import json
import os
import spacy
import re


def get_all_previous_keywords():
    with open('corner_case_list/all_previous_keywords.json', "r") as file:
        privacy_keywords = json.load(file)['data']

    # get all privacy keyword: 1970
    print('get all privacy keyword:', len(privacy_keywords))

    return privacy_keywords


def get_all_complaince_data():
    data_path_dir = r'C:\Master\PrivacyLegal\code\FiCo\evaluation_record\complaince_record\complaince_result_en'
    data_set = set()
    for file_name in os.listdir(data_path_dir):
        file_path = os.path.join(data_path_dir, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            for word in content.split(','):
                word = word.strip()
                if word.strip() == '':
                    continue
                data_set.add(word)
    print('all complaince data:', len(data_set))

    return data_set


def calculate_phrase_similarity(phrase1, phrase2):
    nlp = spacy.load("en_core_web_md")
    doc1 = nlp(phrase1)
    doc2 = nlp(phrase2)
    similarity = doc1.similarity(doc2)
    return similarity


def measure_un_corner_case_number():
    data_count = 0
    similar_count = 0

    data_set = get_all_complaince_data()
    privacy_keywords = get_all_previous_keywords()
    for data in data_set:
        data_count += 1
        for keyword in privacy_keywords:
            if calculate_phrase_similarity(data, keyword) > 0.8:
                similar_count += 1
                print(data, keyword)

    print('Data count: ', data_count)
    print('Similar count: ', similar_count)


def get_all_ture_positive_for_each_app(evaluation_dir):
    tp_dict = {}
    fp_dict = {}
    fn_dict = {}

    all_tp_count = 0
    all_fp_count = 0
    all_fn_count = 0
    for file in os.listdir(evaluation_dir):
        file_path = os.path.join(evaluation_dir, file)
        all_p_items = set()
        fp_items = set()
        fn_items = set()
        with open(file_path, 'r', encoding='utf-8') as f:
            tp_count = 0
            fp_count = 0
            fn_count = 0
            line = f.readline()
            while line:
                cur_line = line.strip()
                line = f.readline()
                if cur_line.startswith('All positive: '):
                    line_pat_app_p = r'All positive: {(.*)}'
                    mat_res = re.match(line_pat_app_p, cur_line)
                    if mat_res:
                        all_p_items = set(mat_res.group(1).replace('\'', '').replace(' ', '').split(','))
                elif cur_line.startswith('All false positive:'):
                    line_pat_fp = r'All false positive: {(.*)}'
                    mat_res = re.match(line_pat_fp, cur_line)
                    if mat_res == '':
                        fp_items = set()
                    elif mat_res:
                        fp_items = set(mat_res.group(1).replace('\'', '').replace(' ', '').split(','))
                        if len(fp_items) == 1 and '' in fp_items:
                            fp_items = set()
                elif cur_line.startswith('All false negative: '):
                    line_pat_fn = r'All false negative: {(.*)}'
                    mat_res = re.match(line_pat_fn, cur_line)
                    if mat_res == '':
                        fn_items = set()
                    elif mat_res:
                        fn_items = set(mat_res.group(1).replace('\'', '').replace(' ', '').split(','))
                        if len(fn_items) == 1 and '' in fn_items:
                            fn_items = set()

                elif cur_line.startswith('Length: '):
                    line_pat_fn = r'Length: all positive: (.*), false positive: (.*), false negative: (.*)'
                    mat_res = re.match(line_pat_fn, cur_line)
                    if mat_res:
                        all_p = int(mat_res.group(1))
                        if all_p == 0:
                            print(file)
                            print('p is 0')
                            continue
                        fp = int(mat_res.group(2))
                        fn = int(mat_res.group(3))
                        tp_count += (all_p - fp)
                        fp_count += fp
                        fn_count += fn

            all_tp_count += tp_count
            all_fp_count += fp_count
            all_fn_count += fn_count

            if len(all_p_items) == 0:
                print(file_path)
                continue
            tp_items = all_p_items - fp_items
            tp_dict[file] = tp_items
            fp_dict[file] = fp_items
            fn_dict[file] = fn_items

            if tp_count != len(tp_items):
                print(file, 'tp error')
                print(len(tp_items), all_p_items, fn_items, fp_items)
            if fp_count != len(fp_items):
                print(file, 'fp error')
                print(fp_items, fp_count)
            if fn_count != len(fn_items):
                print(file, 'fn error.')

    return tp_dict, fp_dict, fn_dict


def find_all_corner_case():
    privacy_keywords = get_all_previous_keywords()

    # en
    all_personal_data_for_each_app, _, _ = get_all_ture_positive_for_each_app(r'C:\Master\PrivacyLegal\code\FiCo\evaluation_record\performance_record\evaluation_record_en')
    print(all_personal_data_for_each_app)
    all_personal_data = list()
    for app in all_personal_data_for_each_app:
        all_personal_data.extend(all_personal_data_for_each_app[app])
    print(len(all_personal_data))

    corner_data_en = set(all_personal_data) - set(privacy_keywords)
    print(len(corner_data_en))

    for app in all_personal_data_for_each_app:
        all_personal_data_for_each_app[app] = list(set(all_personal_data_for_each_app[app]) - set(privacy_keywords))

    with open('./corner_data_en_tmp.json', 'w') as f:
        json.dump(all_personal_data_for_each_app, f, indent=4)

    # zh
    with open(r'C:\Master\PrivacyLegal\code\FiCo\evaluation_record\performance_record\translated.json', 'r') as f:
        all_personal_data_for_each_app = json.load(f)
    all_personal_data = list()
    for app in all_personal_data_for_each_app:
        all_personal_data.extend(all_personal_data_for_each_app[app])
    print(len(all_personal_data))

    corner_data_zh = set(all_personal_data) - set(privacy_keywords)
    print(len(corner_data_zh))

    for app in all_personal_data_for_each_app:
        all_personal_data_for_each_app[app] = list(set(all_personal_data_for_each_app[app]) - set(privacy_keywords))

    with open('./corner_data_zh_tmp.json', 'w') as f:
        json.dump(all_personal_data_for_each_app, f, indent=4)


def get_all_corner_case_per_app():
    with open('C:\Master\PrivacyLegal\code\FiCo\corner_data.json', 'r') as f:
        all_personal_data_for_each_app = json.load(f)

    # 1. how many app suffer from corner case?
    app_count, zh_app_count, en_app_count = 0, 0, 0
    for lang in all_personal_data_for_each_app:
        for app in all_personal_data_for_each_app[lang]:
            app_count += 1
            if lang == 'zh':
                zh_app_count += 1
            else:
                en_app_count += 1
    print('how many app suffer from corner case? ', app_count, 'the proportion is ', str(app_count / 71), ' zh count: ', zh_app_count, 'en count: ', en_app_count)

    # 2. what's the proportion of corner case in personal data?
    corner_case_count, corner_case_count_zh, corner_case_count_en = 0, 0, 0
    for lang in all_personal_data_for_each_app:
        for app in all_personal_data_for_each_app[lang]:
            data_count = len(all_personal_data_for_each_app[lang][app])
            corner_case_count += data_count
            if lang == 'zh':
                corner_case_count_zh += data_count
            else:
                corner_case_count_en += data_count
    print('what\'s the proportion of corner case in personal data?', corner_case_count, 'the proportion is ', str(corner_case_count/1258),
          ' zh count: ', corner_case_count_zh, ' zh proportion: ', str(corner_case_count_zh/632), 'en count: ', str(corner_case_count_en), 'en proportion: ', str(corner_case_count_en/626))

    # 3. what is the proportion of corner case data in non complaint data?
    # with open(r'C:\Master\PrivacyLegal\code\FiCo\non_complaint_data.json') as f:
    #     all_non_complaint_data_for_each_app = json.load(f)
    with open(r'C:\Master\PrivacyLegal\code\FiCo\non_complaint_data_corner_case.json') as f:
        all_non_complaint_data_corner_case_for_each_app = json.load(f)

    proportion_corner_complaint_zh = []
    proportion_corner_complaint_en = []
    proportion_corner_complaint_total = []
    for lang in all_personal_data_for_each_app:
        for app in all_personal_data_for_each_app[lang]:
            if app in all_non_complaint_data_corner_case_for_each_app[lang]:
                if len(all_personal_data_for_each_app[lang][app]) == 0:
                    continue
                proportion_corner = len(all_non_complaint_data_corner_case_for_each_app[lang][app]) / len(all_personal_data_for_each_app[lang][app])
            else:
                proportion_corner = 0
            proportion_corner_complaint_total.append(proportion_corner)
            if lang == 'zh':
                proportion_corner_complaint_zh.append(proportion_corner)
            else:
                proportion_corner_complaint_en.append(proportion_corner)
    print('what is the proportion of corner case data in non complaint data?', 'total: ', sum(proportion_corner_complaint_total) / len(proportion_corner_complaint_total),
          'zh proportion: ', sum(proportion_corner_complaint_zh) / len(proportion_corner_complaint_zh),
          'en proportion: ', sum(proportion_corner_complaint_en) / len(proportion_corner_complaint_en))

    # 4. generate the corner_case_list
    corner_case_set = set()
    for lang in all_personal_data_for_each_app:
        for app in all_personal_data_for_each_app[lang]:
            for word in all_personal_data_for_each_app[lang][app]:
                corner_case_set.add(word)

    print(len(corner_case_set), corner_case_set)
    # with open(r'C:\Master\PrivacyLegal\code\FiCo\released_corner_case.json', 'w', encoding='utf-8') as f:
    #     release_corner_case = dict()
    #     release_corner_case['corner_case'] = list(corner_case_set)
    #     json.dump(release_corner_case, f, indent=4)



def get_non_complaint_data_corner_case():
    dir = r'C:\Master\PrivacyLegal\code\FiCo\temp\uncorner_data_list_to_release'
    pat = 'data: (.*); similar: (.*)'

    non_corner_case_set = set()
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        with open(file_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            mat = re.match(pat, line)
            if mat is not None:
                non_corner_case_set.add(mat.group(1).strip().lower())

    with open(r'/evaluation_record/complaince_record/non_complaint_data.json', 'r') as f:
        all_non_complaint_data_for_each_app = json.load(f)

    all_non_complaint_data_corner_case_for_each_app = dict()
    for lang in all_non_complaint_data_for_each_app:
        if lang not in all_non_complaint_data_corner_case_for_each_app:
            all_non_complaint_data_corner_case_for_each_app[lang] = dict()
        for app in all_non_complaint_data_for_each_app[lang]:
            if app not in all_non_complaint_data_corner_case_for_each_app[lang]:
                all_non_complaint_data_corner_case_for_each_app[lang][app] = list()
            for word in all_non_complaint_data_for_each_app[lang][app]:
                if word not in non_corner_case_set:
                    all_non_complaint_data_corner_case_for_each_app[lang][app].append(word)

    with open(r'C:\Master\PrivacyLegal\code\FiCo\non_complaint_data_corner_case.json', 'w', encoding='utf-8') as outfile:
        json.dump(all_non_complaint_data_corner_case_for_each_app, outfile, indent=4)




if __name__ == '__main__':
    # get_all_previous_keywords()
    # translate_chinese()
    # get_all_complaince_data()
    # get_non_complaint_data_corner_case()
    # find_all_corner_case()
    get_all_corner_case_per_app()

