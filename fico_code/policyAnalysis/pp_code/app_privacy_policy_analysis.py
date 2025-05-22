import os

from privacypolicy_analysis_util import analyze_host_app_pp
from privacypolicy_analysis_util_zn import analyze_host_app_pp_zn
from Utils import check_folder
from Utils import count_characters
from html2txt_util import html2txt
import hanlp


def txt_result(txt_path, result_path, data_path):
    hanlp_mtl = hanlp.load(hanlp.pretrained.mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE)
    print(f'Save result in {result_path}')
    English_f = count_characters(txt_path)
    if English_f:
        print("English pp")
        analyze_host_app_pp(txt_path, result_path, data_path, hanlp_mtl, print_flag=True)
    else:
        print("处理中文")
        analyze_host_app_pp_zn(txt_path, result_path, data_path, hanlp_mtl, print_flag=True)


if __name__ == '__main__':
    # input: html output: txt
    html_privacy_dir = r"\tmp_policy"
    txt_privacy_dir = r"\tmp_policy_txt" # English test
    # txt_privacy_zh = "..\\test\\"
    result_privacy_dir = r"\tmp_policy_result"
    data_usage_dir = r'\tmp_policy_data'
    os.makedirs(txt_privacy_dir, exist_ok=True)
    os.makedirs(result_privacy_dir, exist_ok=True)
    os.makedirs(data_usage_dir, exist_ok=True)
    # process html
    for html_name in os.listdir(html_privacy_dir):
        html_path = os.path.join(html_privacy_dir, html_name)
        if html_name.endswith('.html'):
            txt_name = html_name.replace('.html', '.txt')

            txt_file_path = os.path.join(txt_privacy_dir, txt_name)
            html2txt(html_path, txt_file_path)

            result_name = html_name.replace('.html', '.txt')
            result_path = os.path.join(result_privacy_dir, result_name)

            data_name = result_name
            data_path = os.path.join(data_usage_dir, data_name)
            txt_result(txt_file_path, result_path, data_path)
