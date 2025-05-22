import hanlp
import uiautomator2 as u2
import torch
from transformers import BertTokenizer
import os
# from dynamicExplore.app_explorer import *

#### replace the pkg_name
pkg_name = 'com.newleaf.app.android.victor'

'''
Total running time (second).
'''
TOTAL_TIME = 1200


# bert
model_path = r'C:\Master\PrivacyLegal\code\FiCo\model\bert-base-chinese'
bert_model_classifier = torch.load('C:\Master\PrivacyLegal\code\FiCo\model\model_stu_chinese.bin', map_location=torch.device('cpu'))
bert_model_tokenizer = BertTokenizer.from_pretrained(model_path)
bert_model_classifier.eval()

HanLP = None
HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)
# HanLP = hanlp.load(hanlp.pretrained.mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE)

nlp_cache = {}


privacy_keywords = set()


privacy_keyword_path = r'C:\Master\PrivacyLegal\code\FiCo\fico_code\config\PrivacyDataItem_zh.txt'

with open(privacy_keyword_path, 'r', encoding='utf-8') as f:
    line = f.readline()
    while line:
        line = line.strip()
        for word in line.split(','):
            privacy_keywords.add(word.strip())
        line = f.readline()

