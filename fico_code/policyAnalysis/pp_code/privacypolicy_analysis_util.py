# coding: utf-8
# Author: kaifa.zhao@connect.polyu.hk
# Copyright 2021@Kaifa Zhao (Zachary)
# Date: 2023/4/28
# System: linux
import os
import re
import phrasetree
import codecs
from nltk import PorterStemmer
import phrasetree.tree
import string

from Constants import WWWWWH, NOT_DATA, SHARE_ACTION, COLLECT_ACTION, TPL_LIST


def preprocess_sen(sen):
    # if the sentence include hww words, continue
    # if the sen is a word, return false
    if not ' ' in sen:
        return sen, False
    for w in WWWWWH:
        if w in sen.lower():
            return sen, False
    # remove unicode characters
    # 将一个句子分割成单词列表，将其中的乱码字符去除，因为要提取五元组，所以长度小于5的直接去除掉
    # {数据实体、动作、{数据类型}、{数据接收者}、{neg}}
    pattern = re.compile(r'[\uf001-\uf0ff]')
    sen = re.sub(pattern, '', sen).strip()
    tmp = sen.split(' ')
    if len(tmp) < 5:
        return sen, False

    return sen, True


def read_sentences_from_file(file_name):
    sentence = []
    with open(file_name, 'r', encoding='utf-8') as file:
        context = file.readlines()
    # f = open(file_name, 'r')
    # context = f.readlines()
    for c in context:
        s = c.split('\n')
        #
        if len(s) > 2:
            print("ERROR" + str(len(s)))
            print(s)
        s = s[0]
        #
        sen = s.split('. ')
        sen = list(filter(None, sen))
        sentence.extend(sen)
    file.close()
    return sentence


def analyze_host_app_share(hanlp_results):
    # 把所有hanlp的模型分析出来的结果保存，tok分词，dep依存句法，con成分分析，pos词性
    hanlp_token = hanlp_results['tok']
    hanlp_dep = hanlp_results['dep']
    hanlp_ptree = hanlp_results['con']
    hanlp_pos = hanlp_results['pos']

    # 传入分词和依存句法关系，返回5元组，【依存关系，依存的单词位置，依存的具体单词，单词本身位置，单词本身】存入到wordDepInfo
    wordDepInfo = token_list_to_dep_info(hanlp_token, hanlp_dep)
    # merger noun phase，token里面存储分词结果
    token = hanlp_token.copy()
    # 传入分词结果，成分分析，完整的hanlp的结果传入进去， 返回
    token = noun_merge_tree_based(token, hanlp_ptree, hanlp_results)
    verb_flag = [0 for i in token]
    DED = []
    # 用于记录长句第二个单词开始的位置
    beg_flag = 0
    # label SoC verb
    for i in range(len(hanlp_token)):
        # pos is verb and in SoC list
        if hanlp_pos[i] == 'VV' or hanlp_pos[i] == 'VERB':
            # 提取出动词
            verb = PorterStemmer().stem(hanlp_token[i])
            # 有2个词语list，两类词汇
            if (verb in SHARE_ACTION or verb in COLLECT_ACTION) and verb_flag[i] != 1:
                actor, action, data, entity, neg = [], [verb], [], [], []
                if verb in SHARE_ACTION:
                    action = ['share']
                elif verb in COLLECT_ACTION:
                    action = ['collect']

                # 根部动词下标
                root_verb_idx = i
                verb_flag[i] = 1
                if 'provid' in verb and token[i - 1] in ['you', 'user', 'users']:
                    actor.append('user')
                if i < beg_flag != 0:
                    beg_flag = 0
                    for z_idx in range(i, -1, -1):
                        if token[z_idx] in [';', '.', ',']:
                            beg_flag = z_idx + 1
                            break
                data = extract_data_obj_from_tree(verb_idx=i, hanlp_results=hanlp_results)

                if len(data) == 0 or (not check_data(data)):
                    data = []
                    continue

                for s_idx in range(beg_flag, len(wordDepInfo)):
                    # get actor and entity by "nsubj" and "dobj"
                    sactor = wordDepInfo[s_idx]
                    cur_tok = token[s_idx]
                    # sentence break
                    if cur_tok in [';', '.']:
                        if s_idx < i:
                            continue
                        #

                        tt = [len(t) for t in [actor, action, data, entity, neg]]

                        if tt.count(0) < 4:
                            if len(data) == 0:  # or (not check_data(data)
                                continue

                            data = list(set(data))
                            actor = list(set(actor))
                            if not [actor, action, data, entity, neg] in DED:
                                DED.append([actor, action, data, entity, neg])
                        actor, action, data, entity, neg = [], [], [], [], []
                        beg_flag = s_idx + 1
                        break

                    if cur_tok in string.punctuation or sactor[0] == 'punct':
                        continue

                    ##
                    if sactor[1] - 1 == root_verb_idx:

                        if sactor[0] == 'nsubj':
                            actor.append(cur_tok)
                        elif sactor[0] == 'pobj':
                            # if not check_data(cur_tok):
                            #   continue
                            entity.append(cur_tok)
                        elif cur_tok in ['we', 'us', 'you'] and 'provid' in verb:
                            entity.append(cur_tok)
                        elif cur_tok in ['we', 'us', 'you']:
                            actor.append(cur_tok)
                        elif sactor[0] == 'accmod' or sactor[0] == 'mmod' or sactor[0] == 'advmod':
                            if cur_tok in ['no', 'not', 'never', 'won\'t']:
                                neg.append('negative')
                            if cur_tok == 'if':
                                actor, action, data, entity, neg = [], [], [], [], []
                                beg_flag = s_idx + 1
                                break
                        elif sactor[0] == 'neg':
                            neg.append(cur_tok)
                        elif sactor[0] == 'dep' or sactor[0] == 'conj':
                            # coordinating verb
                            if cur_tok in COLLECT_ACTION or cur_tok in SHARE_ACTION:
                                actor.append(cur_tok)
                                data.extend(extract_data_obj_from_tree(s_idx, hanlp_results))
                                verb_flag[s_idx] = 1
                        elif sactor[0] == 'mark':
                            tmp_actor, tmp_entity = pattern_for_complement(' '.join(hanlp_token))
                            if tmp_actor != []:
                                actor.extend(tmp_actor)
                            if tmp_entity != []:
                                entity.extend(tmp_entity)
                    elif sactor[3] - 1 == root_verb_idx:
                        # the word itself
                        if sactor[0] == 'pcomp' or sactor[0] == 'conj' or sactor[0] == 'ccomp':
                            if sactor[2] == sactor[-1]:
                                continue
                            # （介词补语， 找到主语） or (并列动词，找到主语）
                            if hanlp_pos[sactor[1] - 1] == 'VV' or hanlp_pos[sactor[1] - 1] == 'VERB':
                                tmp_actor = get_ppcomp_subj(wordDepInfo, s_idx, hanlp_pos, token)
                                actor.extend(tmp_actor)
                        elif sactor[0] == 'dep' and sactor[2] in ['no', 'not']:
                            neg.append('negative')
                            tar_idx = sactor[1] - 1
                            for ti in range(tar_idx + 1, s_idx):
                                if wordDepInfo[ti][1] - 1 == tar_idx and wordDepInfo[ti][0] == 'dobj':
                                    actor.append(token[wordDepInfo[ti][-2] - 1])
                        elif sactor[0] == 'dep':
                            if sactor[2] == sactor[-1]:
                                continue
                            tar_idx = sactor[1] - 1

                            for ti in range(s_idx, -1, -1):
                                if wordDepInfo[ti][1] - 1 == tar_idx and wordDepInfo[ti][0] == 'nsubj':
                                    actor.append(wordDepInfo[ti][-1])
                                if wordDepInfo[ti][1] - 1 == tar_idx and wordDepInfo[ti][0] == 'dobj' and (
                                        wordDepInfo[ti][-1]).lower() in ['you', 'user', 'users']:
                                    actor.append(wordDepInfo[ti][-1])
                                    break
                    # add ner info here

                    if (hanlp_pos[s_idx] == "VV" or hanlp_pos[s_idx] == "VERB") and s_idx > i:
                        if cur_tok in SHARE_ACTION or cur_tok in COLLECT_ACTION:
                            tt = [len(t) for t in [actor, action, data, entity, neg]]
                            if tt.count(0) < 4:
                                if "service" in data or len(data) == 0:
                                    break

                                DED.append([actor, action, data, entity, neg])
                            actor = []
                            action = [cur_tok]
                            data = []
                            entity = []
                            neg = []
                            beg_flag = s_idx + 1
                            break

                if not check_data(data):
                    data = []
                tt = [len(t) for t in [actor, action, data, entity, neg]]
                if tt.count(0) < 4:
                    if (not check_data(data)) or len(data) == 0:
                        continue
                    if len(data) == 1 and (not check_data(data)):
                        continue
                    data = list(set(data))
                    if not [actor, action, data, entity, neg] in DED:
                        DED.append([actor, action, data, entity, neg])
    if len(DED) == 0:

        sentence = ' '.join(hanlp_token)

        if sentence.lower().startswith('for example'):
            nouns = extract_noun_subtree(hanlp_ptree, [])
            DED = [[['app'], ['collect'], nouns]]
        else:
            DED = extract_use_pattern_en(sentence, wordDepInfo, token)
    if len(DED) == 0:
        return []
    ###
    tmp = list(set(hanlp_token).intersection(set(TPL_LIST)))
    if len(tmp) > 0:
        DED.extend([['TPLs'], ['are'], tmp])

    return DED


def analyze_host_app_pp(file_name, save_name, data_name, hanlp_mtl, print_flag=True):
    if os.path.exists(save_name):
        # return
        print("exist")
    sentence = read_sentences_from_file(file_name)

    data_set = set()

    fs = open(save_name, 'w', encoding='utf-8')
    for sen in sentence:
        # preprocess sentence
        sen, c_flag = preprocess_sen(sen)
        if not c_flag:
            continue
        #
        print(sen)
        if len(sen) < 10:
            continue
        try:
            hanlp_results = hanlp_mtl(sen)
        except Exception:
            print('\t cannot be handled by hanlp:\t' + sen)
            continue
        #
        if len(hanlp_results.keys()) == 0:
            continue
        DED = analyze_host_app_share(hanlp_results)
        fs.write(sen + ':\n')
        for z_tmp in DED:
            fs.write('\t' + str(z_tmp) + '\n')

            # get data_usage
            if len(z_tmp) >= 3:
                for data in z_tmp[2]:
                    data_set.add(data)
    fs.close()

    with open(data_name, 'w', encoding='utf-8') as f:
        for data in data_set:
            f.write(data+'\n')


def get_data_usage(file_name, save_name, hanlp_mtl, print_flag=True):
    if os.path.exists(save_name):
        return
    # fs = open(save_name, 'w')
    sentence = read_sentences_from_file(file_name)
    previous_sen = ""
    for sen in sentence:
        # preprocess sentence
        # 可以判断是否为单词，并且长度要大于5，因为返回的是{app, share, {personal information}, {service provider}, false}这个5元组
        sen, c_flag = preprocess_sen(sen)
        if not c_flag:
            continue
        #
        print(sen)
        if len(sen) < 10:
            continue
        try:
            hanlp_results = hanlp_mtl(sen)
        except Exception:
            print('\t cannot be handled by hanlp:\t' + sen)
            continue
        #
        if len(hanlp_results.keys()) == 0:
            continue

        DED, sen = extract_soc_pattern_en(sen, hanlp_results)
        write_flag = filter_soc(DED, sen)
        # if print_flag:
        #     print(DED)
        #     print('\n')
        if write_flag:
            with open(save_name, 'w', encoding='utf-8') as fs:
                fs.write(sen + ':\n')
                for z_tmp in DED:
                    fs.write('\t' + str(z_tmp) + '\n')


def extract_soc_pattern_en(sen, hanlp_results):
    #
    DED = extract_parse_tree(hanlp_results)
    DED = post_process_soc(DED)
    return DED, sen


def extract_parse_tree(hanlp_results):
    #
    hanlp_token = hanlp_results['tok']
    hanlp_dep = hanlp_results['dep']
    hanlp_ptree = hanlp_results['con']
    hanlp_pos = hanlp_results['pos']

    wordDepInfo = token_list_to_dep_info(hanlp_token, hanlp_dep)
    # merger noun phase
    token = hanlp_token.copy()
    token = noun_merge_tree_based(token, hanlp_ptree, hanlp_results)
    verb_flag = [0 for i in token]
    DED = []
    # 用于记录长句第二个单词开始的位置
    beg_flag = 0
    # label SoC verb
    for i in range(len(hanlp_token)):
        # pos is verb and in SoC list
        if hanlp_pos[i] == 'VV' or hanlp_pos[i] == 'VERB':
            verb = PorterStemmer().stem(hanlp_token[i])
            if (verb in SHARE_ACTION or verb in COLLECT_ACTION) and verb_flag[i] != 1:
                actor, action, data, entity, neg = [], [verb], [], [], []
                if verb in SHARE_ACTION:
                    action = ['share']
                elif verb in COLLECT_ACTION:
                    action = ['collect']

                root_verb_idx = i
                verb_flag[i] = 1
                if 'provid' in verb and token[i - 1] in ['you', 'user', 'users']:
                    actor.append('user')
                if i < beg_flag and beg_flag != 0:
                    beg_flag = 0
                    for z_idx in range(i, -1, -1):
                        if token[z_idx] in [';', '.', ',']:
                            beg_flag = z_idx + 1
                            break
                data = extract_data_obj_from_tree(verb_idx=i, hanlp_results=hanlp_results)

                if len(data) == 0 or (not check_data(data)):
                    data = []
                    continue

                for s_idx in range(beg_flag, len(wordDepInfo)):
                    # get actor and entity by "nsubj" and "dobj"
                    sactor = wordDepInfo[s_idx]
                    cur_tok = token[s_idx]
                    # sentence break
                    if cur_tok in [';', '.']:
                        if s_idx < i:
                            continue
                        #

                        tt = [len(t) for t in [actor, action, data, entity, neg]]

                        if tt.count(0) < 4:
                            if len(data) == 0:  # or (not check_data(data)
                                continue

                            data = list(set(data))
                            actor = list(set(actor))
                            if not [actor, action, data, entity, neg] in DED:
                                DED.append([actor, action, data, entity, neg])
                        actor, action, data, entity, neg = [], [], [], [], []
                        beg_flag = s_idx + 1
                        break

                    if cur_tok in string.punctuation or sactor[0] == 'punct':
                        continue

                    ##
                    if sactor[1] - 1 == root_verb_idx:

                        if sactor[0] == 'nsubj':
                            actor.append(cur_tok)
                        elif sactor[0] == 'pobj':
                            # if not check_data(cur_tok):
                            #   continue
                            entity.append(cur_tok)
                        elif cur_tok in ['we', 'us', 'you'] and 'provid' in verb:
                            entity.append(cur_tok)
                        elif cur_tok in ['we', 'us', 'you']:
                            actor.append(cur_tok)
                        elif sactor[0] == 'accmod' or sactor[0] == 'mmod' or sactor[0] == 'advmod':
                            if cur_tok in ['no', 'not', 'never', 'won\'t']:
                                neg.append('negative')
                            if cur_tok == 'if':
                                actor, action, data, entity, neg = [], [], [], [], []
                                beg_flag = s_idx + 1
                                break
                        elif sactor[0] == 'neg':
                            neg.append(cur_tok)
                        elif sactor[0] == 'dep' or sactor[0] == 'conj':
                            # coordinating verb
                            if cur_tok in COLLECT_ACTION or cur_tok in SHARE_ACTION:
                                actor.append(cur_tok)
                                data.extend(extract_data_obj_from_tree(s_idx, hanlp_results))
                                verb_flag[s_idx] = 1
                        elif sactor[0] == 'mark':
                            tmp_actor, tmp_entity = pattern_for_complement(' '.join(hanlp_token))
                            if tmp_actor != []:
                                actor.extend(tmp_actor)
                            if tmp_entity != []:
                                entity.extend(tmp_entity)
                    elif sactor[3] - 1 == root_verb_idx:
                        # the word itself
                        if sactor[0] == 'pcomp' or sactor[0] == 'conj' or sactor[0] == 'ccomp':
                            if sactor[2] == sactor[-1]:
                                continue
                            # （介词补语， 找到主语） or (并列动词，找到主语）
                            if hanlp_pos[sactor[1] - 1] == 'VV' or hanlp_pos[sactor[1] - 1] == 'VERB':
                                tmp_actor = get_ppcomp_subj(wordDepInfo, s_idx, hanlp_pos, token)
                                actor.extend(tmp_actor)
                        elif sactor[0] == 'dep' and sactor[2] in ['no', 'not']:
                            neg.append('negative')
                            tar_idx = sactor[1] - 1
                            for ti in range(tar_idx + 1, s_idx):
                                if wordDepInfo[ti][1] - 1 == tar_idx and wordDepInfo[ti][0] == 'dobj':
                                    actor.append(token[wordDepInfo[ti][-2] - 1])
                        elif sactor[0] == 'dep':
                            if sactor[2] == sactor[-1]:
                                continue
                            tar_idx = sactor[1] - 1

                            for ti in range(s_idx, -1, -1):
                                if wordDepInfo[ti][1] - 1 == tar_idx and wordDepInfo[ti][0] == 'nsubj':
                                    actor.append(wordDepInfo[ti][-1])
                                if wordDepInfo[ti][1] - 1 == tar_idx and wordDepInfo[ti][0] == 'dobj' and (
                                        wordDepInfo[ti][-1]).lower() in ['you', 'user', 'users']:
                                    actor.append(wordDepInfo[ti][-1])
                                    break
                    # add ner info here

                    if (hanlp_pos[s_idx] == "VV" or hanlp_pos[s_idx] == "VERB") and s_idx > i:
                        if cur_tok in SHARE_ACTION or cur_tok in COLLECT_ACTION:
                            tt = [len(t) for t in [actor, action, data, entity, neg]]
                            if tt.count(0) < 4:
                                if "service" in data or len(data) == 0:
                                    break

                                DED.append([actor, action, data, entity, neg])
                            actor = []
                            action = [cur_tok]
                            data = []
                            entity = []
                            neg = []
                            beg_flag = s_idx + 1
                            break

                if not check_data(data):
                    data = []
                tt = [len(t) for t in [actor, action, data, entity, neg]]
                if tt.count(0) < 4:
                    if (not check_data(data)) or len(data) == 0:
                        continue
                    if len(data) == 1 and (not check_data(data)):
                        continue
                    data = list(set(data))
                    if not [actor, action, data, entity, neg] in DED:
                        DED.append([actor, action, data, entity, neg])
    if len(DED) == 0:

        sentence = ' '.join(hanlp_token)

        if sentence.lower().startswith('for example'):
            nouns = extract_noun_subtree(hanlp_ptree, [])
            DED = [[['app'], ['collect'], nouns]]
        else:
            DED = extract_use_pattern_en(sentence, wordDepInfo, token)
    if len(DED) == 0:
        return []

    return DED


def extract_use_pattern_en(sentence, wordDepInfo, token):
    data = []
    pattern1 = r'(.*)you(.*)[provide|store|save|fill](.*)information(.*)'

    if 'following information:' in sentence or 'including:' in sentence:
        for info in wordDepInfo:
            if info[2] == 'information' and (info[0] == 'nn' or info[0] == 'conj'):
                data.append(token[info[3] - 1])
            elif info[-1] == 'information:':
                data.append(token[info[3] - 1])
        return [[['app'], ['collect'], list(set(data)), [], []]]
    else:
        for i in ['provide', 'store', 'save', 'fill']:
            if i in sentence:
                re_tmp = re.match(pattern1, sentence)
                if re_tmp is not None:
                    beg_idx = re_tmp.regs[3][1]
                    end_idx = beg_idx + 1
                    ii = 0
                    tar_idx = -1
                    for idx, tok in enumerate(wordDepInfo):
                        if ii <= beg_idx and (ii + len(tok[-1])) >= end_idx:
                            tar_idx = idx
                            break
                        ii = ii + len(tok[-1])
                    if tar_idx == -1:
                        return []
                    else:
                        # data.append(tok[-1])
                        data.append(token[tar_idx])
                    # 根据依赖关系找并列谓语
                    data.extend(find_conj_noun(wordDepInfo, data, idx, token))
                    if list(set(data)) == ["information"]:
                        return []
                    tmp = [[["user"], [i], list(set(data)), [], []]]
                    return tmp
    return []


def token_list_to_dep_info(tokenResult, depResult):
    depInfoWords = []
    for tokenIndex in range(1, len(tokenResult) + 1):
        depWord = tokenResult[tokenIndex - 1]
        depInfo = depResult[tokenIndex - 1]
        govWordIndex = depInfo[0]
        govWord = tokenResult[govWordIndex - 1]
        rel = depInfo[1]
        # print(tokenIndex, depWord, '<-%s-' % rel, govWordIndex, govWord)
        depInfoWords.append([rel, govWordIndex, govWord, tokenIndex, depWord])
    # depInfoWords返回5元组，【依存关系，依存的单词位置，依存的具体单词，单词本身位置，单词本身】
    return depInfoWords


# 传入分词结果，成分分析，完整的hanlp的结果,合并名词短语返回
def noun_merge_tree_based(token, cur_tree, han_result):
    # 分词结果
    raw_token = han_result['tok']
    # cixing = han_result['pos/ctb']

    for i in cur_tree:
        if type(i) is phrasetree.tree.Tree:
            if len(i.leaves()) > 5:
                token = noun_merge_tree_based(token, i, han_result)
            elif i.label() == "NP":
                if i.leaves()[0] == "if":
                    token = noun_merge_tree_based(token, i, han_result)
                    continue
                lexicon = i.leaves()
                noun = " ".join(lexicon)
                # locate_noun 用于定位词语位置
                idx = locate_noun(lexicon, raw_token)
                # idx = raw_token.index(lexicon[-1])
                if idx == -1:
                    continue
                else:
                    for punc in [",", "or", "and"]:
                        if punc in noun:
                            noun = noun.split(punc)[-1]
                token[idx] = noun
            elif "NP" in i.pformat_latex_qtree():
                token = noun_merge_tree_based(token, i, han_result)
                # print(tree2string(i))
                # if cixing[idx] == "NN":
                #     token[idx] = noun
    # print(token)
    return token


def filter_soc(DED, sen):
    if 'send us email' in sen.lower() or 'send us an email' in sen.lower():
        return False

    return True


def locate_noun(lexicon, raw_token):
    if len(lexicon) > 1:
        for idx, i in enumerate(lexicon):
            if raw_token.count(i) == 1:
                return raw_token.index(i) + len(lexicon) - idx - 1

        for idx in range(0, len(lexicon) - 1):
            i = lexicon[idx]
            for z1 in range(raw_token.index(i), len(raw_token) - len(lexicon)):
                if raw_token[z1] == i and raw_token[z1 + 1] == lexicon[idx + 1]:
                    return z1 + len(lexicon) - idx - 1

        # for idx, i in enumerate(lexicon):
        #     for z1 in range(raw_token.index(i), len(raw_token) - len(lexicon)):
        #         if raw_token[z1] == i and raw_token[z1 + 1] == lexicon[idx + 1]:
        #             return z1 + len(lexicon) - idx - 1
    else:
        if raw_token.count(lexicon[-1]) == 1:
            return raw_token.index(lexicon[-1])

    # if raw_token.count(lexicon[-1]) == 1:
    #     return raw_token.index(lexicon[-1])
    # else:
    #     tmp = []
    # for i in lexicon:
    #     tmp.append(raw_token.index(i))
    # for i in range(len(tmp) - 1):
    #     if tmp[i + 1] - tmp[i] == 1:
    #         return tmp[i] + len(tmp) - i - 1

    return -1


def check_data(data):
    for tmp in data:
        if 'service' in tmp or tmp == 'you' or tmp == 'your':
            return False
        for d in NOT_DATA:
            if len(inter(d, tmp)) > 0:
                return False

    if data in NOT_DATA:
        return False
    return True


def inter(a, b):
    return list(set(a.split(' ')) & set(b.split(' ')))


def find_conj_noun(depInfo, data, tarIdx, tokens):
    # 根据依赖关系找并列谓语
    for z1, info in enumerate(depInfo):
        if info[1] == tarIdx and info[0] == 'conj':
            data.append(tokens[z1])
        elif info[1] == tarIdx and info[0] == 'nn':
            data.append(tokens[z1])
        elif z1 == tarIdx and info[0] == 'dep':
            data.append(tokens[info[1] - 1])
            data.extend(find_conj_noun(depInfo, data, info[1], tokens))
        elif info[3] - 1 == tarIdx and info[0] == "dobj":
            # 查找同动词宾语
            # demo sentence: "您使用我们的服务时所储存的信息，如姓名、性别、出生日期、出生地、户籍所在地、民族、联系电话、收件地址、电子邮箱、联系人信息等。"
            data.extend(find_same_verb_noun(depInfo, data, info[1], tokens))
    return data


def find_same_verb_noun(depInfo, data, tarIdx, tokens):
    # 查找同动词宾语
    for z1 in range(tarIdx, len(depInfo)):
        info = depInfo[z1]
        if info[1] == tarIdx and info[0] == "dobj":
            data.append(info[-1])
            data.extend(find_conj_noun(depInfo, data, info[3], tokens))
    return list(set(data))


def extract_noun_subtree(subtree, noun):
    # tree_str = subtree.performat_latex_qtree()
    tree_str = str(subtree)
    if not ('CC' in tree_str or 'PU' in tree_str or 'SBAR' in tree_str):
        return [' '.join(subtree.leaves())]
    if subtree.label() in ['NP', 'NN'] and len(subtree.leaves()) < 7 and (not 'SBAR' in tree_str):
        return [' '.join(subtree.leaves())]

    for i in subtree:
        if type(i) is phrasetree.tree.Tree:
            if 'SBAR' in i.pformat_latex_qtree():
                continue
            if 'CC' in i.pformat_latex_qtree():
                noun.extend(extract_noun_subtree(i, []))
            elif i.label() in ['NP', 'NN'] and len(i.leaves()) < 7:
                noun.append(' '.join(i.leaves()))
            elif i.label() in ['NP', 'NN'] and len(i.leaves()) >= 7:
                noun.extend(extract_noun_subtree(i, []))
            elif (not i.label() in ['NP', 'NN']) and len(i.leaves()) > 5:
                noun.extend(extract_noun_subtree(i, []))
    return noun


def post_process_soc(raw_soc):
    new_soc = []
    for idx, soc in enumerate(raw_soc):
        data = soc[2]
        tmp_idx = []
        for i, d in enumerate(data):
            if d.startswith("your"):
                data[i] = d.replace("your", "")
            if "/" in d:
                data.extend(d.split("/"))
                tmp_idx.append(i)
                # data.remove(d)
            elif "、" in d:
                data.extend(d.split("、"))
                tmp_idx.append(i)
                # data.remove(d)
            if not check_data(d):
                tmp_idx.append(i)
        for ziiii in tmp_idx[::-1]:
            data.pop(ziiii)
        soc[2] = list(set(data))
        soc[1] = list(set(soc[1]))

        # for
        if len(data) == 1 and len(data[0]) < 2:
            raw_soc.pop(idx)
            continue
        new_soc.append(soc)
        if len(data) == 0:
            return []
    return new_soc


def extract_data_obj_from_tree(verb_idx, hanlp_results):
    # 成分分析结果
    hanlp_tree = hanlp_results['con']

    tree_idx = hanlp_tree.leaf_treeposition(verb_idx)
    subtree = hanlp_tree
    for idx, num in enumerate(tree_idx[:-2]):
        subtree = subtree[num]
    noun_tree_id = tree_idx[-2] + 1
    nountree = None
    if len(subtree) > noun_tree_id:
        if subtree[noun_tree_id].label() in ['AS', 'PU']:
            if len(subtree) > tree_idx[-2] + 2:
                nountree = subtree[tree_idx[-2] + 2]
            else:
                return []
        if subtree[noun_tree_id].label() in ['NP', 'NN']:
            nountree = subtree[noun_tree_id]
    else:
        return []

    if nountree is None or not nountree.label() in ['NP', 'NN']:
        return []
    if len(nountree) == 1 and nountree[0].label() == "PRON" and (len(subtree) > (noun_tree_id + 1)):
        nountree = subtree[noun_tree_id + 1]
    noun = []
    noun = extract_noun_subtree(nountree, noun)
    return noun


def pattern_for_complement(sentence):
    pattern1 = r'(.*)we(.*)ask(.*)you(.*)provid(.*)'
    re_tmp = re.match(pattern1, sentence)
    if re_tmp is not None:
        return ['user'], ['app']
    return [], []


def get_ppcomp_subj(wordDepInfo, s_idx, hanlp_pos, token):
    actor = []
    # find the verb
    cur_dep = wordDepInfo[s_idx]
    tar_idx = cur_dep[1] - 1
    tar_pos = hanlp_pos[tar_idx]

    count = 0
    while tar_pos != "VV":
        cur_dep = wordDepInfo[tar_idx]
        tar_idx = cur_dep[1] - 1
        tar_pos = hanlp_pos[tar_idx]
        count += 1
        if count > len(hanlp_pos):
            break
    if tar_pos != 'VV':
        return []

    for i in range(0, tar_idx):
        cur_dep = wordDepInfo[i]
        if cur_dep[1] - 1 == tar_idx:
            if cur_dep[0] == "nsubj":
                actor.append(token[i])

    return actor
