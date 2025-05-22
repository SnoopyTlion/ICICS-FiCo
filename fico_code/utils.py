import json
import math
import re
import sys
import xml.dom.minidom as minidom
import hashlib
import xml.etree.ElementTree as ET
import unicodedata
import uiautomator2 as u2
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


direction_UP = 'up'
direction_DOWN = 'down'
direction_LEFT = 'left'
direction_RIGHT = 'right'

ui_type_UNKNOWN = 'UI_unknown'  # need to know what type an ui is, just clickable
ui_type_VIEW = 'UI_view' # a normal clickable view, no need further analysis
ui_type_DIALOG = 'UI_dialog'  # dialog, a node responsible for user enter.
ui_type_INPUT = 'UI_input'  # input, a node responsible for user enter.
ui_type_INDIRECT_INPUT = 'UI_indirect_input' # indirect input, responsible for user enter, need explore further.
ui_type_TEXT = 'UI_text' # label node
ui_types_INTEREACT = [ui_type_DIALOG, ui_type_INPUT, ui_type_INDIRECT_INPUT]

edge_type_CLICK = 'click'
edge_type_SWIPE = 'swipe'

interaction_component = ['android.widget.EditText', 'android.widget.AutoCompleteTextView',
                         'android.widget.CompoundButton',
                         'android.widget.Switch', 'android.widget.CheckBox', 'android.widget.RadioButton',
                         'android.widget.NumberPicker', 'android.widget.ToggleButton', 'android.widget.CheckedTextView',
                         'android.widget.Spinner', ]

def get_normal_chinese(text: str):
    # drop all special escape characters, such as space, line breaks, hex-characters.
    special_chars = []
    code_names = ['CJK', 'SPACE', 'LATIN', 'DIGIT']
    for char in text:
        # english or space, normal chars
        if char.isalpha() or char == ' ':
            continue
        if char == '(' or char == ')' or char.isdecimal():
            continue
        char_name = unicodedata.name(char, "")
        normal = False
        # a char is not Chinese and is not regular encoding
        for code_name in code_names:
            if code_name in char_name:
                normal = True
            if not normal:
                special_chars.append(char)
    for char in special_chars:
        text = text.replace(char, '')
    return text


def dfs_traverse(node: ET.fromstring, nodes_inserted: set, root_node_insert: set, entry_node: ET.Element, visited: set):
    """
    need find all root nodes of the inserted node. The entry node of it is not contained by `node_inserted`, but itself is contained by `node_inserted`.
    :param node:
    :param nodes_inserted:
    :param root_node_insert:
    :param entry_node: the parent node.
    :param visited:
    :return:
    """
    if node in visited:
        return
    visited.add(node)
    if entry_node not in nodes_inserted and node in nodes_inserted:
        root_node_insert.add(node)
    for sub_node in node:
        dfs_traverse(sub_node, nodes_inserted, root_node_insert, node, visited)


def text_in_inner_node(node: ET.Element):
    for child_node in node.iter():
        if node == child_node:
            continue
        if 'text' in child_node.attrib and child_node.attrib['text'].strip() != '':
            return True
    return False


def has_resource_id(node: ET.Element):
    meaningful_resource = ['edit', 'more', 'setting', 'user', 'account', 'home', 'search', 'saved']
    has_meaningful_resource = False
    if 'resource-id' in node.attrib and node.attrib['resource-id'].strip() != '':
        for resource in meaningful_resource:
            if resource in node.attrib['resource-id'].lower():
                has_meaningful_resource = True
    return has_meaningful_resource

def resource_id_resolve(node: ET.Element):
    if 'resource-id' in node.attrib and node.attrib['resource-id'].strip() != '':
        idx = node.attrib['resource-id'].find(':id')
        if idx != -1:
            node.attrib['resource-id'] = node.attrib['resource-id'][idx + 4:]
            if 'edit' in node.attrib['resource-id'].lower():
                node.attrib['text'] = 'edit'
            elif 'more' in node.attrib['resource-id'].lower():
                node.attrib['text'] = 'more'
            elif 'setting' in node.attrib['resource-id'].lower():
                node.attrib['text'] = 'settings'
            elif 'home' in node.attrib['resource-id'].lower():
                node.attrib['text'] = 'home'
            elif 'search' in node.attrib['resource-id'].lower():
                node.attrib['text'] = 'search'
            elif 'saved' in node.attrib['resource-id'].lower():
                node.attrib['text'] = 'saved'
            elif 'setup_userheader' in node.attrib['resource-id'].lower():
                node.attrib['text'] = 'user'
            # elif 'me' in node.attrib['resource-id'].lower():
            #     node.attrib['text'] = 'my'

        could_set_clickable = False
        for sub_node in node:
            could_set_clickable |= dfs_transfer_clickable(sub_node, False, False)
        if could_set_clickable:
            node.attrib['clickable'] = 'true'


def dfs_transfer_clickable(node: ET.Element, has_clickable, has_text_info):
    # If a node's `text` attribute is not empty, and its inner node has `onclick`, then transfer the `onclick` the node.
    # Note that, use dfs to traverse the node.
    # If all nodes in one path of dfs have empty `text`, `content-desc` and `resource-id`, and a node in this path has ture `clickable`,
    # then set the `clickable` attribute of this node to True.
    # We do that to transfer the actual information which users interact with.
    transfer_state = False
    cur_node_has_clickable = node.attrib['clickable'] == 'true'
    cur_node_has_text_info = node.attrib['text'].strip() != '' or node.attrib['content-desc'] != '' and node.attrib['content-desc'] is not None or has_resource_id(node)
    if len(node) == 0:
        return (has_clickable | cur_node_has_clickable) and not has_text_info and not cur_node_has_text_info
    for sub_node in node:
        transfer_state |= dfs_transfer_clickable(sub_node, cur_node_has_clickable | has_clickable, cur_node_has_text_info | has_text_info)
    return transfer_state


def get_xml_from_view_tree(device):
    """
    Transfer the result of dump_hierarchy to xml structure, and do a series of preprocess.
    1. drop the system toolbar and keyboard.
    2. transfer Imageview to text
    :param device:
    :param convertor:
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
        # set the `clickable` attribute to true in case it is false.
        if 'class' in node.attrib:
            if node.attrib['class'] in interaction_component:
                node.attrib['clickable'] = 'true'
        if 'text' in node.attrib:
            node.attrib['text'] = get_normal_chinese(node.attrib['text'])
            if text_in_inner_node(node):
                continue
            # some label node would occur in form of Custom View.
            # in this case, the text information would be contained in `content-desc`, so add logic to identity `content-desc`
            if node.attrib['text'] == '':
                if node.attrib['content-desc'] != '' and node.attrib['content-desc'] is not None and '_' not in node.attrib['content-desc']:
                    node.attrib['text'] = node.attrib['content-desc']
                elif has_resource_id(node):
                    resource_id_resolve(node)
            if node.attrib['text'].lower() == 'go to profile':
                node.attrib['text'] = 'profile'
            if 'my profile' in node.attrib['text'].lower():
                node.attrib['text'] = 'profile'
            if 'network' in node.attrib['text'].lower():
                node.attrib['text'] = 'network'
    return root_element


def parse_corr(bounds):
    pattern = r'\[(.*),(.*)\]\[(.*),(.*)\]'
    corr = []
    mat_res = re.match(pattern=pattern, string=bounds)
    x1, y1, x2, y2 = -1, -1, -1, -1
    if mat_res:
        x1, y1, x2, y2 = int(mat_res.group(1)), int(mat_res.group(2)), int(mat_res.group(3)), int(mat_res.group(4))
    if x1 == -1:
        print('location error! ', bounds)
        return corr
    corr = [x1, y1, x2, y2]
    return corr


def calculate_angle(bound_n: str, bound_m: str):
    n_corr = parse_corr(bound_n)
    m_corr = parse_corr(bound_m)
    angle = 360

    if abs(n_corr[0]-m_corr[0]) < 100 or abs(n_corr[2]-m_corr[2]) < 100 or abs((n_corr[0] + n_corr[2]) / 2 - (m_corr[0] + m_corr[2]) / 2) < 10:
        angle = 90.0
    elif abs(n_corr[1]-m_corr[1]) < 10 or abs(n_corr[3]-m_corr[3]) < 10 or abs((n_corr[1] + n_corr[3]) / 2 - (m_corr[1] + m_corr[3]) / 2) < 10:
        angle = 0
    else:
        n_point = [[n_corr[0], n_corr[1]], [n_corr[0], n_corr[3]], [n_corr[2], n_corr[1]], [n_corr[2], n_corr[3]]]
        m_point = [[m_corr[0], m_corr[1]], [m_corr[0], m_corr[3]], [m_corr[2], m_corr[1]], [m_corr[2], m_corr[3]]]
        tmp_angles = []
        for i in range(4):
            x1, y1 = n_point[i]
            x2, y2 = m_point[i]
            slope = (y2 - y1) / (x2 - x1)
            t_angle = math.degrees(math.atan(slope))
            if t_angle < 0:
                t_angle = abs(t_angle)
            tmp_angles.append(t_angle)
        closest_corr = 90
        for t_angle in tmp_angles:
            if min(90-t_angle, t_angle) < closest_corr:
                closest_corr = min(90-t_angle, t_angle)
                angle = t_angle

    return int(angle)


def calculate_relative_location(bound_n: str, bound_m: str):
    n_corr = parse_corr(bound_n)
    m_corr = parse_corr(bound_m)

    n_point = [[n_corr[0], n_corr[1]], [n_corr[0], n_corr[3]], [n_corr[2], n_corr[1]], [n_corr[2], n_corr[3]]]
    m_point = [[m_corr[0], m_corr[1]], [m_corr[0], m_corr[3]], [m_corr[2], m_corr[1]], [m_corr[2], m_corr[3]]]
    distance = 99999
    for n_i in n_point:
        for m_i in m_point:
            distance = min(math.sqrt((n_i[0]-m_i[0])**2+(n_i[1]-m_i[1])**2), distance)
    return distance


def pretty_view_tree(root_element):
    # create an empty dom
    doc = minidom.Document()

    # add Element to the dom
    root_node = doc.createElement(root_element.tag)
    doc.appendChild(root_node)

    # add child node and attribute.
    def add_element(parent, element):
        node = doc.createElement(element.tag)
        parent.appendChild(node)
        for key, value in element.attrib.items():
            node.setAttribute(key, value)
        for child in element:
            add_element(node, child)

    add_element(root_node, root_element)
    xml_content = doc.toprettyxml(indent="  ")
    print(xml_content)


ui_path_count_dict = dict()
def count_ui_path(ui_path: str):
    print('### ui path: ', ui_path)
    ui_path_list = ui_path.split('-')
    if len(ui_path_list) < 2:
        ui_path_t = '-'.join(ui_path_list[:-1])
        if ui_path_t == '':
            ui_path_t = '#'
        if ui_path_t not in ui_path_count_dict:
            ui_path_count_dict[ui_path_t] = 0
        ui_path_count_dict[ui_path_t] += 1
        if ui_path_count_dict[ui_path_t] > 15:
            return False
    else:
        for i in range(2, len(ui_path_list)):
            ui_path_t = '-'.join(ui_path_list[:i])
            if ui_path_t not in ui_path_count_dict:
                ui_path_count_dict[ui_path_t] = 0
            ui_path_count_dict[ui_path_t] += 1
            if ui_path_count_dict[ui_path_t] > (5-i)*10:
                return False
            print('ui_path_limitation: ', ui_path_count_dict[ui_path_t], ': ', (5-i)*10)
    return True



def is_privacy_keyword(text, privacy_keywords):
    """
    Check whether a data item is in privacy keyword list or not.
    If checked, the data item is privacy item.
    :return:
    """
    if text.lower() in privacy_keywords:
        return True
    for keyword in privacy_keywords:
        if keyword in text.lower():
            return True
    return False


def get_nlp_analysis(node: ET.Element, HanLP, nlp_cache, privacy_keywords, lang='zh'):
    """
    Analyze norm in text information to extract data item which describe user's action and judge it whether is privacy data or not.
    :return: [privacy_words, noun_words, text]
    """
    privacy_words = []
    # extract norm to prepare for extend.
    noun_words = []
    if 'text' not in node.attrib:
        return privacy_words
    text = node.attrib['text'].lower()
    if text == '':
        return privacy_words

    if HanLP is None:
        print("Initialize HanLP!")
        sys.exit(1)


    if lang == 'zh':
        tok = 'tok/fine'
        tok_coarse = 'tok/coarse'
        pos = 'pos/ctb'
        if text not in nlp_cache:
            # chinese logic
            nlp_cache[text] = HanLP(text, tasks=[tok, pos])
        nlp_doc = nlp_cache[text]

        if 'VERB' in nlp_doc[pos]:
            # analyse obj
            for i in range(len(nlp_doc[tok])):
                # Chinese logic
                if nlp_doc[pos][i] == 'NN' or nlp_doc[pos][i] == 'NOUN':
                    noun_words.append(nlp_doc[tok][i])
                    # for a norm, check whether it is in privacy keyword or not.
                    if is_privacy_keyword(nlp_doc[tok][i], privacy_keywords):
                        privacy_words.append(nlp_doc[tok][i])
        else:
            for i in range(len(nlp_doc[tok])):
                # Chinese logic
                if nlp_doc[pos][i] == 'NN' or nlp_doc[pos][i] == 'NOUN':
                    noun_words.append(nlp_doc[tok][i])
                    # for a norm, check whether it is in privacy keyword or not.
                    if is_privacy_keyword(nlp_doc[tok][i], privacy_keywords):
                        privacy_words.append(nlp_doc[tok][i])

        return [privacy_words, noun_words, text]

    tok = 'tok/fine'
    tok_en = 'tok'
    tok_coarse = 'tok/coarse'
    pos = 'pos/ctb'
    pos_en = 'pos'

    # debug
    # if text == 'sleeping habits':
    #     print('debug')

    # first do pos-tagging
    if text not in nlp_cache:
        # chinese logic
        # nlp_cache[text] = HanLP(text, tasks=[tok_en, pos_en])
        # english logic
        nlp_cache[text] = HanLP(text)
    nlp_doc = nlp_cache[text]

    if 'VERB' in nlp_doc[pos_en]:
        # analyse obj
        for i in range(len(nlp_doc[tok_en])):
            # Chinese logic
            # if nlp_doc[pos][i] == 'NN' or nlp_doc[pos][i] == 'NOUN':
            #     noun_words.append(nlp_doc[tok][i])
            #     # for a norm, check whether it is in privacy keyword or not.
            #     if is_privacy_keyword(nlp_doc[tok][i], privacy_keywords):
            #         privacy_words.append(nlp_doc[tok][i])

            # English logic
            if nlp_doc[pos_en][i] == 'NOUN' and (nlp_doc['dep'][i][1] == 'obj' or nlp_doc['dep'][i][1] == 'obl'):
                noun_words.append(nlp_doc[tok_en][i])
                # for a norm, check whether it is in privacy keyword or not.
                if is_privacy_keyword(nlp_doc[tok_en][i], privacy_keywords):
                    privacy_words.append(nlp_doc[tok_en][i])
    else:
        for i in range(len(nlp_doc[tok_en])):
            # Chinese logic
            # if nlp_doc[pos][i] == 'NN' or nlp_doc[pos][i] == 'NOUN':
            #     noun_words.append(nlp_doc[tok][i])
            #     # for a norm, check whether it is in privacy keyword or not.
            #     if is_privacy_keyword(nlp_doc[tok][i], privacy_keywords):
            #         privacy_words.append(nlp_doc[tok][i])

            # English logic
            if nlp_doc[pos_en][i] == 'NOUN':
                noun_words.append(nlp_doc[tok_en][i])
                # for a norm, check whether it is in privacy keyword or not.
                if is_privacy_keyword(nlp_doc[tok_en][i], privacy_keywords):
                    privacy_words.append(nlp_doc[tok_en][i])

    return [privacy_words, noun_words, text]


def get_width(node: ET.Element):
    corr = parse_corr(node.attrib['bounds'])
    width = corr[2] - corr[0]
    return width


def get_height(node: ET.Element):
    corr = parse_corr(node.attrib['bounds'])
    height = corr[3] - corr[1]
    return height


def calcul_similarity(vector_a: list, vector_b: list):
    vector1 = np.array(vector_a)
    vector2 = np.array(vector_b)

    # transform to 2-dimension
    vector1 = vector1.reshape(1, -1)
    vector2 = vector2.reshape(1, -1)

    # calculate cosine similarity
    similarity = cosine_similarity(vector1, vector2)

    return similarity[0][0]


if __name__ == '__main__':
    import uiautomator2 as u2
    # from global_vars import HanLP, nlp_cache, privacy_keywords
    # text = 'Where I went to school'
    # doc = HanLP(text)
    # print(doc)
    calcul_similarity([1,1,0,0], [1,1,0,0])
