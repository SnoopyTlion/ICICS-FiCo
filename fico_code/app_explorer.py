import copy
import os
import time
from typing import List

from operator import itemgetter
from utils import *
import logger
from customExceptions import *
from global_vars import *
import torch


def get_bounds_text_dict(hierarchy_root_element: ET.Element):
    bounds_text_dict = dict()
    for node in hierarchy_root_element.iter():
        if 'bounds' in node.attrib and 'text' in node.attrib:
            if node.attrib['text'].strip() != '':
                bounds_text_dict[node.attrib['bounds']] = node.attrib['text']
    return bounds_text_dict


def has_inner_text(node: ET.Element):
    for sub_node in node.iter():
        if 'text' in sub_node.attrib and sub_node.attrib['text'].strip() != '':
            return True
    return False


def dfs_traverse_for_pattern(click_node: ET.Element, cur_node: ET.Element, vector_pair_dict: dict, view_type_dict: dict, inner_nodes: list, bound_views: set):
    """
    consider `dispatch onclick` in android.

    :param vector_pair_dict:
    :param click_node:
    :param cur_node:
    :param inner_nodes:
    :return:
    """
    jump_class = ['android.widget.ImageView', 'android.widget.Image']
    if click_node is not None:
        if 'clickable' in cur_node.attrib and 'text' in cur_node.attrib:
            if cur_node.attrib['text'].strip() != '' and cur_node.attrib['class'] not in jump_class and cur_node.attrib['clickable'] != 'true':
                inner_nodes.append(cur_node)
            elif cur_node.attrib['class'] in jump_class:
                return

    need_resolve = False
    if 'clickable' in cur_node.attrib and 'text' in cur_node.attrib:
        if cur_node.attrib['clickable'] == 'true':
            if cur_node.attrib['text'].strip() == '':
                if has_inner_text(cur_node):
                    need_resolve = True
                    inner_nodes.clear()
                    inner_nodes = []

                    for node in inner_nodes:
                        view_type_dict[node] = ui_type_TEXT
            elif cur_node.attrib['text'].strip() != '':
                view_type_dict[cur_node] = ui_type_UNKNOWN

            if is_user_input(cur_node):
                view_type_dict[cur_node] = ui_type_INPUT
            if has_inner_text(cur_node):
                need_resolve = True
                inner_nodes.clear()
                inner_nodes = []

            click_node = cur_node

    for node in cur_node:
        dfs_traverse_for_pattern(click_node, node, vector_pair_dict, view_type_dict, inner_nodes, bound_views)


    if not need_resolve:
        return
    if len(inner_nodes) == 2:
        for i in range(len(inner_nodes)):
            for j in range(i + 1, len(inner_nodes)):
                node_i, node_j = inner_nodes[i], inner_nodes[j]
                if node_i in bound_views or node_j in bound_views:
                    continue

                i_x, i_y, _, _ = parse_corr(node_i.attrib['bounds'])
                j_x, j_y, _, _ = parse_corr(node_j.attrib['bounds'])
                if i_x < j_x or i_y < j_y:
                    node_i, node_j = node_j, node_i
                relative_distance = calculate_relative_location(node_i.attrib['bounds'], node_j.attrib['bounds'])
                angel = calculate_angle(node_i.attrib['bounds'], node_j.attrib['bounds'])

                if angel not in vector_pair_dict:
                    vector_pair_dict[angel] = []
                vector_pair_dict[angel].append((node_i, node_j, relative_distance))
                # assign top or left view as text
                view_type_dict[node_i] = ui_type_UNKNOWN
                view_type_dict[node_j] = ui_type_TEXT

                bound_views.add(node_i)
                bound_views.add(node_j)
    elif len(inner_nodes) == 1:
        view_type_dict[inner_nodes[0]] = ui_type_UNKNOWN

def is_user_input(node: ET.Element):
    """
    judge whether an interactive view is a user input
    """
    if 'class' not in node.attrib:
        return False

    if node.attrib['class'] in interaction_component:
        return True

    return False


def handle_inner_clickable_nodes_for_pattern(root_element: ET.Element, vector_pair_dict: dict, view_type_dict: dict, bound_view_groups: set):
    """
    Identity clickable view。

    :param root_element:
    :param vector_pair_dict:
    :param bound_view_groups:
    :return:
    """
    dfs_traverse_for_pattern(None, root_element, vector_pair_dict, view_type_dict, [], bound_view_groups)
    # all remaining views is textview
    for node in root_element.iter():
        if node not in view_type_dict:
            if is_user_input(node):
                view_type_dict[node] = ui_type_INPUT
            elif 'text' in node.attrib and node.attrib['text'].strip() != '':
                view_type_dict[node] = ui_type_TEXT

    # debug
    special_text = ['save']
    for view in view_type_dict:
        if view.attrib['text'].lower() in special_text:
            view_type_dict[view] = ui_type_VIEW

    # for view in view_type_dict:
    #     print(f"{view.attrib['text']}, {view_type_dict[view]}")


def calcul_angel_distance(root: ET.Element, vector_pair_dict: dict, view_type_dict: dict, bound_view_groups: set):
    """

    :param root:
    :param view_type_dict:
    :return:
    """
    # hint is -2
    SPECIAL_PATTERN = -2
    vector_pair_dict[SPECIAL_PATTERN] = []

    # use `view_type_dict` store "UI" —— "T" map
    UI_nodes, T_nodes = [], []
    for node in view_type_dict:
        if node in bound_view_groups:
            continue
        if view_type_dict[node] != ui_type_TEXT:
            UI_nodes.append(node)
        elif view_type_dict[node] == ui_type_TEXT:
            T_nodes.append(node)
    for ui_node in UI_nodes:
        if ui_node.attrib['text'] != '':
            vector_pair_dict[SPECIAL_PATTERN].append((ui_node, ui_node, 0))
        for t_node in T_nodes:
            relative_distance = calculate_relative_location(ui_node.attrib['bounds'], t_node.attrib['bounds'])
            angel = calculate_angle(ui_node.attrib['bounds'], t_node.attrib['bounds'])
            if angel not in vector_pair_dict:
                vector_pair_dict[angel] = []
            vector_pair_dict[angel].append((ui_node, t_node, relative_distance))
    # debug
    # for angel in vector_pair_dict:
    #     print('angel:', angel)
    #     for u_t_d in vector_pair_dict[angel]:
    #         print(u_t_d[0].attrib['text'], u_t_d[1].attrib['text'], u_t_d[2])


def map_text_ui(vector_pair_dict: dict):
    """
    Find label for views.
    :return:
    """
    text_ui_dict = dict()
    # sort for views in pattern
    pattern_filter_map = dict()

    for pat in vector_pair_dict:
        # map_info : (ui_view,text_view, distance)
        for i in range(len(vector_pair_dict[pat])):
            map_info = vector_pair_dict[pat][i]
            # if map_info[0].attrib['text'] == 'Add your preferred name' and map_info[1].attrib['text'] == 'Name':
            #     print('debug')
            # if map_info[1].attrib['text'] == 'Add your preferred name' and map_info[0].attrib['text'] == 'Name':
            #     print('debug')
            if map_info[0].attrib['text'].lower() in map_info[1].attrib['text'].lower() or map_info[1].attrib['text'].lower() in map_info[0].attrib['text'].lower():
                vector_pair_dict[pat][i] = (map_info[0], map_info[1], max(map_info[2]-5, 0))


    visited_views = set()


    for pat in vector_pair_dict:
        text_nodes_map = dict()
        # map_info : (ui_view, text_view, distance)
        equal_views = dict()
        for map_info in vector_pair_dict[pat]:
            if map_info[1] not in text_nodes_map:
                # if map_info[1].attrib['text'] == 'Name' or map_info[1].attrib['text'] == 'Bio':
                #     print('debug')
                #     print(map_info[1].attrib['text'],map_info[0].attrib['text'],map_info[2])
                text_nodes_map[map_info[1]] = (map_info[0], map_info[2])
            else:
                if map_info[2] <= text_nodes_map[map_info[1]][1]:
                    if map_info[2] == text_nodes_map[map_info[1]][1]:
                        if map_info[1] not in equal_views:
                            equal_views[map_info[1]] = []
                        equal_views[map_info[1]].append([map_info[0], map_info[2]])

                    # if map_info[1].attrib['text'] == 'Name' or map_info[1].attrib['text'] == 'Bio':
                    #     print('debug')
                    #     print(map_info[1].attrib['text'], map_info[0].attrib['text'], map_info[2])
                    text_nodes_map[map_info[1]] = (map_info[0], map_info[2])

            for text_node in equal_views:
                cur_view_node, cur_distance = text_nodes_map[text_node][0], text_nodes_map[text_node][1]
                for t_node in text_nodes_map:
                    if text_nodes_map[t_node][0] == cur_view_node and text_nodes_map[t_node][1] < cur_distance and t_node != text_node:
                        cur_view_node = equal_views[text_node][0][0]
                text_nodes_map[text_node] = (cur_view_node, cur_distance)

        pattern_filter_map[pat] = text_nodes_map

    # debug
    # for pat in pattern_filter_map:
    #     text_nodes_map = pattern_filter_map[pat]
    #     for text_node in text_nodes_map:
    #         print(text_node.attrib['text'], text_nodes_map[text_node][0].attrib['text'], pat, text_nodes_map[text_node][1])

    # print('----------')

    pat_len_info = []

    for pat in pattern_filter_map:
        pat_len_info.append((pat, len(pattern_filter_map[pat])))

    pat_len_sorted = sorted(pat_len_info, key=lambda x: x[1], reverse=True)
    for i in range(len(pat_len_sorted) - 1):
        pat_len_cur = pat_len_sorted[i]
        pat_len_after = pat_len_sorted[i + 1]

        if pat_len_cur[1] == pat_len_after[1]:
            cur_distance = sum([map_info[1][1] for map_info in list(pattern_filter_map[pat_len_cur[0]].items())])
            after_distance = sum([map_info[1][1] for map_info in list(pattern_filter_map[pat_len_after[0]].items())])
            if cur_distance > after_distance:
                pat_len_sorted[i], pat_len_sorted[i + 1] = pat_len_sorted[i + 1], pat_len_sorted[i]

    for pat_len in pat_len_sorted:
        text_nodes_map = pattern_filter_map[pat_len[0]]
        for text_node in text_nodes_map:
            ui_node = text_nodes_map[text_node][0]
            distance = text_nodes_map[text_node][1]

            if text_node not in text_ui_dict:
                text_ui_dict[text_node] = (ui_node, pat_len[0], distance)

    # # debug
    # for text_node in text_ui_dict:
    #     print(text_node.attrib['text'], text_ui_dict[text_node][0].attrib['text'])

    return text_ui_dict


def divide_view_groups(hierarchy_root_element: ET.Element):
    # store angle between views
    vector_pair_dict = dict()
    # store the view's type
    view_type_dict = dict()
    # store the view groups
    bound_view_groups = set()
    handle_inner_clickable_nodes_for_pattern(hierarchy_root_element, vector_pair_dict, view_type_dict, bound_view_groups)

    calcul_angel_distance(hierarchy_root_element, vector_pair_dict, view_type_dict, bound_view_groups)

    view_groups = map_text_ui(vector_pair_dict)
    text_viewgroup_dict = dict()
    for text_node in view_groups:
        ui_node = view_groups[text_node][0]
        distance = view_groups[text_node][2]
        angle = view_groups[text_node][1]
        text_viewgroup_dict[text_node] = ViewGroup(interact_type=view_type_dict[ui_node], label=text_node, view=ui_node,
                                                   angle=angle, distance=distance)

    return view_type_dict, text_viewgroup_dict


class ViewGroup:
    """
    Representation for mapping of view and label
    """
    def __init__(self, interact_type: str, label: ET.Element, view: ET.Element, angle: int, distance: int):
        # interact type is view type
        self.interact_type = interact_type
        # label is text description for viewgroup
        self.label = label
        # view is the actual node for interaction
        self.view = view
        # angle is the relative coordination between view and label
        self.angle = angle
        # distance of view and label
        self.distance = distance


class Edge:
    def __init__(self, interact_type: str, source, target, perform_text: str, perform_bounds: str):
        self.source = source
        self.target = target

        self.interact_type = interact_type
        self.perform_text = perform_text
        self.perform_bounds = perform_bounds

class Node:
    """
    Node is representation for a page.
    """
    def __init__(self, hierarchy_root_element: ET.Element, is_dummy=False):
        """
        init.
        """
        # is_dummy is used to identity initial node
        self.is_dummy = is_dummy
        self.hierarchy_root_element = hierarchy_root_element
        self.bounds_text_dict = get_bounds_text_dict(self.hierarchy_root_element)
        self.text_set = set(self.bounds_text_dict.values())
        self.bounds_set = set(self.bounds_text_dict.keys())
        # view_type_dict is a map which stores view type，key is view，value is type. if type is ui_unknown, it would be collected
        # view_groups is a dict, the key is textview为key, the value is [ui_node, angle, distance]
        self.view_type_dict, self.view_groups = divide_view_groups(self.hierarchy_root_element)
        self.view_to_collect = self.generate_view_to_collect()
        self.stack_count = 0

        self.ui_path = ''

        # construct edges
        self.in_edges = []
        self.out_edges = []

        # swipe status
        self.swipe_count = 0

        # privacy data
        # {text_label_view: [[user_personal_data, nouns, texts], 'analysis'/'extend']}
        self.view_user_personal_data_dict = dict()

    def generate_view_to_collect(self):
        view_to_collect = set()
        for view in self.view_type_dict:
            if self.view_type_dict[view] == ui_type_UNKNOWN:
                view_to_collect.add(view)
        return view_to_collect

    def has_in_edge(self, edge: Edge):
        for in_edge in self.in_edges:
            if edge.source == in_edge.source and edge.target == in_edge.target:
                return True
        return False

    def has_out_edge(self, edge: Edge):
        for in_edge in self.in_edges:
            if edge.source == in_edge.source and edge.target == in_edge.target:
                return True
        return False


class ClickTrace:
    def __init__(self, begin_node: Node, perform_view: ET.Element):
        # the beginning node when the click event occurs
        self.begin_node = begin_node
        # the view to click
        self.perform_view = perform_view
        # the bounds to click
        self.perform_bounds = perform_view.attrib['bounds'] if perform_view is not None else ''
        # the text to click
        self.perform_text = perform_view.attrib['text'] if perform_view is not None else ''


class AppExplorer:
    # affinity threshold
    affinity_threshold = 0.79

    def __init__(self, pkg_name: str, app_name: str, device):
        self.pkg_name = pkg_name
        self.app_name = app_name
        self.graph = []
        self.device = device
        self.new_page_threshold = 0.5
        self.cur_page_node = None
        # record the page need to click
        self.click_stack = []
        # record the clicked pages
        self.stack_visited = set()
        # record the analyzed page
        self.analysis_visited = set()
        # record the text and privacy data
        self.privacy_data = dict()
        # record initial page
        self.dummy_page = None
        # record the initial texts
        self.inner_text_set = set()
        # prevent click a page for too many times
        self.same_page_click = 0
        # store the ui path classify result
        self.ui_path_cache = dict()
        self.init_state()

    def init_state(self):
        """
        Use initial page as dummy page.
        :return:
        """
        if self.device is None:
            return
        self.start_app()
        dummy_page_node = self.generate_new_node_from_view_tree_xml(get_xml_from_view_tree(device=self.device))
        dummy_page_node.is_dummy = True
        self.cur_page_node = dummy_page_node
        self.dummy_page = dummy_page_node
        self.add_node(pre_node=None, node=dummy_page_node, edge_type=None, perform_text=None, perform_bounds=None)

    def start_app(self):
        adb_stop = f'adb shell am force-stop {self.pkg_name}'
        adb_start_1 = 'adb shell dumpsys window | findstr mCurrentFocus'
        adb_start_2 = f'adb shell monkey -p {self.pkg_name} 1'

        os.system(adb_stop)
        time.sleep(0.5)
        os.system(adb_start_1)
        os.system(adb_start_2)
        time.sleep(4)


    def generate_new_node_from_view_tree_xml(self, root_element: ET.Element):
        cur_page_node = Node(hierarchy_root_element=root_element, is_dummy=False)
        return cur_page_node

    def add_node(self, pre_node: Node, node: Node, edge_type: str, perform_text: str, perform_bounds: str):
        is_new, similar_node = self.is_new_node(node)
        if is_new:
            # add new node in graph
            self.graph.append(node)
            # construct the new edge
            if not node.is_dummy:
                edge = Edge(interact_type=edge_type, source=pre_node, target=node, perform_text=perform_text, perform_bounds=perform_bounds)
                pre_node.out_edges.append(edge)
                node.in_edges.append(edge)
        elif not is_new and similar_node is not None:
            if not node.is_dummy:
                edge = Edge(interact_type=edge_type, source=pre_node, target=node, perform_text=perform_text, perform_bounds=perform_bounds)
                if not pre_node.has_out_edge(edge):
                    pre_node.out_edges.append(edge)
                if not similar_node.has_in_edge(edge):
                    similar_node.in_edges.append(edge)


    def is_new_node(self, node: Node):
        """
        Determine a node whether is a new node, using text set.
        """
        is_new = True
        similar_node = None
        node_text_set = node.text_set

        # compare with dummy node first
        dummy_text_set = self.dummy_page.text_set
        if len(dummy_text_set | node_text_set) == 0:
            return True, None
        if len(dummy_text_set & node_text_set) / len(dummy_text_set | node_text_set) > self.new_page_threshold:
            is_new = False
            similar_node = self.dummy_page
            return is_new, similar_node

        for node_i in self.graph:
            node_i_text_set = node_i.text_set
            intersection = node_i_text_set & node_text_set
            union = node_i_text_set | node_text_set
            if len(union) == 0:
                continue
            diff = node_text_set - node_i_text_set
            if 'View profile' in diff and 'Settings' in diff or 'view profile' in diff and 'settings' in diff:
                diff_view_to_collect = set()
                for view in node.view_to_collect:
                    if view.attrib['text'] in diff:
                        diff_view_to_collect.add(view)
                node.view_to_collect = diff_view_to_collect
                return True, None
            if len(intersection) / len(union) > self.new_page_threshold:
                is_new = False
                similar_node = node_i
                break
        return is_new, similar_node

    def is_same_node(self, node_i: Node, node_j: Node):
        """
        Same as `is_new_node`, determine whether two node is same node.
        """
        is_same = False
        if node_j is None or node_i is None:
            return False
        node_i_text_set = node_i.text_set
        node_j_text_set = node_j.text_set
        intersection = node_i_text_set & node_j_text_set
        union = node_i_text_set | node_j_text_set
        diff = union - intersection
        if 'confirm' in diff and 'cancel' in diff or 'View profile' in diff and 'Settings' in diff or 'view profile' in diff and 'settings' in diff:
            return False
        if len(union) == 0:
            return True
        if len(intersection) / len(union) > self.new_page_threshold:
            is_same = True
        return is_same

    def find_path_from_x_to_y(self, x: Node, y: Node) -> List[Edge]:
        """find the shortest path"""
        stack_paths = []
        if self.is_same_node(x, y):
            return stack_paths
        for out_edge in x.out_edges:
            dfs_stack = []
            dfs_stack.append(out_edge)
            visited = set()
            found = self.dfs_find_path(source_node=out_edge.target, stack=dfs_stack, target_node=y, visited=visited)
            if found:
                stack_paths.append(dfs_stack)

        if len(stack_paths) == 0:
            return None
        else:
            stack_paths = sorted(stack_paths, key=len)
            return stack_paths[0]

    def dfs_find_path(self, source_node: Node, stack: list, target_node: Node, visited: set):
        if source_node in visited:
            return False
        visited.add(source_node)
        if self.is_same_node(source_node, target_node):
            return True
        for out_edge in source_node.out_edges:
            next_node = out_edge.target
            stack.append(out_edge)
            found = self.dfs_find_path(next_node, stack, target_node, visited)
            if found:
                return found
            else:
                stack.pop()
        return False

    def handle_input_box(self):
        dumped_hierarchy = self.device.dump_hierarchy()
        if 'com.baidu.input_mi' in dumped_hierarchy or 'com.google.android.inputmethod.latin' in dumped_hierarchy:
            self.device.press('back')
            time.sleep(0.5)

    def update_node_ui_path(self, node: Node, pre_node: Node):
        if pre_node.is_dummy:
            node.ui_path = node.in_edges[-1].perform_text
        else:
            node.ui_path = pre_node.ui_path + '-' + node.in_edges[-1].perform_text


    def is_choice(self, node: Node):
        """
        determine whether a node is a user choice resort to tiny coordination.
        :param node:
        """
        if len(node.view_type_dict) == 0:
            return False

        button_text_set = set()
        for view in node.view_type_dict:
            if 'text' in view.attrib:
                button_text_set.add(view.attrib['text'].replace(' ', '').lower())
        if ('确认' in button_text_set or '确定' in button_text_set) and '取消' in button_text_set or (
                'cancel' in button_text_set and 'confirm' in button_text_set):
            return True
        # find all clickable texts, calculation their angles.
        onclick_views = []
        tidy_nodes = set()
        for view in node.view_type_dict:
            if node.view_type_dict[view] == ui_type_UNKNOWN:
                onclick_views.append(view)

        if len(onclick_views) < 3:
            return False

        # for view in onclick_views:
        #     print(view.attrib['text'] + ' ' + node.view_type_dict[view] + ' ' + view.attrib['bounds'])

        onclick_vector_pair_dict = dict()
        for i in range(len(onclick_views) - 1):
            for j in range(i + 1, len(onclick_views)):
                node_i, node_j = onclick_views[i], onclick_views[j]
                angel = calculate_angle(node_i.attrib['bounds'], node_j.attrib['bounds'])
                if angel not in onclick_vector_pair_dict:
                    onclick_vector_pair_dict[angel] = []
                onclick_vector_pair_dict[angel].append((node_i, node_j))
                if angel == 0 or angel == 90:
                    tidy_nodes.add(node_i)
                    tidy_nodes.add(node_j)

        if len(tidy_nodes) == len(onclick_views) or len(tidy_nodes) == len(onclick_views) - 1 or len(tidy_nodes) == len(onclick_views) - 2:
            return True
        return False


    def perform_swipe(self, pre_node: Node):
        print('perform swipe...')
        #
        # 1. swipe
        #
        self.device.swipe(10, 1700, 10, 600, duration=0.2)
        time.sleep(0.5)
        self.device.swipe(10, 1700, 10, 600, duration=0.2)
        time.sleep(0.5)
        node_after_swipe = self.generate_new_node_from_view_tree_xml(get_xml_from_view_tree(self.device))

        #
        # 2. judge new
        #
        occur_new = False
        #
        # 3. remain graph
        #
        if not self.is_same_node(node_after_swipe, pre_node):
            occur_new = True
            node_after_swipe.swipe_count = self.cur_page_node.swipe_count + 1
            #
            # 3.1 add new node
            #
            self.add_node(pre_node=pre_node, node=node_after_swipe, edge_type=edge_type_SWIPE,
                          perform_text='', perform_bounds='')
            # update ui-path
            node_after_swipe.ui_path = pre_node.ui_path
            # record log
            logger.logger.log_cur_page('SWIPE', node_after_swipe.text_set)

            #
            # 3.2 update cur node
            #
            self.cur_page_node = node_after_swipe
        else:
            self.perform_back()

        # query node
        is_new, similar_node = self.is_new_node(node_after_swipe)
        if not is_new:
            node_after_swipe = similar_node

        return occur_new, node_after_swipe


    def handle_swipe(self, pre_node: Node):
        """
        Swipe a page, creating a new node.
        1. swipe；
        2. new；
        3. graph；
        4. dfs
        :return:
        """

        # only swipe twice
        if pre_node.swipe_count >= 2:
            return

        self.guide_to_target(target_node=pre_node)

        #
        # 1. swipe
        # 2. new
        # 3. graph
        #
        occur_new, node_after_swipe = self.perform_swipe(pre_node=pre_node)


        #
        # 4. dfs
        #
        if occur_new:
            for view in node_after_swipe.view_to_collect:
                new_click_trace = ClickTrace(begin_node=node_after_swipe, perform_view=view)
                # classify ui-path
                if view.attrib['text'].strip() == '':
                    continue

                if count_ui_path(pre_node.ui_path):
                    self.click_stack.append(new_click_trace)


    def try_analysis_privacy(self, node:Node, mode='normal'):
        can_analysis = True
        for view in node.view_to_collect:
            if node.view_type_dict[view] == ui_type_UNKNOWN:
                can_analysis = False
                break
        if not can_analysis and mode == 'normal':
            return

        for text_label in node.view_groups:
            view_group = node.view_groups[text_label]
            view = view_group.view

            if node.view_type_dict[view] in ui_types_INTEREACT:
                nlp_analysis = get_nlp_analysis(text_label, HanLP=HanLP, nlp_cache=nlp_cache, privacy_keywords=privacy_keywords)
                node.view_user_personal_data_dict[text_label] = [nlp_analysis, 'analysis']
        # extend
        AppExplorer.affinity_extend(node)

        log_privacy_data_set = set()
        log_privacy_label_set = set()
        print('analysis privacy information: ')
        for view_info in node.view_user_personal_data_dict:
            print('text info: ', view_info.attrib['text'])
            print(node.view_user_personal_data_dict[view_info])
            log_privacy_label_set.add(view_info.attrib['text'])
            for data in node.view_user_personal_data_dict[view_info][0][0]:
                log_privacy_data_set.add(data)
        # log the privacy
        logger.logger.log_privacy_data(info='privacy', data_extend=log_privacy_data_set, data_label=log_privacy_label_set)

        # swipe
        if mode == 'normal' and can_analysis:
            self.handle_swipe(pre_node=node)

    @classmethod
    def affinity_extend(cls, node: Node):
        """
        Extend the privacy based on affinity。
        vector [width, height, margin_x, margin_y, alignment]
        :return:
        """

        data_extend = set()
        label_extend = set()

        pat_view_dict = dict()

        # store alignment
        for t_view in node.view_groups:
            pattern = node.view_groups[t_view].angle
            if pattern not in pat_view_dict:
                pat_view_dict[pattern] = []
            pat_view_dict[pattern].append(t_view)

        # generate embeddings
        view_embeddings_dict = dict()
        for t_view in node.view_groups:
            u_view = node.view_groups[t_view].view
            width = (get_width(t_view) + get_width(u_view)) / 2
            height = (get_height(t_view) + get_height(u_view)) / 2
            marginx = min(parse_corr(t_view.attrib['bounds'])[0], parse_corr(u_view.attrib['bounds'])[0])
            marginy = min(parse_corr(t_view.attrib['bounds'])[1], parse_corr(u_view.attrib['bounds'])[1])
            view_embeddings_dict[t_view] = [width, height, marginx, marginy, node.view_groups[t_view].distance, node.view_groups[t_view].angle]
        # min-max normalization
        max_d = [-1, -1, -1, -1, -1, -1]
        min_d = [99999, 99999, 99999, 99999, 99999, 99999]
        for embedding in view_embeddings_dict.values():
            for d in range(6):
                min_d[d] = min(min_d[d], embedding[d])
                max_d[d] = max(max_d[d], embedding[d])
        for t_view in view_embeddings_dict:
            embedding = view_embeddings_dict[t_view]
            for d in range(6):
                max_min = 1
                if min_d[d] - min_d[d] != 0:
                    max_min = max_d[d] - min_d[d]
                embedding[d] = (embedding[d] - min_d[d]) / max_min
            view_embeddings_dict[t_view] = embedding

        # viewgroup
        for pat in pat_view_dict:
            view_distance_dict = dict()
            views = pat_view_dict[pat]

            for i in range(len(views) - 1):
                for j in range(i + 1, len(views)):
                    view_i, view_j = views[i], views[j]
                    relative_distance = calculate_relative_location(view_i.attrib['bounds'], view_j.attrib['bounds'])
                    if view_i not in view_distance_dict:
                        view_distance_dict[view_i] = []
                    if view_j not in view_distance_dict:
                        view_distance_dict[view_j] = []
                    view_distance_dict[view_i].append((view_j, relative_distance))
                    view_distance_dict[view_j].append((view_i, relative_distance))

            for view in view_distance_dict:
                view_distance_dict[view] = sorted(view_distance_dict[view], key=lambda x: x[1])

            privacy_views = set()
            for view in view_distance_dict:
                # view_user_personal_data_dict = dict()
                # {text_label_view: ([[user_personal_data], [nouns], [texts]], 'analysis'/'extend')}

                if view not in node.view_user_personal_data_dict or view not in node.view_type_dict:
                    continue
                if len(node.view_user_personal_data_dict[view][0][0]) != 0:
                    privacy_views.add(view)

            label_view_type = dict()
            for label in node.view_groups:
                view = node.view_groups[label].view
                label_view_type[label] = node.view_type_dict[view]
            while True:
                pre_len = len(privacy_views)
                for view in view_distance_dict:
                    if view in privacy_views:
                        view_vector = view_embeddings_dict[view]

                        simi_t = -1
                        for view_ex in view_distance_dict[view]:
                            view_ex_vector = view_embeddings_dict[view_ex[0]]
                            simi = calcul_similarity(view_vector, view_ex_vector)
                            if simi < AppExplorer.affinity_threshold:
                                continue

                            if simi >= simi_t and label_view_type[view_ex[0]] in ui_types_INTEREACT:
                                simi_t = simi
                                privacy_views.add(view_ex[0])
                                if view_ex[0] not in node.view_user_personal_data_dict:
                                    print('err! not interact node want to extend')
                                    sys.stderr.write('err! not interact node want to extend\n')
                                    continue

                                # extend
                                node.view_user_personal_data_dict[view_ex[0]][0][0] = node.view_user_personal_data_dict[view_ex[0]][0][1]
                                node.view_user_personal_data_dict[view_ex[0]][1] = 'extend'

                                if len(node.view_user_personal_data_dict[view_ex[0]][0][1]) > 0:
                                    label_extend.add(node.view_user_personal_data_dict[view_ex[0]][0][2])
                                for ex_data in node.view_user_personal_data_dict[view_ex[0]][0][1]:
                                    data_extend.add(ex_data)
                            elif simi-simi_t < 0.02 and label_view_type[view_ex[0]] in ui_types_INTEREACT:
                                if view_ex[0] in privacy_views:
                                    continue
                                privacy_views.add(view_ex[0])

                                # extend
                                view_ex_vector[2] = 0

                                node.view_user_personal_data_dict[view_ex[0]][0][0] = node.view_user_personal_data_dict[view_ex[0]][0][1]
                                node.view_user_personal_data_dict[view_ex[0]][1] = 'extend'

                                if len(node.view_user_personal_data_dict[view_ex[0]][0][1]) > 0:
                                    label_extend.add(node.view_user_personal_data_dict[view_ex[0]][0][2])
                                for ex_data in node.view_user_personal_data_dict[view_ex[0]][0][1]:
                                    data_extend.add(ex_data)
                after_len = len(privacy_views)
                if pre_len == after_len:
                    break

            logger.logger.log_privacy_data(info='extend', data_extend=data_extend, data_label=label_extend)

    def collect_input(self, pre_node: Node, cur_node: Node, perform_bounds: str, perform_text: str, need_back=False):
        # find perform view
        perform_view = None
        perform_view_same_logic = []
        for view in pre_node.view_to_collect:
            if view.attrib['bounds'] == perform_bounds:
                perform_view = view
                break
        if perform_view is None:
            for view in pre_node.view_to_collect:
                if view.attrib['text'] == perform_text:
                    perform_view = view
        if perform_view is None:
            print('debug for collect_input: can not find perform view.')
            self.try_analysis_privacy(pre_node, mode='force')
            return
        for view in pre_node.view_to_collect:
            if view.attrib['text'] == perform_view.attrib['text']:
                perform_view_same_logic.append(view)

        # If there is an interaction type in perform_view in the same logic, then directly let all the same logic use this interaction type
        # If there is no interaction type, then choose an unknown view as the next analysis object
        if len(perform_view_same_logic) != 0:
            all_view_interact = False
            all_view_type = ''
            for same_perform_view in perform_view_same_logic:
                if pre_node.view_type_dict[same_perform_view] in ui_types_INTEREACT:
                    all_view_interact = True
                    all_view_type = pre_node.view_type_dict[same_perform_view]
                    break
            if all_view_interact:
                for same_perform_view in perform_view_same_logic:
                    if all_view_interact and all_view_type != '':
                        pre_node.view_type_dict[same_perform_view] = all_view_type
                print('Collect by logic: ' + all_view_type)
        if pre_node.view_type_dict[perform_view] != ui_type_UNKNOWN:
            return

        nodes_preds_of_pre_node = self.get_preds_nodes(pre_node)
        for node in nodes_preds_of_pre_node:
            if self.is_same_node(cur_node, node):
                pre_node.view_type_dict[perform_view] = ui_type_VIEW
        if pre_node.view_type_dict[perform_view] == ui_type_UNKNOWN:
            # indirect_input
            for view in cur_node.view_type_dict:
                if cur_node.view_type_dict[view] == ui_type_INPUT:
                    pre_node.view_type_dict[perform_view] = ui_type_INDIRECT_INPUT
                    break
            # choice
            if pre_node.view_type_dict[perform_view] == ui_type_UNKNOWN:
                if self.is_choice(cur_node):
                    pre_node.view_type_dict[perform_view] = ui_type_DIALOG
            # normal view
            if pre_node.view_type_dict[perform_view] == ui_type_UNKNOWN:
                pre_node.view_type_dict[perform_view] = ui_type_VIEW
        print('debug for collect_input: the collected text-- ' + perform_view.attrib['text'] + ' --is: ', pre_node.view_type_dict[perform_view])

        if need_back:
            self.perform_back()
        self.try_analysis_privacy(pre_node)


    def switch_page(self, switch_bounds: str, switch_text: str):
        #
        # 1. find coordination to click
        #
        if 'log out' in switch_text.lower() or 'sign out' in switch_text.lower() or 'delete account' in switch_text.lower() or 'anonymous browsing' in switch_text.lower():
            return

        pre_page_node = self.cur_page_node

        bounds = ''
        tmp_page_node = self.generate_new_node_from_view_tree_xml(get_xml_from_view_tree(self.device))

        find_bounds = True
        if switch_bounds not in tmp_page_node.bounds_text_dict:
            find_bounds = False
            if switch_text.strip() == '' or switch_text not in pre_page_node.text_set:
                traceback.print_stack()
                print(switch_text, ' can not be found, raise an exception.')
                raise NoTextException("NoTextException")

            print(switch_text, ' Text to click not in current screen, try use imprecise way.')

            for tmp_bounds in tmp_page_node.bounds_text_dict:
                if tmp_page_node.bounds_text_dict[tmp_bounds] == switch_text:
                    switch_bounds = tmp_bounds
                    find_bounds = True
                    break

        if find_bounds:
            bounds = switch_bounds
        if bounds == '':
            print(switch_text, ' Text to click not in current screen, jump this switch.')
            self.guide_to_target(self.dummy_page)
            # raise NoNeedExploreException('no need explore')
            return

        corr = parse_corr(bounds)

        print(f'click: {bounds}, text: {switch_text}')

        #
        # 2. perform click
        #

        self.device.click((corr[0] + corr[2]) / 2, (corr[1] + corr[3]) / 2)
        logger.logger.log_click(switch_text)
        time.sleep(2)

        # handle input box
        self.handle_input_box()
        if self.is_out_of_app():
            self.perform_back()

        #
        # 3. add new node in graph
        #

        cur_hierarchy = get_xml_from_view_tree(self.device)
        cur_page_node = self.generate_new_node_from_view_tree_xml(cur_hierarchy)
        cur_page_node_to_collect = self.generate_new_node_from_view_tree_xml(cur_hierarchy)

        logger.logger.log_cur_page(switch_text, cur_page_node.text_set)

        if self.is_out_of_app():
            self.perform_back()
            self.collect_input(pre_node=pre_page_node, cur_node=cur_page_node, perform_text=switch_text,
                               perform_bounds=switch_bounds)
            return

        is_new, similar_node = self.is_new_node(cur_page_node)

        has_new_input_elements_in_same_page = False

        new_occur_views = set()
        new_occur_texts = set()
        if self.is_same_node(pre_page_node, cur_page_node):

            for view in cur_page_node_to_collect.view_type_dict:
                if view.attrib['text'] not in pre_page_node.text_set and view.attrib['text'].strip() != '':
                    new_occur_texts.add(view.attrib['text'])
                is_new_occur_views = True
                for pre_view in pre_page_node.view_type_dict:
                    if pre_view.attrib['bounds'] == view.attrib['bounds']:
                        is_new_occur_views = False
                        break
                if is_new_occur_views:
                    new_occur_views.add(view)

            view_to_delete = set()
            for view in cur_page_node_to_collect.view_type_dict:
                if view not in new_occur_views:
                    view_to_delete.add(view)

            # input
            for view in new_occur_views:
                if is_user_input(view):
                    has_new_input_elements_in_same_page = True
            # choice
            for view in view_to_delete:
                del cur_page_node_to_collect.view_type_dict[view]
            if len(new_occur_views) != 0:
                if self.is_choice(cur_page_node_to_collect):
                    has_new_input_elements_in_same_page = True

            if has_new_input_elements_in_same_page:
                cur_page_node_to_collect.text_set = new_occur_texts
                self.same_page_click = 0

            self.same_page_click += 1
        else:
            self.same_page_click = 0

        if is_new:
            self.add_node(pre_node=pre_page_node, node=cur_page_node, edge_type=edge_type_CLICK, perform_text=switch_text, perform_bounds=switch_bounds)

            # update ui-path
            self.update_node_ui_path(node=cur_page_node, pre_node=pre_page_node)
        else:
            ui_path = similar_node.ui_path
            self.stack_visited.add(ui_path)

        if not is_new:
            cur_page_node = similar_node
        self.cur_page_node = cur_page_node

        #
        # 4. update graph
        #

        need_back = False
        if len(new_occur_views) >= 2:
            need_back = True
        self.collect_input(pre_node=pre_page_node, cur_node=cur_page_node_to_collect, perform_text=switch_text, perform_bounds=switch_bounds, need_back=need_back)




    def get_succs_nodes(self, node: Node):
        visited = set()
        succs_nodes = []
        bfs_queue = []
        bfs_queue.append(node)
        while len(bfs_queue) != 0:
            head_node = bfs_queue.pop()
            if head_node in visited:
                continue
            visited.add(head_node)

            succs_nodes.append(head_node)

            for out_edge in head_node.out_edges:
                next_page = out_edge.target
                bfs_queue.append(next_page)

        return succs_nodes

    def get_preds_nodes(self, node: Node):
        visited = set()
        preds_nodes = []
        bfs_queue = []
        bfs_queue.append(node)
        while len(bfs_queue) != 0:
            head_node = bfs_queue.pop()
            if head_node in visited:
                continue
            visited.add(head_node)

            preds_nodes.append(head_node)

            for in_edge in head_node.in_edges:
                pre_node = in_edge.source
                bfs_queue.append(pre_node)

        return preds_nodes

    def perform_back(self):
        pre_page_node = self.cur_page_node
        self.device.press('back')
        time.sleep(2)
        cur_page_node = self.generate_new_node_from_view_tree_xml(get_xml_from_view_tree(self.device))

        logger.logger.log_back(set(cur_page_node.bounds_text_dict.values()))

        is_new, similar_node = self.is_new_node(cur_page_node)
        if not is_new and similar_node is not None:
            self.cur_page_node = similar_node
        else:
            self.cur_page_node = cur_page_node

    def is_out_of_app(self):
        back_root = get_xml_from_view_tree(self.device)
        pkg_count = 0
        back_count = 0
        back_app = False
        for element in back_root.iter():
            if 'package' in element.attrib:
                if element.attrib['package'] == self.pkg_name:
                    pkg_count += 1
                else:
                    back_count += 1
        if pkg_count < back_count:
            back_app = True

        return back_app

    def restart_app(self):
        adb_stop = f'adb shell am force-stop {self.pkg_name}'

        adb_start_1 = 'adb shell dumpsys window | findstr mCurrentFocus'
        adb_start_2 = f'adb shell monkey -p {self.pkg_name} 1'

        os.system(adb_stop)
        time.sleep(0.5)
        os.system(adb_start_1)
        os.system(adb_start_2)
        time.sleep(4)

        restarted_node = self.generate_new_node_from_view_tree_xml(get_xml_from_view_tree(self.device))
        if self.is_same_node(restarted_node, self.dummy_page):
            self.cur_page_node = self.dummy_page
        else:
            restarted_node.is_dummy = self.dummy_page.is_dummy
            restarted_node.in_edges = self.dummy_page.in_edges
            restarted_node.out_edges = self.dummy_page.out_edges
            self.cur_page_node = restarted_node
            self.dummy_page = restarted_node


    def guide_to_target(self, target_node: Node):
        if self.is_same_node(target_node, self.cur_page_node):
            return

        if target_node.is_dummy:
            print('restart app and guide to dummy node. ')
            self.restart_app()
            return

        # First perform rollback. The rollback node must be target_node or a node that can reach target_node.
        # Determine whether the current node is the successor node of the target node.
        # If yes, keep going back until the current node is the target node or the predecessor node of the target node.
        # If not, keep going back until the current node is the predecessor node of the target node;
        # Extract all nodes reachable by target
        target_preds_nodes = self.get_preds_nodes(target_node)

        is_cur_page_node_pres_of_target = False
        for access_node in target_preds_nodes:
            if self.is_same_node(access_node, self.cur_page_node):
                is_cur_page_node_pres_of_target = True
                break

        # record back count for restart
        back_count = 0
        while not is_cur_page_node_pres_of_target:

            self.perform_back()
            back_count += 1

            if self.is_out_of_app():
                self.restart_app()
                break

            if back_count >= 3:
                self.restart_app()
                break

            for access_node in target_preds_nodes:
                if self.is_same_node(access_node, self.cur_page_node):
                    is_cur_page_node_pres_of_target = True
                    break

        is_access_from_cur_page_node_to_target = False
        succs_nodes = self.get_succs_nodes(self.cur_page_node)
        for access_node in succs_nodes:
            if access_node.is_dummy and target_node.is_dummy:
                is_access_from_cur_page_node_to_target = True
                break
            if self.is_same_node(access_node, target_node):
                is_access_from_cur_page_node_to_target = True
                break
        if not is_access_from_cur_page_node_to_target:
            raise NoNeedExploreException('the path can not be found: \n from: ' + str(target_node.text_set) + '\n to:' + str(self.cur_page_node.text_set))
        else:
            interaction_edges = self.find_path_from_x_to_y(self.cur_page_node, target_node)
            if interaction_edges is None:
                time.sleep(3)
                print('Error when finding path.')
                raise NoNeedExploreException('the path can not be found: \n from: ' + str(target_node.text_set) + '\n to:' + str(self.cur_page_node.text_set))
            else:
                for interaction_edge in interaction_edges:
                    interaction_type = interaction_edge.interact_type
                    if interaction_type == edge_type_CLICK:
                        perform_text = interaction_edge.perform_text
                        perform_bounds = interaction_edge.perform_bounds

                        self.switch_page(switch_bounds=perform_bounds, switch_text=perform_text)

                    elif interaction_type == edge_type_SWIPE:
                        self.perform_swipe(pre_node=self.cur_page_node)

    def run_bfs(self):
        """
        only run when init
        :return:
        """
        init_count = 5
        init_text = dict()
        inner_text = set()
        for text in self.cur_page_node.text_set:
            init_text[text] = 0

        bfs_view_queue = self.cur_page_node.view_type_dict
        print('the init text set: ')
        for view in bfs_view_queue:
            print(view.attrib['text'])
        for view in bfs_view_queue:
            new_click_trace = ClickTrace(begin_node=self.cur_page_node, perform_view=view)
            self.click_stack.append(new_click_trace)
        bfs_click_trace = copy.deepcopy(self.click_stack)
        for click_trace in bfs_click_trace:
            try:
                if len(inner_text) == 0:
                    self.guide_to_target(self.dummy_page)
                else:
                    self.inner_text_set = inner_text

                    if click_trace.perform_text not in inner_text:
                        continue
                    if click_trace.perform_text not in self.cur_page_node.text_set:
                        self.guide_to_target(self.dummy_page)

            except Exception as e:
                print(e)
            print(click_trace.perform_text)
            try:
                if init_count <= 0 and click_trace.perform_text not in inner_text:
                    continue
                self.switch_page(switch_text=click_trace.perform_text, switch_bounds=click_trace.perform_bounds)
                for text in self.cur_page_node.text_set:
                    if text not in init_text:
                        continue
                    else:
                        init_text[text] += 1
            except Exception as e:
                print(e)

            # build inner_text
            init_count -= 1
            inner_text_count = 0
            if init_count == 0:
                if len(inner_text) == 0:
                    sorted_text = dict(sorted(init_text.items(), key=itemgetter(1), reverse=True))
                    for text in sorted_text:
                        if init_text[text] >= 2:
                            inner_text.add(text)
                            inner_text_count += 1
                            if inner_text_count >= 5:
                                break
                    for text in sorted_text:
                        if 'account' in text.lower() or 'profile' in text.lower() or 'me' == text.lower():
                            inner_text.add(text)

        for node in self.graph:
            ui_path_nodes = node.ui_path.split('-')
            node.ui_path = ui_path_nodes[-1]
        self.click_stack.clear()
        return inner_text


    def run_dfs(self):
        while len(self.click_stack) != 0:

            # end the program if the time exceed than 20 minutes
            if time.time() - logger.logger.start_time > TOTAL_TIME:
                break

            next_trace = self.click_stack.pop()
            try:
                self.stack_visited.add(next_trace.begin_node)

                next_trace.begin_node.stack_count += 1
                if next_trace.begin_node.stack_count >= 30:
                    continue
                if next_trace.begin_node.view_type_dict[next_trace.perform_view] != ui_type_UNKNOWN:
                    continue

                self.guide_to_target(next_trace.begin_node)
                self.switch_page(switch_text=next_trace.perform_text, switch_bounds=next_trace.perform_bounds)

                if self.same_page_click >= 5:
                    print('stop click a same page')
                    self.same_page_click = 0
                    raise NoNeedExploreException('')
                if self.cur_page_node in self.stack_visited or self.cur_page_node.is_dummy:
                    continue

                # # classify ui-path
                # if view_to_collect.attrib['text'].strip() == '':
                #     continue
                ui_path = self.cur_page_node.ui_path
                if classify_ui_path(ui_path, self.ui_path_cache, self.inner_text_set):
                    for view_to_collect in self.cur_page_node.view_to_collect:
                        if self.cur_page_node.view_type_dict[view_to_collect] != ui_type_UNKNOWN:
                            continue
                        new_click_trace = ClickTrace(begin_node=self.cur_page_node, perform_view=view_to_collect)

                        if count_ui_path(ui_path):
                            self.click_stack.append(new_click_trace)
                else:
                    # deal with the normal click track which is not privacy-related.
                    for view_to_collect in self.cur_page_node.view_to_collect:
                        if self.cur_page_node.view_type_dict[view_to_collect] != ui_type_UNKNOWN:
                            continue
                        new_click_trace = ClickTrace(begin_node=self.cur_page_node, perform_view=view_to_collect)

                        if count_ui_path(ui_path):
                            self.click_stack.insert(0, new_click_trace)

            except NoTextException as e:
                print(e)
                for i in range(len(self.click_stack)-1, -1, -1):
                    if self.is_same_node(self.click_stack[i].begin_node, next_trace.begin_node):
                        self.click_stack.pop()
                    else:
                        break
            except NoNeedExploreException as e:
                print(e)
                for i in range(len(self.click_stack) - 1, -1, -1):
                    if self.is_same_node(self.click_stack[i].begin_node, next_trace.begin_node):
                        self.click_stack.pop()
                    else:
                        break
        for node in self.graph:
            if len(node.view_user_personal_data_dict) == 0:
                self.try_analysis_privacy(node, mode='force')

    def run(self):
        inner_text = self.run_bfs()
        init_path_dict = {}
        for node in self.graph:

            if node.ui_path not in init_path_dict:
                init_path_dict[node.ui_path] = []

            if node.is_dummy:
                continue
            for sub_view in node.view_to_collect:
                sub_view_text = sub_view.attrib['text'].strip().lower()
                if sub_view_text == '' or sub_view_text in inner_text:
                    continue
                ui_path = node.ui_path + '-' + sub_view_text

                # Explore the privacy-related pages first,
                # then, the normal pages.
                if classify_ui_path(ui_path, self.ui_path_cache, self.inner_text_set, mode='init'):
                    new_click_trace = ClickTrace(begin_node=node, perform_view=sub_view)
                    init_path_dict[node.ui_path].append(new_click_trace)
                else:
                    # we use stack to store the click trace,
                    # so add the normal click track first so that it can be explored later.
                    new_click_trace = ClickTrace(begin_node=node, perform_view=sub_view)
                    self.click_stack.append(new_click_trace)


        sorted_dict = sorted(init_path_dict.items(), key=lambda x: len(x[1]), reverse=True)
        # add the normal, un-privacy-related ui paths first.
        for i in range(len(sorted_dict)-1, 0, -1):
            for new_click_trace in sorted_dict[i][1]:
                self.click_stack.append(new_click_trace)

        self.run_dfs()

        for node in self.graph:
            for view in node.view_user_personal_data_dict:
                if len(node.view_user_personal_data_dict[view][0]) != 0:
                    logger.logger.log_result(info='result', data=node.view_user_personal_data_dict[view])


'''
debug
'''
def classify_ui_path(ui_path: str, ui_path_cache: dict, inner_text_set: set, mode='normal'):
    ui_path = ui_path.lower()

    # debug following
    # if ui_path.count('-') == 1:
    #     if 'edit profile' not in ui_path:
    #         return False

    for text in inner_text_set:
        if ui_path.split('-')[-1].lower() == text.lower():
            return False
    if ui_path.split('-')[-1].lower() == 'settings':
        return True

    path_len = ui_path.count('-') + 1
    if path_len >= 5:
        return False

    if ui_path in ui_path_cache:
        return ui_path_cache[ui_path]

    model_input = pre_process_text(ui_path.replace('-', '[SEP]'))
    model_output = bert_model_classifier(model_input['input_ids'].unsqueeze(0), attention_mask=model_input['attention_mask'].unsqueeze(0))
    logits = torch.argmax(model_output.logits, dim=1)
    preds = logits.detach()
    if preds.item() == 1:
        print(f'ui_path# {ui_path}  #  is predicted into  #  {preds.item()}  #')
        ui_path_cache[ui_path] = True
        return True
    ui_path_cache[ui_path] = False
    return False


def pre_process_text(text):

    tokens = bert_model_tokenizer(text, add_special_tokens=True, return_tensors='pt', max_length=128,
                                        truncation=True,
                                        padding='max_length')

    input_ids = tokens['input_ids'].view(128)
    attention_mask = tokens['attention_mask'].view(128)
    return {'text': text,
            'input_ids': input_ids,
            'attention_mask': attention_mask
            }



def main(pkg_name):
    import sys

    t_pkg_name = pkg_name
    if len(sys.argv) > 1:
        t_pkg_name = sys.argv[1]


    import uiautomator2 as u2

    device = u2.connect()
    logger.init_logger(t_pkg_name)
    pkg_name = t_pkg_name

    app_explorer = AppExplorer(pkg_name=pkg_name, app_name='', device=device)
    try:
        app_explorer.run()
    except Exception as e:
        traceback.print_exc()

    import random_explorer
    try:
        random_explorer.random_run(pkg_name=pkg_name, device=device, remaining_time=TOTAL_TIME-(time.time()-logger.logger.start_time),
                                   app_explorer=app_explorer, logger=logger.logger)
    except Exception as e:
        traceback.print_exc()

    print('END')
    os.system('adb shell am force-stop ' + t_pkg_name)


if __name__ == '__main__':
    main()



