import os.path
import subprocess
import time
import re

log_dir = 'logdir'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def get_activity():
    current_activity = ''
    cmd = 'adb shell "dumpsys activity top | grep ACTIVITY | tail -n 1"'
    output = subprocess.check_output(cmd, shell=True).decode('utf-8')
    pat = ' *ACTIVITY (.*) .* pid.*'
    mat = re.match(pat, output)
    if mat:
        current_activity = mat.group(1)

    return current_activity



class Logger:
    def __init__(self, pkg_name):
        self.start_time = time.time()
        self.log_path = os.path.join(log_dir, pkg_name + '.txt')
        self.pkg_name = pkg_name

    def log_time(self):
        return int(time.time() - self.start_time)

    def init_logger(self):
        with open(self.log_path, 'w', encoding='utf-8') as f:
            log_stamp = self.log_time()
            info = 'init for ' + self.pkg_name
            line = f'log_time: {log_stamp}, info: {info}, text: none, entry_text: none, activity: [{get_activity()}]\n'
            f.write(line)

    def log_cur_page(self, entry_text, text_set):
        with open(self.log_path, 'a', encoding='utf-8') as f:
            log_stamp = self.log_time()
            info = 'switch a page'
            text = str(text_set)
            line = f'log_time: {log_stamp}, info: {info}, entry_text: {entry_text}, text: {text}, activity: [{get_activity()}]\n'
            f.write(line)

    def log_back(self, text_set):
        with open(self.log_path, 'a', encoding='utf-8') as f:
            log_stamp = self.log_time()
            info = 'back'
            text = str(text_set)
            line = f'log_time: {log_stamp}, info: {info}, text: {text}, entry_text: none, activity: [{get_activity()}]\n'
            f.write(line)

    def log_click(self, text_click):
        with open(self.log_path, 'a', encoding='utf-8') as f:
            log_stamp = self.log_time()
            info = 'click'
            line = f'log_time: {log_stamp}, info: {info}, text: {text_click}, entry_text: none, activity: [{get_activity()}]\n'
            f.write(line)

    def log_privacy_data(self, info: str, data_extend: set, data_label: set):
        with open(self.log_path, 'a', encoding='utf-8') as f:
            log_stamp = self.log_time()
            line = f'log_time: {log_stamp}, info: {info}, text: {str(data_extend)}, privacy_label: {str(data_label)} entry_text: none, activity: [{get_activity()}]\n'
            f.write(line)

    def log_result(self, info: str, data):
        with open(self.log_path, 'a', encoding='utf-8') as f:
            log_stamp = self.log_time()
            line = f'log_time: {log_stamp}, info: {info}, text: {str(data)}, entry_text: none, activity: [{get_activity()}]\n'
            f.write(line)

logger = None
def init_logger(pkg_name):
    global logger
    logger = Logger(pkg_name)
