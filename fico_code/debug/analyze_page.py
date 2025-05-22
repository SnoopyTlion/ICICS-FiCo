import sys
sys.path.append('../')
from utils import *


if __name__ == '__main__':
    import uiautomator2 as u2
    device = u2.connect()
    pretty_view_tree(get_xml_from_view_tree(device))

