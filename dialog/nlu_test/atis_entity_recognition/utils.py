import os
import logging, sys, argparse


def check_multi_path(path):
    assert isinstance(path, str) and len(path) > 0
    if '\\' in path:
        path.replace('\\', '/')
    childs = path.split('/')
    root = childs[0]
    for index, cur_child in enumerate(childs):
        if index > 0:
            root = os.path.join(root, cur_child)
        if not os.path.exists(root):
            os.mkdir(root)

