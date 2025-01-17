import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functional import seq
from collections import deque


# 设置起始和结束ID
id_start = 10001
id_end = 114466

# 定义SQLite数据库文件路径 
db_file = 'mj_quezha.db'


def custom_pretty_print(obj, indent=0):
    def is_short_list(lst):
        return all(isinstance(item, (str, int, float)) and len(str(item)) < 20 for item in lst)
    
    def print_aligned_list(lst, indent):
        if all(isinstance(item, dict) for item in lst):
            keys = lst[0].keys()
            max_lens = {key: max(len(str(item[key])) for item in lst) for key in keys}
            header = ' '.join(f'{key:<{max_lens[key]}}' for key in keys)
            print(' ' * indent + header)
            for item in lst[:10]:
                row = ' '.join(f'{str(item[key]):<{max_lens[key]}}' for key in keys)
                print(' ' * indent + row)
            if len(lst) > 10:
                print(' ' * indent + '...')
        else:
            max_len = max(len(str(item)) for item in lst)
            row_len = 80 // (max_len + 2)
            for i in range(0, min(len(lst), 10), row_len):
                row_items = lst[i:i + row_len]
                print(' ' * indent + ' '.join(f'{str(item):<{max_len}}' for item in row_items))
            if len(lst) > 10:
                print(' ' * indent + '...')
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(' ' * indent + str(key) + ':')
            if isinstance(value, dict):
                custom_pretty_print(value, indent + 4)
            elif isinstance(value, list):
                if is_short_list(value):
                    print(' ' * indent + '[', ', '.join(map(str, value[:10])), ', ...]' if len(value) > 10 else ']')
                else:
                    if all(isinstance(item, dict) for item in value):
                        print_aligned_list(value, indent + 4)
                    else:
                        print(' ' * indent + '[')
                        for item in value[:10]:
                            if isinstance(item, dict):
                                custom_pretty_print(item, indent + 4)
                            else:
                                print(' ' * (indent + 4) + str(item) + ',')
                        if len(value) > 10:
                            print(' ' * (indent + 4) + '...')
                        print(' ' * indent + ']')
            else:
                print(' ' * (indent + 4) + str(value))
    elif isinstance(obj, list):
        if is_short_list(obj):
            print(' ' * indent + '[', ', '.join(map(str, obj[:10])), ', ...]' if len(obj) > 10 else ']')
        else:
            print(' ' * indent + '[')
            print_aligned_list(obj, indent + 4)
            print(' ' * indent + ']')
def custom_pretty_print(obj, indent=0, max_inline_length=20, max_line_length=10):
    def is_short_list(lst):
        return all(isinstance(item, (str, int, float)) and len(str(item)) < 20 for item in lst)

    def print_aligned_list(lst, indent):
        if all(isinstance(item, dict) for item in lst):
            keys = lst[0].keys()
            max_lens = {key: max(len(str(item[key])) for item in lst) for key in keys}
            header = ' '.join(f'{key:<{max_lens[key]}}' for key in keys)
            print(' ' * indent + header)
            for item in lst[:max_line_length]:
                row = ' '.join(f'{str(item[key]):<{max_lens[key]}}' for key in keys)
                print(' ' * indent + row)
            if len(lst) > max_line_length:
                print(' ' * indent + '...')
        else:
            max_len = max(len(str(item)) for item in lst)
            row_len = 80 // (max_len + 2)
            for i in range(0, min(len(lst), 10), row_len):
                row_items = lst[i:i + row_len]
                print(' ' * indent + ' '.join(f'{str(item):<{max_len}}' for item in row_items))
            if len(lst) > max_line_length:
                print(' ' * indent + '...')

    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, dict):
                print(' ' * indent + str(key) + ':')
                custom_pretty_print(value, indent + 4)
            elif isinstance(value, list):
                print(' ' * indent + str(key) + ':')
                if is_short_list(value):
                    print(' ' * indent + '[', ', '.join(map(str, value[:10])), ', ...]' if len(value) > 10 else ']')
                else:
                    if all(isinstance(item, dict) for item in value):
                        print_aligned_list(value, indent + 4)
                    else:
                        print(' ' * indent + '[')
                        for item in value[:max_line_length]:
                            if isinstance(item, dict):
                                custom_pretty_print(item, indent + 4)
                            else:
                                print(' ' * (indent + 4) + str(item) + ',')
                        if len(value) > max_line_length:
                            print(' ' * (indent + 4) + '...')
                        print(' ' * indent + ']')
            else:
                # Print key-value pair in a single line if the combined length is short
                if len(str(key)) + len(str(value)) + 2 <= max_inline_length:
                    print(' ' * indent + f'{key}: {value}')
                else:
                    print(' ' * indent + str(key) + ':')
                    print(' ' * (indent + 4) + str(value))
    elif isinstance(obj, list):
        if is_short_list(obj):
            print(' ' * indent + '[', ', '.join(map(str, obj[:10])), ', ...]' if len(obj) > 10 else ']')
        else:
            print(' ' * indent + '[')
            print_aligned_list(obj, indent + 4)
            print(' ' * indent + ']')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


from common_file_cache import file_cache


def _dictConvertValue(obj, convertFunc):
    """
    对字典中指定 `key` 对应的 `value` 调用转换函数，而不修改 `key`。
    
    :param data: dict or其他嵌套数据结构（list, dict, tuple, set 等）
    :param target_key: 要匹配的目标 key
    :param transform_func: 转换函数，仅对 target_key 的 value 应用
    :return: 转换后的数据结构
    """
    if isinstance(obj, dict):
        # 如果是字典，检查每个键值对
        return {
            key: _dictConvertValue(value, convertFunc)
            for key, value in obj.items()
        }
    elif isinstance(obj, list):
        # 如果是列表，递归处理每个元素
        return [_dictConvertValue(item, convertFunc) for item in obj]
    elif isinstance(obj, tuple):
        # 如果是元组，递归处理每个元素，并保持元组类型
        return tuple(_dictConvertValue(item, convertFunc) for item in obj)
    else:
        # 其他类型直接返回
        return convertFunc(obj)
def dictConvertValue(container, target_key, convertFunc):
    """
    对字典中指定 `key` 对应的 `value` 调用转换函数，而不修改 `key`。
    
    :param data: dict or其他嵌套数据结构（list, dict, tuple, set 等）
    :param target_key: 要匹配的目标 key
    :param transform_func: 转换函数，仅对 target_key 的 value 应用
    :return: 转换后的数据结构
    """
    obj = container[target_key]
    container[target_key] = _dictConvertValue(obj, convertFunc)
    return container


def debugTensor(t, title):
    if type(t) == torch.Tensor:
        t = list(t.reshape(-1))
    elif type(t) != list:
        t = [t,]
    print(title)
    txt = ''
    for i in range(len(t)):
        txt += f'{i}: {t[i]:.4f} '
        if i % 10 == 9:
            print(txt)
            txt = ''
    if len(txt) > 0:
        print(txt)