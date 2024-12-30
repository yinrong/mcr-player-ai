import os
import pickle
import functools
import hashlib

def _hash_args(*args, **kwargs):
    """
    对函数的参数进行哈希处理，返回 MD5 的前几位（默认8位）。
    """
    # 将 args 和 kwargs 转换为字符串表示
    key = str(args) + str(sorted(kwargs.items()))
    # 生成 MD5 哈希
    md5_hash = hashlib.md5(key.encode('utf-8')).hexdigest()
    return md5_hash[:8]  # 返回前8位，足够区分

def file_cache(func):
    """
    使用 pickle 实现函数调用结果的文件缓存，文件名使用函数名 + 参数哈希。
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 确保 .file_cache 目录存在
        cache_dir = os.path.join(os.getcwd(), '.file_cache')
        os.makedirs(cache_dir, exist_ok=True)

        # 构造文件名：函数名 + 参数哈希
        func_name = func.__name__
        arg_hash = _hash_args(*args, **kwargs)
        filename = f"{func_name}__{arg_hash}.pkl"
        cache_path = os.path.join(cache_dir, filename)

        # 如果缓存文件存在，直接读取并返回结果
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        # 计算函数结果并缓存到文件
        result = func(*args, **kwargs)
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f)

        return result

    return wrapper
