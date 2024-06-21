from common import *

import requests
import json
import sqlite3
from tqdm import tqdm

def main():

    # 创建数据库表
    def initialize_db():
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY,
                data TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS progress (
                id INTEGER PRIMARY KEY
            )
        ''')
        conn.commit()
        conn.close()

    # 获取上次下载的ID
    def get_last_downloaded_id():
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute('SELECT MAX(id) FROM progress')
        row = cursor.fetchone()
        conn.close()
        return row[0] if row[0] is not None else id_start

    # 保存记录到数据库
    def save_record(record_id, data):
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute('INSERT OR IGNORE INTO records (id, data) VALUES (?, ?)', (record_id, json.dumps(data)))
        cursor.execute('INSERT OR REPLACE INTO progress (id) VALUES (?)', (record_id,))
        conn.commit()
        conn.close()

    # 发送请求并获取数据
    def fetch_data(record_id):
        url = 'https://tziakcha.xyz/_qry/record/'
        headers = {
            'Accept': '*/*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7',
            'Connection': 'keep-alive',
            'Content-Type': 'text/plain;charset=UTF-8',
            'Cookie': '__p=ee469c7bc00d96ca79a40ccf62adf02506bc473d7a30a535e190aeb3ab60f98a',
            'Origin': 'https://tziakcha.xyz',
            'Referer': 'https://tziakcha.xyz/record/?id=1659451',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
            'sec-ch-ua': '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Linux"'
        }
        data = f'id={record_id}'
        response = requests.post(url, headers=headers, data=data)
        return response.json()

    # 初始化数据库
    initialize_db()

    # 获取上次下载的ID
    last_downloaded_id = get_last_downloaded_id()

    # 下载数据并保存到数据库
    with tqdm(total=id_end - id_start + 1, initial=last_downloaded_id - id_start, unit='record') as pbar:
        for record_id in range(last_downloaded_id, id_end + 1):
            data = fetch_data(record_id)
            save_record(record_id, data)
            pbar.update(1)

    print("数据下载完成")

if __name__ == '__main__':
    main()