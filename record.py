from common import *

import sqlite3
import json

conn = sqlite3.connect(db_file)
cursor = conn.cursor()
def read_record(record_id):
    cursor.execute('SELECT data FROM records WHERE id = ?', (record_id,))
    row = cursor.fetchone()
    return json.loads(row[0]) if row else None
# conn.close()

def main():
    # 指定要读取的ID
    record_id = 10005  # 示例ID，可以更改为其他ID

    # 读取记录
    record = read_record(record_id)
    
    if record:
        # 使用 custom_pretty_print 打印
        custom_pretty_print(record['script']['a'], max_line_length=200)
    else:
        print(f"Record with ID {record_id} not found.")

if __name__ == "__main__":
    #main()
    cursor.execute('SELECT * FROM records order by id desc limit 1')
    row = cursor.fetchone()
    print(row)