import sys

"""File operation:
- write()     # 并不直接将数据写入文件，而是先写入内存中特定的缓冲区
- flush()     # 刷新缓冲区，即将缓冲区中的数据立刻写入文件，同时清空缓冲区
- close()     # 内部先调用 flush 刷新缓冲区，再执行关闭操作，这样即使缓冲区数据未满也能保证数据的完整
- truncate()  # 截断文件，如果没指定 size，表示从当前位置截断[如刚打开文件，即文件首，全部清除]
"""


class Logger:
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w', encoding='UTF-8')  # 打开时自动清空文件

    def write(self, msg):
        self.terminal.write(msg)  # 命令行打印
        self.log.write(msg)

    def flush(self):  # 必有，不然 AttributeError: 'Logger' object has no attribute 'flush'
        pass

    def close(self):
        self.log.close()
