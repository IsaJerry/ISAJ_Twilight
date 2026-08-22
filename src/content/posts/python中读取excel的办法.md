---
title: Python中读取excel的办法
published: 2026-08-22T21:48:00.000+08:00
updated: 2026-08-22T21:48:00.000+08:00
tags:
  - Python
category: Python
draft: false
---
pip 安装 openpyxl

然后如下进行使用：

```
def get_data(filename, data_num, sheet1_name, sheet2_name, is_target=0):
    book = openpyxl.load_workbook(filename, read_only=True)
    bandgaps = np.array(list(book[sheet1_name].values))[:data_num, :].astype(np.float32)
    parameters = np.array(list(book[sheet2_name].values))[:data_num, :].astype(np.float32)
    book.close()
```
