#导入库函数
import numpy as np                
import pandas as pd               
import matplotlib.pyplot as plt   
import seaborn as sns             
import os

#设置中文字体，防止可视化出现中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'PingFang SC']
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'PingFang SC']

#Task1:数据预处理

#1.1数据读取

raw_data=pd.read_csv('ICData.csv',sep='\t')
print("\n 数据集前五行为：")
print(raw_data.head())

print("\n数据集基本信息：")
print(f"数据行数：{raw_data.shape[0]} 行")
print(f"数据列数：{raw_data.shape[1]} 列")
print("\n各列数据类型：")
print(raw_data.dtypes)

#1.2 时间解析
#使用 pd.to_datetime 转换，format ,让时间输出格式更加清晰
raw_data["交易时间"]=pd.to_datatime(raw_data['交易时间'],format='%Y/%m/%d %H:%M')

# 从交易时间中提取小时（整数 0~23），新增为 'hour' 列
raw_data['hour'] = raw_data['交易时间'].dt.hour

print("打印新增hour列：")
print(raw_data[['交易时间','hour']].head())#后续时间分析必须依赖hour


