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

raw_data=pd.read_csv('ICData.csv',sep=',')#使用‘，’分割数据，得到首标题
#打印数据集前五行
print("\n 数据集前五行为：")
print(raw_data.head())
print(raw_data.columns.tolist())
#打印数据集基本信息
print("\n数据集基本信息：")
print(f"数据行数：{raw_data.shape[0]} 行")
print(f"数据列数：{raw_data.shape[1]} 列")
print("\n各列数据类型：")
print(raw_data.dtypes)

#1.2 时间解析
#人工智能生成核心代码
#使用 pd.to_datetime 转换，打印时间
raw_data["交易时间"]=pd.to_datetime(raw_data['交易时间'])

# 从交易时间中提取小时（整数 0~23），新增为 'hour' 列
raw_data['hour'] = raw_data['交易时间'].dt.hour

print("打印新增hour列前五行：")
print(raw_data[['交易时间','hour']].head())#后续时间分析必须依赖hour


#1.3 构造衍生字段-搭乘站点数
#人工智能生成核心代码
#确保上下车站点为数值类型，errors设置防止出现的错误引发程序崩溃
raw_data['上车站点'] = pd.to_numeric(raw_data['上车站点'], errors='coerce')
raw_data['下车站点'] = pd.to_numeric(raw_data['下车站点'], errors='coerce')

#计算搭乘站点数：两站点序号之差的绝对值
raw_data['ride_stops'] = (raw_data['下车站点'] - raw_data['上车站点']).abs()

# 记录删除前的行数，用于后续打印
rows_before_drop = len(raw_data)

#保留ride_stops不为0 的行
clean_data = raw_data[raw_data['ride_stops'] != 0]

rows_after_drop = len(clean_data)
deleted_rows = rows_before_drop - rows_after_drop

print(f"\n删除 ride_stops = 0 的异常记录数：{deleted_rows} 行")
print(f"删除后数据集剩余：{rows_after_drop} 行")

#1.4缺失值检查与处理
print("\n各列缺失值数量统计：")
print(clean_data.isnull().sum())

#处理策略：若存在缺失值，由于数据量较大，删除部分值不影响数据分析，因此直接删除对应行
key_columns = ['交易时间', 'hour', 'ride_stops', '线路号', '上车站点', '下车站点']
clean_data = clean_data.dropna(subset=key_columns)#查询缺失数据

print(f"\n缺失值处理后最终数据行数：{len(clean_data)} 行")
