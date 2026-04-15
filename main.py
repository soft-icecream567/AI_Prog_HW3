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


#Task2：时间分布分析

#2.1 早晚时段刷卡量统计

#人工智能核心代码
#筛选删除个刷卡记录
boarding_data = clean_data[clean_data['刷卡类型'] == 0].copy()
print(f"上车刷卡总记录数：{len(boarding_data)} 条")

#使用numpy bool索引统计早晚时段
# 将 hour 列转换为 numpy 数组（必须使用 numpy 完成统计）
hour_array = boarding_data['hour'].values

# 早峰前：hour < 7
is_early_morning = hour_array < 7 
early_count = np.sum(is_early_morning)   # np.sum 对布尔数组求和，True=1，False=0

# 深夜：hour >= 22
is_late_night = hour_array >= 22
late_count = np.sum(is_late_night)

total_boarding_count = len(boarding_data)#计算总数方便输出占比

#输出刷卡量和占比
print(f"\n早峰前时段 (<7:00) 刷卡量：{early_count} 次，占比 {early_count/total_boarding_count:.2%}")
print(f"深夜时段 (≥22:00) 刷卡量：{late_count} 次，占比 {late_count/total_boarding_count:.2%}")

#2.2 24小时刷卡量可视化（使用matplotlib）
# 按小时统计上车刷卡量
hourly_boarding_counts = boarding_data.groupby('hour').size()

#确保 0~23 每个小时都有值
full_hour_index = range(24)
#缺失的小时补 0
hourly_boarding_counts = hourly_boarding_counts.reindex(full_hour_index, fill_value=0)

#人工智能输出核心代码

# 创建颜色列表：早峰前 (<7) 和深夜 (>=22) 用橙色高亮，其余用蓝色
bar_colors = []
for hour_value in full_hour_index:
    if hour_value < 7 or hour_value >= 22:
        bar_colors.append('orange')      # 高亮颜色
    else:
        bar_colors.append('steelblue')   # 非高光颜色
#其中注意在Task1 前的准备工作已完成中文、负号的输出格式优化

# 绘图图形大小设置
plt.figure(figsize=(12, 6))

# 绘制柱状图：x 轴为小时 0~23，y 轴为对应刷卡量，颜色按 bar_colors 列表
plt.bar(hourly_boarding_counts.index, hourly_boarding_counts.values, 
        color=bar_colors, edgecolor='black', linewidth=0.5)#对宽度 边框颜色也进行设置，优化输出

# 添加标题和坐标轴标签（中文）
plt.title('24小时公交上车刷卡量分布', fontsize=16)
plt.xlabel('小时', fontsize=12)
plt.ylabel('刷卡量（次）', fontsize=12)

# 设置 x 轴刻度：显示 0,2,4,...,22，步长为2
plt.xticks(ticks=range(0, 24, 2), labels=range(0, 24, 2))

# 添加水平网格线（只对 y 轴方向，便于阅读数值）
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 手动创建图例（因为颜色是手动指定的，需要自定义图例句柄）
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor='orange', label='早峰前/深夜时段'),
    Patch(facecolor='steelblue', label='其余时段')
]
plt.legend(handles=legend_handles)

# 自动调整布局，防止标签被裁剪
plt.tight_layout()

# 保存图像
plt.savefig('Task2_24小时刷卡量分布可视化.png')
plt.close()   # 关闭当前图像，释放内存


#Task3：路线站点分析
#AI生成核心代码

#3.1 定义函数并计算各线路平均搭乘点数目

#函数定义
#函数用来计算平均搭乘点和标准差
def analyze_route_stops(df, route_col='线路号', stops_col='ride_stops'):
    
    # 按线路号分组，对搭乘站点数求平均值和标准差
    grouped_stats = df.groupby(route_col)[stops_col].agg(['mean', 'std']).reset_index()
    # 重命名列，使其符合返回要求
    grouped_stats.columns = [route_col, 'mean_stops', 'std_stops']
    # 按平均搭乘站点数降序排序
    result_df = grouped_stats.sort_values('mean_stops', ascending=False).reset_index(drop=True)
    return result_df

route_stop_stats=analyze_route_stops(clean_data)

print("\n 前10行各线路平均搭乘点数：")
print(route_stop_stats.head(10))

#3.2 使用seaborn barplot绘制
