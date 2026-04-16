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

raw_data = pd.read_csv(
    'ICData.csv',
    sep=',',
    dtype=str,                # 全部读为字符串，避免任何自动类型推断
    skipinitialspace=True,    # 跳过逗号后的空格
    encoding='utf-8'
)
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

# 上车站点、下车站点、刷卡类型、小时等转为数值
raw_data['上车站点'] = pd.to_numeric(raw_data['上车站点'], errors='coerce')
raw_data['下车站点'] = pd.to_numeric(raw_data['下车站点'], errors='coerce')
raw_data['刷卡类型'] = pd.to_numeric(raw_data['刷卡类型'], errors='coerce')
raw_data['hour'] = raw_data['交易时间'].dt.hour

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

invalid_ids = ['', 'nan', 'NaN', 'None', '0', '0.0']
clean_data = clean_data[~clean_data['驾驶员编号'].isin(invalid_ids)]
clean_data = clean_data[~clean_data['线路号'].isin(invalid_ids)]
clean_data = clean_data[~clean_data['车辆编号'].isin(invalid_ids)]
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
plt.savefig('hour_distribution.png',dpi=150)
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

#3.2 使用seaborn barplot 可视化各线路平均搭乘点和标准差

# 获取均值最高的前15条线路编号（按均值降序）
top15_route_ids = route_stop_stats.head(15)['线路号'].tolist()

# 从清洗后的数据中筛选出这些线路的记录
data_top15 = clean_data[clean_data['线路号'].isin(top15_route_ids)].copy()

# 将线路号转为字符串，避免 seaborn 将其视为数值
data_top15['线路号'] = data_top15['线路号'].astype(str)

# 指定水平条形图的显示顺序：均值从高到低在图上对应从上到下（即 y 轴反向）
order_for_plot = [str(route) for route in top15_route_ids[::-1]]

plt.figure(figsize=(10, 8))

sns.barplot(
    data=data_top15,
    x='ride_stops',
    y='线路号',
    order=order_for_plot,
    errorbar='sd',   # 设置误差棒
    capsize=0.3,    # 设置capsize
    palette='Blues_d',  #使用palette调色板设置渐变色
    edgecolor='black',
    linewidth=0.5   #线宽，使可视化更清晰
)
#设置图例
plt.xlabel('平均搭乘站点数', fontsize=12)
plt.ylabel('线路号', fontsize=12)
plt.title('各线路平均搭乘站点数 Top 15（含标准差）', fontsize=14)
#设置合适图例比例
plt.xlim(0, data_top15['ride_stops'].max() * 0.5)
plt.grid(axis='x', linestyle='--', alpha=0.6)
#保存图片
plt.tight_layout()
plt.savefig('route_stops.png', dpi=150)
plt.close()




#Task4: 高峰小时系数计算

#4.1 识别高峰小时
# 利用前面 boarding_data，按小时统计刷卡总量
hourly_total_boarding = boarding_data.groupby('hour').size()

# 利用indxmax 找出刷卡量最大的小时
peak_hour_value = hourly_total_boarding.idxmax()
peak_hour_volume = hourly_total_boarding.max()

print(f"\n高峰小时为 {peak_hour_value}:00 ~ {peak_hour_value+1}:00，刷卡量 {peak_hour_volume} 次")

#4.2 提取高峰小时内数据
#AI生成核心代码

#筛选出该小时内的所有上车记录
peak_hour_data = boarding_data[boarding_data['hour'] == peak_hour_value].copy()
#以交易时间为索引，方便后面使用resample进行时间索引
peak_hour_data = peak_hour_data.set_index('交易时间').sort_index()

#4.3 5分钟粒度统计
five_minute_counts = peak_hour_data.resample('5min').size()

# 找出最大的5分钟刷卡量
max_5min_volume = five_minute_counts.max()
max_5min_start_time = five_minute_counts.idxmax()  #找出最大五分钟刷卡量的起始时间

# 计算窗口结束时间（起始时间 + 5分钟）
max_5min_end_time = max_5min_start_time + pd.Timedelta(minutes=5)

# 计算 PHF5 = 高峰小时总刷卡量 / (12 × 最大5分钟刷卡量)
PHF5_value = peak_hour_volume / (12 * max_5min_volume)
print(f"最大5分钟刷卡量（{max_5min_start_time.strftime('%H:%M')}~{max_5min_end_time.strftime('%H:%M')}）：{max_5min_volume} 次")
print(f"PHF5 = {peak_hour_volume} / (12 × {max_5min_volume}) = {PHF5_value:.4f}")

#4.4 15分钟粒度统计
# 找出最大的15分钟刷卡量
fifteen_minute_counts = peak_hour_data.resample('15min').size()

max_15min_volume = fifteen_minute_counts.max()
max_15min_start_time = fifteen_minute_counts.idxmax()
# 计算窗口结束时间（起始时间 + 15分钟）
max_15min_end_time = max_15min_start_time + pd.Timedelta(minutes=15)

# 计算 PHF15 = 高峰小时总刷卡量 / (4× 最大15分钟刷卡量)
PHF15_value = peak_hour_volume / (4 * max_15min_volume)

print(f"最大15分钟刷卡量（{max_15min_start_time.strftime('%H:%M')}~{max_15min_end_time.strftime('%H:%M')}）：{max_15min_volume} 次")
print(f"PHF15 = {peak_hour_volume} / (4 × {max_15min_volume}) = {PHF15_value:.4f}")


#Task5：路线驾驶员信息批量导出

#复制一份数据，防止task5对数据产生污染
clean_data_backup = clean_data.copy()
#5.1 筛选线路号在 1101 至 1120 之间的所有记录

target_line_numbers = [str(i) for i in range(1101, 1121)]  # 改为字符串
filtered_line_data = clean_data[clean_data['线路号'].isin(target_line_numbers)]
print(f"筛选到 {len(filtered_line_data)} 条记录，涉及线路数：{filtered_line_data['线路号'].nunique()} 条")

#5.2创建输出文件夹 

output_folder = '线路驾驶员信息'
os.makedirs(output_folder, exist_ok=True) 
#AI 生成核心代码

#5.3 按路线号分组并输出路线驾驶员信息txt

for line_number in target_line_numbers:
    # 提取该线路的所有记录
    single_line_data = filtered_line_data[filtered_line_data['线路号'] == line_number]
    
    if single_line_data.empty:
        print(f"警告：线路 {line_number} 无数据，跳过")
        continue
    
    # 选取「车辆编号」和「驾驶员编号」两列，并去重
    vehicle_driver_pairs = single_line_data[['车辆编号', '驾驶员编号']].drop_duplicates()
    
    # 将驾驶员编号转为字符串，防止写入时变成科学计数法
    vehicle_driver_pairs['驾驶员编号'] = vehicle_driver_pairs['驾驶员编号'].astype(str)
    
    # 构造文件路径
    txt_file_path = os.path.join(output_folder, f'{line_number}.txt')
    
    # 写入文件
    with open(txt_file_path, 'w', encoding='utf-8') as file_handle:
        file_handle.write(f'线路号: {line_number}\n')
        file_handle.write('车辆编号 驾驶员编号\n')
        for _, row_data in vehicle_driver_pairs.iterrows():
            file_handle.write(f"{row_data['车辆编号']} {row_data['驾驶员编号']}\n")
    
    print(f"已生成：{txt_file_path}")

print(f"\n所有 txt 文件已保存至文件夹：{os.path.abspath(output_folder)}")


#Task6：服务绩效排名与可视化——热力图


#6.1 统计各维度 Top10
top10_drivers = clean_data_backup['驾驶员编号'].value_counts().head(10)
top10_routes = clean_data_backup['线路号'].value_counts().head(10)
top10_stops = clean_data_backup['上车站点'].value_counts().head(10)
top10_vehicles = clean_data_backup['车辆编号'].value_counts().head(10)

print("\n>>> Top 10 司机（驾驶员编号）：")
print(top10_drivers)
print("\n>>> Top 10 线路：")
print(top10_routes)
print("\n>>> Top 10 上车站点：")
print(top10_stops)
print("\n>>> Top 10 车辆：")
print(top10_vehicles)

#6.2 热力图可视化
def series_to_fixed_length_list(series_obj, length=10):
    """将 Series 转为指定长度的列表，不足补0，超过截断"""
    value_list = series_obj.tolist()
    if len(value_list) < length:
        value_list += [0] * (length - len(value_list))
    return value_list[:length]

# 构建 4 行 10 列的矩阵，每一行对应一个维度
heatmap_data_matrix = np.array([
    series_to_fixed_length_list(top10_drivers),
    series_to_fixed_length_list(top10_routes),
    series_to_fixed_length_list(top10_stops),
    series_to_fixed_length_list(top10_vehicles)
])

# 行标签（维度名称）和列标签（Top1~Top10）
dimension_labels = ['司机', '线路', '上车站点', '车辆']
rank_labels = [f'Top{i+1}' for i in range(10)]

# ---------- 6.3 绘制 seaborn 热力图 ----------
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data_matrix, 
            annot=True,           # 在每个格子中显示数值
            fmt='d',              # 数值格式为整数
            cmap='YlOrRd',        # 颜色映射（黄-橙-红）
            xticklabels=rank_labels,
            yticklabels=dimension_labels,
            linewidths=0.5,
            linecolor='gray',
            cbar_kws={'label': '服务人次'})   # 颜色条标签

plt.title('服务绩效 Top10 热力图\n(司机·线路·上车站点·车辆)', fontsize=14)
plt.xlabel('排名', fontsize=12)
plt.ylabel('维度', fontsize=12)
plt.xticks(rotation=0)   # x轴标签不旋转

plt.tight_layout()
plt.savefig('performance_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n>>> 热力图已保存为 performance_heatmap.png")

# 结论说明
print("\n>>> 热力图结论说明：")
print(
    "从热力图中可观察到以下服务绩效规律：\n"
    "1. 司机维度：编号为 1599 的司机服务人次最高，为 1334 次，远超其他司机。\n"
    "2. 线路维度：线路 1101 服务人次最多，达到 3782 次，是高峰线路。\n"
    "3. 上车站点：站点 1 服务 6722 人次，遥遥领先，为高峰站点。\n"
    "4. 车辆维度：车辆 9132 服务人次最高，为 1353 次，头部集中现象明显。\n"
    "综上，公交系统在司机、线路、站点、车辆层面均存在显著的头部集中现象。"
)

