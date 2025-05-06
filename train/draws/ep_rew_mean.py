import matplotlib.pyplot as plt  # 导入绘图库
import numpy as np  # 导入数值计算库
import pandas as pd  # 导入数据处理库
import os  # 导入操作系统接口库

def plot_comparison(current_dir='.'):
    """
    读取并对比绘制两个CSV文件的数据
    
    参数说明:
        current_dir (str): CSV文件所在的目录路径（默认为当前目录）
    
    文件要求:
        - CSV文件必须包含 'Step' 和 'Value' 两列
        - 目录下至少需要两个CSV文件用于对比
    
    图表设置:
        - 图表大小: 12 x 7 英寸
        - DPI: 300
        - 第一个文件用实线表示
        - 第二个文件用虚线表示
    """
    # ===== 字体设置 =====
    # 设置中文字体，避免中文显示为方块
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    
    # ===== 文件检查 =====
    # 获取目录下所有CSV文件
    csv_files = [f for f in os.listdir(current_dir) if f.endswith('.csv')]
    
    # 检查是否有足够的文件进行对比
    if len(csv_files) < 2:
        print("错误：需要至少两个CSV文件进行对比")
        return
    
    # ===== 创建图表 =====
    # 设置图表大小（单位：英寸）
    plt.figure(figsize=(12, 7))
    
    # ===== 数据处理和绘图 =====
    # 只处理前两个CSV文件
    for i, csv_file in enumerate(csv_files[:2]):
        file_path = os.path.join(current_dir, csv_file)
        try:
            # 读取CSV文件数据
            df = pd.read_csv(file_path)
            
            # 验证数据列是否存在
            if 'Step' not in df.columns or 'Value' not in df.columns:
                print(f"错误：{csv_file} 必须包含 'Step' 和 'Value' 列")
                continue
            
            # 绘制数据曲线
            plt.plot(df['Step'],  # X轴数据（步数）
                    df['Value'],  # Y轴数据（值）
                    label=os.path.splitext(csv_file)[0],  # 图例标签（文件名，不含扩展名）
                    linestyle=['-', '-'][i],  # 线型：实线或虚线
                    linewidth=2)  # 线宽
            
        except Exception as e:
            print(f"读取文件 {csv_file} 时出错: {str(e)}")
            continue
    
    # ===== 图表属性设置 =====
    # 设置标题和轴标签
    plt.title('训练过程对比')  # 可修改为其他标题
    plt.xlabel('训练步数 (Step)')  # X轴标签
    plt.ylabel('值 (Value)')  # Y轴标签
    
    # 设置网格
    plt.grid(True,  # 显示网格
             linestyle='--',  # 网格线型为虚线
             alpha=0.7)  # 网格透明度
    
    # 显示图例
    plt.legend()
    
    # ===== 保存图表 =====
    # 设置保存路径和参数
    save_path = os.path.join(current_dir, 'comparison_plot.png')
    plt.savefig(save_path,  # 保存路径
                dpi=300,  # 图像质量
                bbox_inches='tight')  # 自动调整边距
    plt.close()  # 关闭图表，释放内存
    
    # 打印保存成功信息
    print(f'对比图表已保存至: {save_path}')

# ===== 主程序入口 =====
if __name__ == '__main__':
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 调用绘图函数
    plot_comparison(script_dir)
