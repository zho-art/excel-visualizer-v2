from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import io
import base64
import json
import traceback
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import re

# 设置中文字体
plt.rcParams["font.family"] = ["DejaVu Sans", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
sns.set(font_scale=1.0)

app = Flask(__name__)

class ExcelVisualizer:
    def __init__(self):
        self.df = None
        self.valid_series = {}
        self.categories = []
        
    def load_data(self, file_content, filename):
        """从文件内容加载数据"""
        try:
            if filename.endswith('.csv'):
                self.df = pd.read_csv(io.BytesIO(file_content))
            else:
                self.df = pd.read_excel(io.BytesIO(file_content))
            
            print(f"成功读取数据，维度: {self.df.shape}")
            return True, "数据加载成功"
        except Exception as e:
            return False, f"读取文件失败: {str(e)}"
    
    def prepare_data_simple(self):
        """简化的数据准备逻辑"""
        try:
            if self.df is None or self.df.empty:
                return False, "没有有效数据"
            
            # 使用前15行8列作为数据区域
            sample_df = self.df.iloc[:15, :8].copy()
            
            # 识别数值列
            numeric_cols = []
            for col in sample_df.columns:
                numeric_ratio = pd.to_numeric(sample_df[col], errors='coerce').notna().mean()
                if numeric_ratio > 0.3:  # 降低阈值到30%
                    numeric_cols.append(col)
            
            if not numeric_cols:
                # 如果没有数值列，使用所有列并尝试转换
                numeric_cols = sample_df.columns.tolist()
            
            # 准备数据系列
            self.valid_series = {}
            
            # 使用索引作为类别
            self.categories = [f"项目{i+1}" for i in range(min(15, len(sample_df)))]
            
            # 为每个数值列创建数据系列
            for col in numeric_cols[:5]:  # 最多5个系列
                try:
                    series_data = pd.to_numeric(sample_df[col], errors='coerce')
                    # 处理NaN值
                    if series_data.isna().all():
                        series_data = pd.Series(range(len(sample_df)))
                    
                    series_data = series_data.fillna(0).tolist()
                    self.valid_series[str(col)] = series_data[:15]  # 限制15个数据点
                except:
                    continue
            
            # 如果没有生成有效数据，创建示例数据
            if not self.valid_series:
                self.categories = [f"类别{i+1}" for i in range(6)]
                self.valid_series = {
                    "系列1": [10, 20, 15, 25, 30, 18],
                    "系列2": [5, 15, 10, 20, 25, 12],
                    "系列3": [8, 12, 18, 22, 16, 28]
                }
            
            return True, f"成功准备 {len(self.valid_series)} 个数据系列"
            
        except Exception as e:
            return False, f"数据准备失败: {str(e)}"

    def plot_to_base64(self):
        """将matplotlib图形转换为base64字符串"""
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()  # 重要：关闭图形释放内存
        return img_base64

    def generate_chart(self, chart_type, title="图表"):
        """生成指定类型的图表并返回base64图像"""
        try:
            if not self.valid_series:
                return False, "没有有效数据", None
            
            chart_methods = {
                "柱状图": self._generate_bar_chart,
                "饼图": self._generate_pie_chart,
                "折线图": self._generate_line_chart,
                "面积图": self._generate_area_chart,
                "散点图": self._generate_scatter_chart,
                "气泡图": self._generate_bubble_chart,
                "热力图": self._generate_heatmap,
                "直方图": self._generate_histogram,
                "箱线图": self._generate_boxplot,
                "组合图": self._generate_combo_chart,
                "条形图": self._generate_barh_chart
            }
            
            if chart_type in chart_methods:
                return chart_methods[chart_type](title)
            else:
                return self._generate_bar_chart(title)  # 默认柱状图
                
        except Exception as e:
            return False, f"生成{chart_type}失败: {str(e)}", None

    def _generate_bar_chart(self, title):
        """生成柱状图"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            n_series = len(self.valid_series)
            indices = np.arange(len(self.categories))
            bar_width = 0.8 / max(1, n_series)
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            
            for i, (series_name, data) in enumerate(self.valid_series.items()):
                color = colors[i % len(colors)]
                bars = ax.bar(
                    indices + i * bar_width, 
                    data[:len(self.categories)], 
                    bar_width, 
                    label=series_name, 
                    color=color, 
                    alpha=0.8
                )
            
            ax.set_title(title, fontsize=16, pad=20)
            ax.set_xlabel("类别", fontsize=12)
            ax.set_ylabel("数值", fontsize=12)
            ax.set_xticks(indices + bar_width * (n_series - 1) / 2)
            ax.set_xticklabels(self.categories, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, axis='y', alpha=0.3)
            
            plt.tight_layout()
            img_base64 = self.plot_to_base64()
            return True, "柱状图生成成功", img_base64
            
        except Exception as e:
            return False, f"生成柱状图失败: {str(e)}", None

    def _generate_pie_chart(self, title):
        """生成饼图"""
        try:
            if not self.valid_series:
                return False, "没有数据", None
                
            # 使用第一个数据系列
            first_series = list(self.valid_series.values())[0]
            series_name = list(self.valid_series.keys())[0]
            
            # 过滤有效数据
            data_pairs = []
            for i, value in enumerate(first_series):
                if value > 0 and i < len(self.categories):
                    data_pairs.append((self.categories[i], value))
            
            if not data_pairs:
                # 如果没有正数数据，使用绝对值
                for i, value in enumerate(first_series):
                    if i < len(self.categories) and abs(value) > 0:
                        data_pairs.append((self.categories[i], abs(value)))
            
            if len(data_pairs) < 2:
                return False, "饼图需要至少2个有效数据点", None
            
            labels, values = zip(*data_pairs[:8])  # 最多8个扇区
            
            fig, ax = plt.subplots(figsize=(10, 8))
            wedges, texts, autotexts = ax.pie(
                values, labels=labels, autopct='%1.1f%%', 
                startangle=90, shadow=True
            )
            
            ax.set_title(f"{title}\n({series_name})", fontsize=14, pad=20)
            ax.axis('equal')
            
            img_base64 = self.plot_to_base64()
            return True, "饼图生成成功", img_base64
            
        except Exception as e:
            return False, f"生成饼图失败: {str(e)}", None

    def _generate_line_chart(self, title):
        """生成折线图"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            markers = ['o', 's', '^', 'D', 'v']
            
            for i, (series_name, data) in enumerate(self.valid_series.items()):
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]
                
                ax.plot(
                    range(len(self.categories)), 
                    data[:len(self.categories)], 
                    label=series_name,
                    color=color, 
                    marker=marker,
                    linewidth=2,
                    markersize=6
                )
            
            ax.set_title(title, fontsize=16, pad=20)
            ax.set_xlabel("类别", fontsize=12)
            ax.set_ylabel("数值", fontsize=12)
            ax.set_xticks(range(len(self.categories)))
            ax.set_xticklabels(self.categories, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            img_base64 = self.plot_to_base64()
            return True, "折线图生成成功", img_base64
            
        except Exception as e:
            return False, f"生成折线图失败: {str(e)}", None

    def _generate_area_chart(self, title):
        """生成面积图"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            x = range(len(self.categories))
            
            # 如果是多个系列，使用堆叠面积图
            if len(self.valid_series) > 1:
                data_array = np.array([data[:len(self.categories)] for data in self.valid_series.values()])
                ax.stackplot(x, data_array, labels=self.valid_series.keys(), alpha=0.7)
            else:
                # 单个系列使用普通面积图
                first_series = list(self.valid_series.values())[0]
                ax.fill_between(x, first_series[:len(self.categories)], alpha=0.7)
                ax.plot(x, first_series[:len(self.categories)], linewidth=2)
            
            ax.set_title(title, fontsize=16, pad=20)
            ax.set_xlabel("类别", fontsize=12)
            ax.set_ylabel("数值", fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(self.categories, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            img_base64 = self.plot_to_base64()
            return True, "面积图生成成功", img_base64
            
        except Exception as e:
            return False, f"生成面积图失败: {str(e)}", None

    def _generate_scatter_chart(self, title):
        """生成散点图"""
        try:
            if len(self.valid_series) < 2:
                return False, "散点图需要至少2个数据系列", None
            
            series_list = list(self.valid_series.items())
            x_data = series_list[0][1]
            y_data = series_list[1][1]
            x_name = series_list[0][0]
            y_name = series_list[1][0]
            
            min_len = min(len(x_data), len(y_data), len(self.categories))
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            scatter = ax.scatter(
                x_data[:min_len], 
                y_data[:min_len], 
                c=range(min_len), 
                cmap='viridis', 
                s=100, 
                alpha=0.7
            )
            
            # 添加颜色条
            plt.colorbar(scatter, label='数据点顺序')
            
            ax.set_title(title, fontsize=16, pad=20)
            ax.set_xlabel(x_name, fontsize=12)
            ax.set_ylabel(y_name, fontsize=12)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            img_base64 = self.plot_to_base64()
            return True, "散点图生成成功", img_base64
            
        except Exception as e:
            return False, f"生成散点图失败: {str(e)}", None

    def _generate_bubble_chart(self, title):
        """生成气泡图"""
        try:
            if len(self.valid_series) < 3:
                return False, "气泡图需要至少3个数据系列", None
            
            series_list = list(self.valid_series.items())
            x_data = series_list[0][1]
            y_data = series_list[1][1]
            size_data = series_list[2][1]
            x_name = series_list[0][0]
            y_name = series_list[1][0]
            size_name = series_list[2][0]
            
            min_len = min(len(x_data), len(y_data), len(size_data), len(self.categories))
            
            # 标准化气泡大小
            sizes = np.array(size_data[:min_len])
            if sizes.max() > sizes.min():
                sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min()) * 500 + 50
            else:
                sizes = np.ones(min_len) * 200
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            scatter = ax.scatter(
                x_data[:min_len], 
                y_data[:min_len], 
                s=sizes, 
                c=range(min_len), 
                cmap='viridis', 
                alpha=0.7
            )
            
            # 添加颜色条
            plt.colorbar(scatter, label='数据点顺序')
            
            ax.set_title(title, fontsize=16, pad=20)
            ax.set_xlabel(x_name, fontsize=12)
            ax.set_ylabel(y_name, fontsize=12)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            img_base64 = self.plot_to_base64()
            return True, "气泡图生成成功", img_base64
            
        except Exception as e:
            return False, f"生成气泡图失败: {str(e)}", None

    def _generate_heatmap(self, title):
        """生成热力图"""
        try:
            # 创建数据矩阵
            data_matrix = []
            for series_data in self.valid_series.values():
                data_matrix.append(series_data[:len(self.categories)])
            
            if not data_matrix:
                return False, "没有有效数据生成热力图", None
            
            data_array = np.array(data_matrix)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            im = ax.imshow(data_array, cmap='YlOrRd', aspect='auto')
            
            # 设置坐标轴
            ax.set_xticks(range(len(self.categories)))
            ax.set_yticks(range(len(self.valid_series)))
            ax.set_xticklabels(self.categories, rotation=45, ha='right')
            ax.set_yticklabels(self.valid_series.keys())
            
            # 添加颜色条
            plt.colorbar(im, ax=ax, label='数值')
            
            # 添加数值标注
            for i in range(len(self.valid_series)):
                for j in range(len(self.categories)):
                    if j < len(data_matrix[i]):
                        text = ax.text(j, i, f'{data_matrix[i][j]:.1f}',
                                    ha="center", va="center", color="black", fontsize=8)
            
            ax.set_title(title, fontsize=16, pad=20)
            
            plt.tight_layout()
            img_base64 = self.plot_to_base64()
            return True, "热力图生成成功", img_base64
            
        except Exception as e:
            return False, f"生成热力图失败: {str(e)}", None

    def _generate_histogram(self, title):
        """生成直方图"""
        try:
            if not self.valid_series:
                return False, "没有数据", None
                
            # 使用第一个数据系列
            first_series = list(self.valid_series.values())[0]
            series_name = list(self.valid_series.keys())[0]
            
            # 过滤有效数据
            data = [x for x in first_series if pd.notna(x)]
            
            if len(data) < 3:
                return False, "直方图需要至少3个有效数据点", None
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.hist(data, bins=min(10, len(data)), alpha=0.7, color='skyblue', edgecolor='black')
            
            ax.set_title(f"{title}\n({series_name})", fontsize=16, pad=20)
            ax.set_xlabel("数值", fontsize=12)
            ax.set_ylabel("频数", fontsize=12)
            ax.grid(True, axis='y', alpha=0.3)
            
            plt.tight_layout()
            img_base64 = self.plot_to_base64()
            return True, "直方图生成成功", img_base64
            
        except Exception as e:
            return False, f"生成直方图失败: {str(e)}", None

    def _generate_boxplot(self, title):
        """生成箱线图"""
        try:
            if not self.valid_series:
                return False, "没有数据", None
            
            data = list(self.valid_series.values())
            series_names = list(self.valid_series.keys())
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            box_plot = ax.boxplot(data, labels=series_names, patch_artist=True)
            
            # 设置箱体颜色
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_title(title, fontsize=16, pad=20)
            ax.set_ylabel("数值", fontsize=12)
            ax.grid(True, axis='y', alpha=0.3)
            
            plt.tight_layout()
            img_base64 = self.plot_to_base64()
            return True, "箱线图生成成功", img_base64
            
        except Exception as e:
            return False, f"生成箱线图失败: {str(e)}", None

    def _generate_combo_chart(self, title):
        """生成组合图（柱状图+折线图）"""
        try:
            if len(self.valid_series) < 2:
                return False, "组合图需要至少2个数据系列", None
            
            series_list = list(self.valid_series.items())
            bar_data = series_list[0][1]
            line_data = series_list[1][1]
            bar_name = series_list[0][0]
            line_name = series_list[1][0]
            
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # 柱状图
            bars = ax1.bar(range(len(self.categories)), bar_data[:len(self.categories)], 
                          alpha=0.7, color='skyblue', label=bar_name)
            ax1.set_xlabel("类别", fontsize=12)
            ax1.set_ylabel(bar_name, fontsize=12, color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            
            # 折线图
            ax2 = ax1.twinx()
            ax2.plot(range(len(self.categories)), line_data[:len(self.categories)], 
                    color='red', marker='o', linewidth=2, label=line_name)
            ax2.set_ylabel(line_name, fontsize=12, color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            ax1.set_xticks(range(len(self.categories)))
            ax1.set_xticklabels(self.categories, rotation=45, ha='right')
            
            # 合并图例
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            ax1.set_title(title, fontsize=16, pad=20)
            ax1.grid(True, axis='y', alpha=0.3)
            
            plt.tight_layout()
            img_base64 = self.plot_to_base64()
            return True, "组合图生成成功", img_base64
            
        except Exception as e:
            return False, f"生成组合图失败: {str(e)}", None

    def _generate_barh_chart(self, title):
        """生成水平条形图"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            n_series = len(self.valid_series)
            indices = np.arange(len(self.categories))
            bar_height = 0.8 / max(1, n_series)
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            
            for i, (series_name, data) in enumerate(self.valid_series.items()):
                color = colors[i % len(colors)]
                bars = ax.barh(
                    indices + i * bar_height, 
                    data[:len(self.categories)], 
                    bar_height, 
                    label=series_name, 
                    color=color, 
                    alpha=0.8
                )
            
            ax.set_title(title, fontsize=16, pad=20)
            ax.set_xlabel("数值", fontsize=12)
            ax.set_ylabel("类别", fontsize=12)
            ax.set_yticks(indices + bar_height * (n_series - 1) / 2)
            ax.set_yticklabels(self.categories)
            ax.legend()
            ax.grid(True, axis='x', alpha=0.3)
            
            plt.tight_layout()
            img_base64 = self.plot_to_base64()
            return True, "水平条形图生成成功", img_base64
            
        except Exception as e:
            return False, f"生成水平条形图失败: {str(e)}", None

@app.route('/api/process-excel', methods=['POST'])
def process_excel():
    """处理Excel文件并生成图表的主API"""
    try:
        print("收到图表生成请求")
        
        # 检查文件
        if 'excel_file' not in request.files:
            return jsonify({
                'success': False, 
                'message': '没有上传文件'
            })
        
        file = request.files['excel_file']
        chart_type = request.form.get('chart_type', '柱状图')
        custom_title = request.form.get('title', '数据可视化图表')
        
        if file.filename == '':
            return jsonify({
                'success': False, 
                'message': '文件名为空'
            })
        
        print(f"处理文件: {file.filename}, 图表类型: {chart_type}")
        
        # 初始化可视化器
        visualizer = ExcelVisualizer()
        
        # 读取文件
        file_content = file.read()
        success, message = visualizer.load_data(file_content, file.filename)
        
        if not success:
            return jsonify({
                'success': False, 
                'message': message
            })
        
        # 准备数据
        success, message = visualizer.prepare_data_simple()
        if not success:
            return jsonify({
                'success': False, 
                'message': message
            })
        
        print(f"数据准备完成: {message}")
        
        # 生成图表
        success, message, chart_image = visualizer.generate_chart(chart_type, custom_title)
        
        if success:
            return jsonify({
                'success': True,
                'message': message,
                'chart_image': chart_image,
                'chart_type': chart_type,
                'data_info': {
                    'series_count': len(visualizer.valid_series),
                    'categories_count': len(visualizer.categories),
                    'series_names': list(visualizer.valid_series.keys())
                }
            })
        else:
            return jsonify({
                'success': False, 
                'message': message
            })
            
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"处理请求时发生错误: {str(e)}")
        print(f"错误详情: {error_trace}")
        
        return jsonify({
            'success': False,
            'message': f'服务器内部错误: {str(e)}'
        })

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'success',
        'message': 'Excel可视化API服务正常运行',
        'timestamp': pd.Timestamp.now().isoformat()
    })

@app.route('/api/chart-types', methods=['GET'])
def get_chart_types():
    """获取支持的图表类型"""
    chart_types = [
        {'value': '柱状图', 'label': '柱状图', 'description': '比较不同类别的数值'},
        {'value': '饼图', 'label': '饼图', 'description': '显示各部分占整体的比例'},
        {'value': '折线图', 'label': '折线图', 'description': '显示数据随时间的变化趋势'},
        {'value': '面积图', 'label': '面积图', 'description': '显示数据累积效果'},
        {'value': '散点图', 'label': '散点图', 'description': '显示两个变量之间的关系'},
        {'value': '气泡图', 'label': '气泡图', 'description': '三个变量的关系（x,y,大小）'},
        {'value': '热力图', 'label': '热力图', 'description': '用颜色强度显示数据矩阵'},
        {'value': '直方图', 'label': '直方图', 'description': '显示数据分布情况'},
        {'value': '箱线图', 'label': '箱线图', 'description': '显示数据统计特征'},
        {'value': '组合图', 'label': '组合图', 'description': '柱状图和折线图组合'},
        {'value': '条形图', 'label': '条形图', 'description': '水平条形图'}
    ]
    
    return jsonify({
        'success': True,
        'chart_types': chart_types
    })

# Vercel需要这个
app = app