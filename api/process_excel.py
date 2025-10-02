from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import io
import base64
import json
import traceback

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
        """简化的数据准备"""
        try:
            if self.df is None or self.df.empty:
                return False, "没有有效数据"
            
            # 使用前10行5列
            sample_df = self.df.iloc[:10, :5].copy()
            
            # 准备基础数据
            self.valid_series = {}
            self.categories = [f"项目{i+1}" for i in range(min(10, len(sample_df)))]
            
            # 为每个列创建数据
            for col in sample_df.columns[:3]:  # 最多3个系列
                try:
                    series_data = pd.to_numeric(sample_df[col], errors='coerce').fillna(0).tolist()
                    self.valid_series[str(col)] = series_data[:10]
                except:
                    continue
            
            # 如果没有数据，创建示例
            if not self.valid_series:
                self.categories = [f"类别{i+1}" for i in range(5)]
                self.valid_series = {
                    "系列1": [10, 20, 15, 25, 30],
                    "系列2": [5, 15, 10, 20, 25]
                }
            
            return True, f"准备 {len(self.valid_series)} 个数据系列"
            
        except Exception as e:
            return False, f"数据准备失败: {str(e)}"

    def generate_chart_data(self, chart_type):
        """生成图表数据（不生成图片）"""
        try:
            if not self.valid_series:
                return False, "没有有效数据", None
            
            # 返回结构化数据，让前端绘制
            chart_data = {
                'type': chart_type,
                'categories': self.categories,
                'series': self.valid_series,
                'series_names': list(self.valid_series.keys())
            }
            
            return True, f"{chart_type}数据生成成功", chart_data
                
        except Exception as e:
            return False, f"生成{chart_type}数据失败: {str(e)}", None

@app.route('/api/process-excel', methods=['POST'])
def process_excel():
    """处理Excel文件并返回图表数据"""
    try:
        print("收到图表生成请求")
        
        if 'excel_file' not in request.files:
            return jsonify({'success': False, 'message': '没有上传文件'})
        
        file = request.files['excel_file']
        chart_type = request.form.get('chart_type', '柱状图')
        
        if file.filename == '':
            return jsonify({'success': False, 'message': '文件名为空'})
        
        print(f"处理文件: {file.filename}, 图表类型: {chart_type}")
        
        # 初始化可视化器
        visualizer = ExcelVisualizer()
        
        # 读取文件
        file_content = file.read()
        success, message = visualizer.load_data(file_content, file.filename)
        
        if not success:
            return jsonify({'success': False, 'message': message})
        
        # 准备数据
        success, message = visualizer.prepare_data_simple()
        if not success:
            return jsonify({'success': False, 'message': message})
        
        print(f"数据准备完成: {message}")
        
        # 生成图表数据
        success, message, chart_data = visualizer.generate_chart_data(chart_type)
        
        if success:
            return jsonify({
                'success': True,
                'message': message,
                'chart_data': chart_data,
                'chart_type': chart_type
            })
        else:
            return jsonify({'success': False, 'message': message})
            
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"处理请求时发生错误: {str(e)}")
        
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
        {'value': '折线图', 'label': '折线图', 'description': '显示数据变化趋势'},
        {'value': '散点图', 'label': '散点图', 'description': '显示两个变量之间的关系'}
    ]
    
    return jsonify({
        'success': True,
        'chart_types': chart_types
    })

# Vercel需要这个
app = app
