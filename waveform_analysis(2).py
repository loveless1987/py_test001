import sys
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd  # 添加pandas导入
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import tensorflow as tf
from scipy.optimize import differential_evolution
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                            QComboBox, QGroupBox, QRadioButton, QButtonGroup,
                            QTabWidget, QMessageBox, QProgressBar, QFormLayout,
                            QLineEdit, QCheckBox, QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

# 设置matplotlib中文字体（适配macOS）
plt.rcParams["font.family"] = ["Heiti TC", "Arial Unicode MS", "STHeiti", "DejaVu Sans"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class AnalysisThread(QThread):
    """波形分析的后台线程，避免UI卡顿"""
    progress_updated = pyqtSignal(int)
    analysis_completed = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, model_path, data_path, sample_type):
        super().__init__()
        self.model_path = model_path
        self.data_path = data_path
        self.sample_type = sample_type
        
    def run(self):
        try:
            # 加载模型
            self.progress_updated.emit(10)
            model = tf.keras.models.load_model(self.model_path)
            
            # 加载数据
            self.progress_updated.emit(20)
            data = sio.loadmat(self.data_path)
            valid_keys = [k for k in data.keys() if not k.startswith('__')]
            if len(valid_keys) != 1:
                raise ValueError(f"期望找到1个变量，却发现{len(valid_keys)}个: {valid_keys}")
            var_name = valid_keys[0]
            data = data[var_name]
            
            # 数据预处理
            self.progress_updated.emit(30)
            data_passage = np.expand_dims(data, axis=-1)
            
            # 模型预测
            self.progress_updated.emit(50)
            predictions = model.predict(data_passage)
            
            # 门槛化
            self.progress_updated.emit(60)
            valid_flags = np.where(predictions > 0.9, 1, 0)
            
            # 波形恢复
            self.progress_updated.emit(70)
            restored_wave = self._restore_waveform(data, valid_flags)
            
            # 准备结果
            self.progress_updated.emit(90)
            result = {
                'original_wave': data,
                'restored_wave': restored_wave,
                'sample_type': self.sample_type,
                'model_path': self.model_path,
                'data_path': self.data_path
            }
            
            self.progress_updated.emit(100)
            self.analysis_completed.emit(result)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def _restore_waveform(self, data, valid_flags):
        """波形恢复的核心算法"""
        # 定义模型函数
        def model_function(x, params):
            a, b, c, d = params
            return a * np.exp(-0.005 * x / b) + c * np.cos(0.05 * np.pi * x + d)
        
        # 定义损失函数
        def loss(params):
            return np.sum((model_function(x, params) - y) ** 2)
        
        # 参数范围
        bounds = [
            (-2, 2),    # a
            (0.1, 5),   # b
            (-2, 2),    # c
            (-np.pi, np.pi)  # d
        ]
        
        num_samples = len(data)
        wave_length = data.shape[1] if len(data.shape) > 1 else len(data)
        restored_wave = np.zeros((num_samples, wave_length))
        x_full = np.arange(1, wave_length + 1)
        
        for i in range(num_samples):
            if i % 10 == 0:
                # 更新进度
                progress = 70 + int(20 * i / num_samples)
                self.progress_updated.emit(progress)
                
            valid_array = valid_flags[i].flatten()
            y_data = data[i]
            
            # 去除无效点
            mask = valid_array != 0
            x = x_full[mask]
            y = y_data[mask]
            
            # 优化参数
            result = differential_evolution(loss, bounds)
            
            # 生成恢复波形
            x_fit = np.linspace(1, wave_length, wave_length)
            y_fit = model_function(x_fit, result.x)
            restored_wave[i] = y_fit
        
        return restored_wave

class WaveformCanvas(FigureCanvas):
    """波形显示画布"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.axes = self.fig.add_subplot(111)
        self.clear_plot()
        
    def clear_plot(self):
        self.axes.clear()
        self.axes.set_title("波形显示")
        self.axes.set_xlabel("采样点")
        self.axes.set_ylabel("幅值")
        self.axes.grid(True)
        self.draw()
        
    def plot_waveform(self, data, title="波形", color='blue', label=None):
        self.axes.clear()
        self.axes.set_title(title)
        self.axes.set_xlabel("采样点")
        self.axes.set_ylabel("幅值")
        
        if len(data.shape) > 1:
            # 多条波形
            for i in range(min(5, data.shape[0])):  # 最多显示5条
                self.axes.plot(data[i], label=f"波形 {i+1}" if label is None else label)
        else:
            # 单条波形
            self.axes.plot(data, label=label, color=color)
            
        self.axes.legend()
        self.axes.grid(True)
        self.draw()
        
    def plot_comparison(self, original, restored, title="波形对比"):
        self.axes.clear()
        self.axes.set_title(title)
        self.axes.set_xlabel("采样点")
        self.axes.set_ylabel("幅值")
        
        # 只显示第一条波形进行对比
        if len(original.shape) > 1:
            original = original[0]
        if len(restored.shape) > 1:
            restored = restored[0]
            
        self.axes.plot(original, label="原始波形", color='blue')
        self.axes.plot(restored, label="恢复波形", color='red')
        self.axes.legend()
        self.axes.grid(True)
        self.draw()

class WaveformAnalysisApp(QMainWindow):
    """波形分析应用主窗口"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        # 设置窗口标题和大小
        self.setWindowTitle("波形分析系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中央部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 创建分割器，允许用户调整左右面板大小
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        splitter.addWidget(control_panel)
        
        # 模型选择区域
        model_group = QGroupBox("模型选择")
        model_layout = QVBoxLayout()
        model_group.setLayout(model_layout)
        
        self.model_type = QComboBox()
        self.model_type.addItems(["CT饱和", "励磁涌流"])
        model_layout.addWidget(QLabel("选择模型类型:"))
        model_layout.addWidget(self.model_type)
        
        self.model_path = QLineEdit()
        model_layout.addWidget(QLabel("模型路径:"))
        
        model_path_layout = QHBoxLayout()
        model_path_layout.addWidget(self.model_path)
        browse_model_btn = QPushButton("浏览...")
        browse_model_btn.clicked.connect(self.browse_model)
        model_path_layout.addWidget(browse_model_btn)
        model_layout.addLayout(model_path_layout)  # 修改: 使用addLayout()
        
        control_layout.addWidget(model_group)
        
        # 数据选择区域
        data_group = QGroupBox("数据选择")
        data_layout = QVBoxLayout()
        data_group.setLayout(data_layout)
        
        self.data_path = QLineEdit()
        data_layout.addWidget(QLabel("数据文件:"))
        
        data_path_layout = QHBoxLayout()
        data_path_layout.addWidget(self.data_path)
        browse_data_btn = QPushButton("浏览...")
        browse_data_btn.clicked.connect(self.browse_data)
        data_path_layout.addWidget(browse_data_btn)
        data_layout.addLayout(data_path_layout)  # 修改: 使用addLayout()
        
        # 样品种类勾选
        sample_type_group = QGroupBox("样品种类")
        sample_type_layout = QVBoxLayout()
        sample_type_group.setLayout(sample_type_layout)
        
        self.sample_types = {}
        for type_name in ["CT饱和", "励磁涌流"]:
            checkbox = QCheckBox(type_name)
            if type_name == "CT饱和":
                font = checkbox.font()
                font.setBold(True)
                checkbox.setFont(font)
                checkbox.setStyleSheet("color: red;")  # 设置红色字体
            sample_type_layout.addWidget(checkbox)
            self.sample_types[type_name] = checkbox
            
        self.sample_types["CT饱和"].setChecked(True)  # 默认选中CT饱和
        data_layout.addWidget(sample_type_group)
        
        control_layout.addWidget(data_group)
        
        # 结果保存区域
        save_group = QGroupBox("结果保存")
        save_layout = QVBoxLayout()
        save_group.setLayout(save_layout)
        
        self.save_path = QLineEdit()
        save_layout.addWidget(QLabel("保存路径:"))
        
        save_path_layout = QHBoxLayout()
        save_path_layout.addWidget(self.save_path)
        browse_save_btn = QPushButton("浏览...")
        browse_save_btn.clicked.connect(self.browse_save)
        save_path_layout.addWidget(browse_save_btn)
        save_layout.addLayout(save_path_layout)  # 修改: 使用addLayout()
        
        self.filename_prefix = QLineEdit("result_")
        save_layout.addWidget(QLabel("文件名前缀:"))
        save_layout.addWidget(self.filename_prefix)
        
        control_layout.addWidget(save_group)
        
        # 分析按钮
        self.analyze_btn = QPushButton("开始分析")
        self.analyze_btn.setMinimumHeight(40)
        self.analyze_btn.clicked.connect(self.start_analysis)
        control_layout.addWidget(self.analyze_btn)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        control_layout.addWidget(self.progress_bar)
        
        # 右侧显示面板
        display_panel = QTabWidget()
        splitter.addWidget(display_panel)
        
        # 原始波形显示
        self.original_wave_tab = QWidget()
        original_layout = QVBoxLayout(self.original_wave_tab)
        self.original_canvas = WaveformCanvas(self.original_wave_tab, width=6, height=5)
        original_layout.addWidget(self.original_canvas)
        display_panel.addTab(self.original_wave_tab, "原始波形")
        
        # 恢复波形显示
        self.restored_wave_tab = QWidget()
        restored_layout = QVBoxLayout(self.restored_wave_tab)
        self.restored_canvas = WaveformCanvas(self.restored_wave_tab, width=6, height=5)
        restored_layout.addWidget(self.restored_canvas)
        display_panel.addTab(self.restored_wave_tab, "恢复波形")
        
        # 对比显示
        self.comparison_tab = QWidget()
        comparison_layout = QVBoxLayout(self.comparison_tab)
        self.comparison_canvas = WaveformCanvas(self.comparison_tab, width=6, height=5)
        comparison_layout.addWidget(self.comparison_canvas)
        display_panel.addTab(self.comparison_tab, "波形对比")
        
        # 分析结果信息
        self.result_info_tab = QWidget()
        result_layout = QVBoxLayout(self.result_info_tab)
        self.result_text = QLabel("分析结果将显示在这里")
        self.result_text.setWordWrap(True)
        self.result_text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        result_layout.addWidget(self.result_text)
        display_panel.addTab(self.result_info_tab, "分析结果")
        
        # 设置分割器初始大小
        splitter.setSizes([400, 800])
        
        # 初始化默认路径（使用用户提供的测试数据）
        self.model_path.setText("ct_model.keras")
        self.data_path.setText("sample_ct_test.mat")
        self.save_path.setText(os.path.expanduser("~/Desktop/waveform_results"))
        
        # 状态栏
        self.statusBar().showMessage("就绪")
        
    def browse_model(self):
        """浏览并选择模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "模型文件 (*.h5 *.keras);;所有文件 (*)"
        )
        if file_path:
            self.model_path.setText(file_path)
            # 根据文件扩展名自动选择模型类型
            model_name = os.path.basename(file_path).lower()
            if "ct" in model_name:
                self.model_type.setCurrentText("CT饱和")
            elif "magnetic" in model_name:
                self.model_type.setCurrentText("励磁涌流")
                
    def browse_data(self):
        """浏览并选择数据文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择数据文件", "", "MAT文件 (*.mat);;所有文件 (*)"
        )
        if file_path:
            self.data_path.setText(file_path)
            
    def browse_save(self):
        """浏览并选择保存路径"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择保存目录", os.path.expanduser("~/Desktop")
        )
        if dir_path:
            self.save_path.setText(dir_path)
            
    def start_analysis(self):
        """开始波形分析"""
        # 检查输入
        model_path = self.model_path.text()
        data_path = self.data_path.text()
        
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "警告", f"模型文件不存在: {model_path}")
            return
            
        if not os.path.exists(data_path):
            QMessageBox.warning(self, "警告", f"数据文件不存在: {data_path}")
            return
            
        # 获取选中的样品种类
        selected_types = [name for name, checkbox in self.sample_types.items() if checkbox.isChecked()]
        if not selected_types:
            QMessageBox.warning(self, "警告", "请至少选择一种样品种类")
            return
            
        sample_type = selected_types[0]  # 只取第一个选中的类型
        
        # 禁用按钮，显示进度
        self.analyze_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.statusBar().showMessage("正在分析...")
        
        # 清空之前的结果
        self.original_canvas.clear_plot()
        self.restored_canvas.clear_plot()
        self.comparison_canvas.clear_plot()
        self.result_text.setText("分析中...")
        
        # 创建并启动分析线程
        self.analysis_thread = AnalysisThread(model_path, data_path, sample_type)
        self.analysis_thread.progress_updated.connect(self.update_progress)
        self.analysis_thread.analysis_completed.connect(self.on_analysis_completed)
        self.analysis_thread.error_occurred.connect(self.on_analysis_error)
        self.analysis_thread.start()
        
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
        
    def on_analysis_completed(self, result):
        """分析完成后的回调"""
        # 恢复按钮状态
        self.analyze_btn.setEnabled(True)
        self.statusBar().showMessage("分析完成")
        
        # 显示波形
        self.original_canvas.plot_waveform(
            result['original_wave'], 
            f"原始波形 - {result['sample_type']}"
        )
        
        self.restored_canvas.plot_waveform(
            result['restored_wave'], 
            f"恢复波形 - {result['sample_type']}",
            color='red'
        )
        
        self.comparison_canvas.plot_comparison(
            result['original_wave'], 
            result['restored_wave'],
            f"波形对比 - {result['sample_type']}"
        )
        
        # 显示结果信息
        info = (f"分析结果\n"
                f"样品种类: {result['sample_type']}\n"
                f"模型: {os.path.basename(result['model_path'])}\n"
                f"数据: {os.path.basename(result['data_path'])}\n"
                f"波形数量: {len(result['original_wave'])}\n\n"
                f"波形恢复完成。点击保存按钮将结果保存到文件。")
        self.result_text.setText(info)
        
        # 保存结果
        self.save_results(result)
        
    def on_analysis_error(self, error_msg):
        """分析出错时的回调"""
        self.analyze_btn.setEnabled(True)
        self.statusBar().showMessage("分析出错")
        QMessageBox.critical(self, "错误", f"分析过程中出错:\n{error_msg}")
        self.result_text.setText(f"分析出错: {error_msg}")
        
    def save_results(self, result):
        """保存分析结果"""
        try:
            # 创建保存目录（如果不存在）
            save_dir = self.save_path.text()
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            # 生成保存文件名
            prefix = self.filename_prefix.text()
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}{timestamp}.mat"
            save_path = os.path.join(save_dir, filename)
            
            # 保存结果
            sio.savemat(save_path, {
                'original_wave': result['original_wave'],
                'restored_wave': result['restored_wave'],
                'sample_type': result['sample_type'],
                'model_path': result['model_path'],
                'data_path': result['data_path']
            })
            
            # 更新结果信息
            info = self.result_text.text()
            info += f"\n\n结果已保存至:\n{save_path}"
            self.result_text.setText(info)
            
        except Exception as e:
            QMessageBox.warning(self, "保存失败", f"无法保存结果:\n{str(e)}")

if __name__ == "__main__":
    # 确保中文显示正常
    import matplotlib
    matplotlib.use('Qt5Agg')
    
    app = QApplication(sys.argv)
    window = WaveformAnalysisApp()
    window.show()
    sys.exit(app.exec_())
