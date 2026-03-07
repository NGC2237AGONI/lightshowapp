import sys
import os
import csv
import numpy as np
import traceback
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QComboBox, QFileDialog, 
                             QGroupBox, QSpinBox, QDoubleSpinBox, QTextEdit, QMessageBox, QSplitter,
                             QTabWidget, QListWidget, QListWidgetItem, QAbstractItemView)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from scipy.spatial import cKDTree

import drone_core
import drone_composition 
import safety_analyzer 

# =========================================================================
# 导出线程
# =========================================================================
class ExportWorker(QThread):
    finished = pyqtSignal(bool, str, str, dict) 

    def __init__(self, exporter, optimizer_traj, fbx_path, anim_idx, fps, scale, axis, output_path, safe_dist, max_vel, L, W, H, time_scale, loop_count):
        super().__init__()
        self.exporter = exporter
        self.optimizer_traj = optimizer_traj
        self.fbx_path = fbx_path
        self.anim_idx = anim_idx
        self.fps = fps
        self.scale = scale
        self.axis = axis
        self.output_path = output_path
        self.safe_dist = safe_dist
        self.max_vel = max_vel
        self.L = L; self.W = W; self.H = H
        self.time_scale = time_scale
        self.loop_count = loop_count 

    def run(self):
        try:
            temp_raw_path = "temp_raw_export.csv"
            success, msg = self.exporter.run_raw_export(
                self.fbx_path, self.anim_idx, self.fps, self.scale, self.axis, temp_raw_path
            )
            
            if not success:
                self.finished.emit(False, f"导出失败: {msg}", "", {})
                return

            trim_success, trim_msg = self.optimizer_traj.smart_trim_and_loop(
                temp_raw_path, self.output_path, self.loop_count
            )
            
            if not trim_success:
                self.finished.emit(False, f"裁剪失败: {trim_msg}", "", {})
                return

            opt_success, opt_msg, final_path, info = self.optimizer_traj.optimize_trajectory(
                self.output_path, self.safe_dist, self.max_vel, self.L, self.W, self.H, self.time_scale
            )
            
            if os.path.exists(temp_raw_path):
                os.remove(temp_raw_path)
            
            final_msg = f"{trim_msg}\n----------------\n{opt_msg}"
            self.finished.emit(opt_success, final_msg, final_path, info)
            
        except Exception as e:
            traceback.print_exc()
            self.finished.emit(False, f"流程异常: {str(e)}", "", {})

# =========================================================================
# 静态阵列计算线程
# =========================================================================
class OptimizationWorker(QThread):
    finished = pyqtSignal(bool, str, object, object)

    def __init__(self, optimizer, axis_mode, count, safe_dist):
        super().__init__()
        self.optimizer = optimizer
        self.axis_mode = axis_mode
        self.count = count
        self.safe_dist = safe_dist

    def run(self):
        try:
            success, msg, pts, cols = self.optimizer.run(self.axis_mode, self.count, self.safe_dist)
            self.finished.emit(success, msg, pts, cols)
        except Exception as e:
            self.finished.emit(False, f"线程错误: {str(e)}", [],[])

# =========================================================================
# 主界面逻辑
# =========================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("无人机编队设计系统")
        self.resize(1300, 850)
        
        self.extractor = drone_core.DataExtractor()
        self.optimizer = drone_core.FormationOptimizer()
        self.exporter = drone_core.AnimationExporter()
        self.optimizer_traj = drone_core.TrajectoryOptimizer()
        self.composer = drone_composition.CompositionManager() 
        
        self.current_fbx = ""
        self.animations = [] 
        self.playing = False
        self.anim_frames = {}
        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self.update_plot)
        self.current_frame = 0
        self.total_frames = 0
        self.current_csv = ""
        
        self.scatter = None
        self.current_static_pts = None 
        self.current_static_cols = None
        
        self.ax = None 
        self.worker = None 
        self.export_worker = None

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(380)
        
        self.tabs = QTabWidget()
        
        # --- Tab 1: 静态处理 ---
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        
        grp_file = QGroupBox("1. 文件加载")
        lay_file = QVBoxLayout()
        self.btn_load = QPushButton("选择 FBX 文件")
        self.btn_load.clicked.connect(self.load_file)
        self.lbl_file = QLabel("未选择文件")
        self.lbl_file.setWordWrap(True)
        lay_file.addWidget(self.btn_load)
        lay_file.addWidget(self.lbl_file)
        grp_file.setLayout(lay_file)
        
        grp_ext = QGroupBox("2. 提取设置")
        lay_ext = QVBoxLayout()
        lay_scale = QHBoxLayout()
        lay_scale.addWidget(QLabel("初始缩放:"))
        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setValue(2.0)
        lay_scale.addWidget(self.spin_scale)
        lay_ext.addLayout(lay_scale)
        self.btn_extract = QPushButton("提取全量数据 (PointWithColor)")
        self.btn_extract.clicked.connect(self.run_extract)
        lay_ext.addWidget(self.btn_extract)
        grp_ext.setLayout(lay_ext)
        
        grp_opt = QGroupBox("3. 编队优化 (静态)")
        lay_opt = QVBoxLayout()
        lay_count = QHBoxLayout()
        lay_count.addWidget(QLabel("目标数量:"))
        self.spin_count = QSpinBox()
        self.spin_count.setRange(10, 99999)
        self.spin_count.setValue(600)
        lay_count.addWidget(self.spin_count)
        lay_opt.addLayout(lay_count)
        lay_axis = QHBoxLayout()
        lay_axis.addWidget(QLabel("坐标修正:"))
        self.combo_axis = QComboBox()
        self.combo_axis.addItems(["Mode 0", "Mode 1 (Y/Z)", "Mode 2 (X/Z)", "Mode 3 (X/Y)", "Mode 4 (Rot90)"])
        self.combo_axis.setCurrentIndex(1)
        lay_axis.addWidget(self.combo_axis)
        lay_opt.addLayout(lay_axis)
        self.btn_optimize = QPushButton("生成静态编队 (预览)")
        self.btn_optimize.clicked.connect(self.run_optimize)
        lay_opt.addWidget(self.btn_optimize)
        grp_opt.setLayout(lay_opt)
        
        tab1_layout.addWidget(grp_file)
        tab1_layout.addWidget(grp_ext)
        tab1_layout.addWidget(grp_opt)
        tab1_layout.addStretch()
        
        # --- Tab 2: 动态设置 ---
        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)
        
        grp_config = QGroupBox("0. 表演/物理参数配置")
        lay_config = QVBoxLayout()
        lay_safe_config = QHBoxLayout()
        lay_safe_config.addWidget(QLabel("最小安全距离(m):"))
        self.spin_safe_config = QDoubleSpinBox(); self.spin_safe_config.setRange(0.1, 50.0); self.spin_safe_config.setValue(1.5)
        lay_safe_config.addWidget(self.spin_safe_config)
        lay_config.addLayout(lay_safe_config)
        
        lay_vel_config = QHBoxLayout()
        lay_vel_config.addWidget(QLabel("最大飞行速度(m/s):"))
        self.spin_max_vel = QDoubleSpinBox(); self.spin_max_vel.setRange(0.1, 100.0); self.spin_max_vel.setValue(10.0)
        lay_vel_config.addWidget(self.spin_max_vel)
        lay_config.addLayout(lay_vel_config)
        
        lay_config.addWidget(QLabel("场地边界 (L x W x H):"))
        h4 = QHBoxLayout()
        self.spin_L = QDoubleSpinBox(); self.spin_L.setRange(1, 9e6); self.spin_L.setValue(200); self.spin_L.setPrefix("L:")
        self.spin_W = QDoubleSpinBox(); self.spin_W.setRange(1, 9e6); self.spin_W.setValue(200); self.spin_W.setPrefix("W:")
        self.spin_H = QDoubleSpinBox(); self.spin_H.setRange(1, 9e6); self.spin_H.setValue(150); self.spin_H.setPrefix("H:")
        
        self.spin_L.valueChanged.connect(self.refresh_scene_if_needed)
        self.spin_W.valueChanged.connect(self.refresh_scene_if_needed)
        self.spin_H.valueChanged.connect(self.refresh_scene_if_needed)
        
        h4.addWidget(self.spin_L); h4.addWidget(self.spin_W); h4.addWidget(self.spin_H)
        lay_config.addLayout(h4)
        
        lay_time_scale = QHBoxLayout()
        lay_time_scale.addWidget(QLabel("手动时间缩放:"))
        self.spin_time_scale = QDoubleSpinBox(); self.spin_time_scale.setRange(0.0, 10.0); self.spin_time_scale.setValue(0.0)
        lay_time_scale.addWidget(self.spin_time_scale)
        grp_config.setLayout(lay_config)
        
        grp_anim = QGroupBox("4. 动画生成")
        lay_anim = QVBoxLayout()
        h_scan = QHBoxLayout()
        self.btn_scan = QPushButton("扫描动画"); self.btn_scan.clicked.connect(self.scan_animations)
        self.combo_anim = QComboBox(); 
        self.combo_anim.currentIndexChanged.connect(self.on_anim_selected)
        h_scan.addWidget(self.btn_scan); h_scan.addWidget(self.combo_anim)
        lay_anim.addLayout(h_scan)
        
        h_dur = QHBoxLayout()
        h_dur.addWidget(QLabel("原始时长:"))
        self.lbl_raw_dur = QLabel("- s") 
        h_dur.addWidget(self.lbl_raw_dur)
        h_dur.addStretch()
        lay_anim.addLayout(h_dur)
        
        h_loop = QHBoxLayout()
        h_loop.addWidget(QLabel("循环次数:"))
        self.spin_loop = QSpinBox()
        self.spin_loop.setRange(1, 100); self.spin_loop.setValue(1)
        h_loop.addWidget(self.spin_loop)
        lay_anim.addLayout(h_loop)

        self.btn_export = QPushButton(" 生成并自动优化轨迹")
        self.btn_export.clicked.connect(self.run_export)
        lay_anim.addWidget(self.btn_export)
        grp_anim.setLayout(lay_anim)
        
        grp_verify = QGroupBox("5. 验证与播放")
        lay_verify = QVBoxLayout()
        self.btn_analyze = QPushButton(" 安全性体检")
        self.btn_analyze.clicked.connect(self.run_safety_check)
        lay_verify.addWidget(self.btn_analyze)
        lay_play_ctrl = QHBoxLayout()
        self.btn_play = QPushButton(" 播放/暂停"); self.btn_play.clicked.connect(self.toggle_play)
        self.spin_pt = QSpinBox(); self.spin_pt.setValue(5); self.spin_pt.valueChanged.connect(self.update_point_size)
        lay_play_ctrl.addWidget(self.btn_play); lay_play_ctrl.addWidget(self.spin_pt)
        lay_verify.addLayout(lay_play_ctrl)
        grp_verify.setLayout(lay_verify)

        tab2_layout.addWidget(grp_config); tab2_layout.addWidget(grp_anim); tab2_layout.addWidget(grp_verify); tab2_layout.addStretch()

        # --- Tab 3: 编队合成 ---
        tab3 = QWidget()
        tab3_layout = QVBoxLayout(tab3)
        
        grp_comp_list = QGroupBox("1. 节目单编排")
        lay_comp_list = QVBoxLayout()
        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self.on_list_item_clicked)
        lay_comp_list.addWidget(self.list_widget)
        
        h_comp_btns = QHBoxLayout()
        self.btn_add_csv = QPushButton("➕ 添加 CSV")
        self.btn_add_csv.clicked.connect(self.add_comp_file)
        self.btn_remove_csv = QPushButton("➖ 移除")
        self.btn_remove_csv.clicked.connect(self.remove_comp_file)
        h_comp_btns.addWidget(self.btn_add_csv); h_comp_btns.addWidget(self.btn_remove_csv)
        lay_comp_list.addLayout(h_comp_btns)
        
        # 第一行：过渡时间
        h_trans = QHBoxLayout()
        h_trans.addWidget(QLabel("过渡(s):"))
        self.spin_trans_dur = QDoubleSpinBox(); self.spin_trans_dur.setValue(5.0)
        self.spin_trans_dur.valueChanged.connect(self.update_comp_params)
        h_trans.addWidget(self.spin_trans_dur)
        h_trans.addStretch()
        lay_comp_list.addLayout(h_trans)
        
        # 第二行：自转设置
        h_rot = QHBoxLayout()
        h_rot.addWidget(QLabel("自转(XYZ):"))
        self.spin_rot_x = QDoubleSpinBox(); self.spin_rot_x.setRange(-360, 360)
        self.spin_rot_y = QDoubleSpinBox(); self.spin_rot_y.setRange(-360, 360)
        self.spin_rot_z = QDoubleSpinBox(); self.spin_rot_z.setRange(-360, 360)
        self.spin_rot_x.valueChanged.connect(self.update_comp_params)
        self.spin_rot_y.valueChanged.connect(self.update_comp_params)
        self.spin_rot_z.valueChanged.connect(self.update_comp_params)
        h_rot.addWidget(self.spin_rot_x); h_rot.addWidget(self.spin_rot_y); h_rot.addWidget(self.spin_rot_z)
        lay_comp_list.addLayout(h_rot)
        
        # 【新增】第三行：位移设置
        h_pos = QHBoxLayout()
        h_pos.addWidget(QLabel("位移(XYZ):"))
        self.spin_pos_x = QDoubleSpinBox(); self.spin_pos_x.setRange(-9999, 9999)
        self.spin_pos_y = QDoubleSpinBox(); self.spin_pos_y.setRange(-9999, 9999)
        self.spin_pos_z = QDoubleSpinBox(); self.spin_pos_z.setRange(-9999, 9999)
        self.spin_pos_x.valueChanged.connect(self.update_comp_params)
        self.spin_pos_y.valueChanged.connect(self.update_comp_params)
        self.spin_pos_z.valueChanged.connect(self.update_comp_params)
        h_pos.addWidget(self.spin_pos_x); h_pos.addWidget(self.spin_pos_y); h_pos.addWidget(self.spin_pos_z)
        lay_comp_list.addLayout(h_pos)
        
        grp_comp_list.setLayout(lay_comp_list)
        
        grp_comp_act = QGroupBox("2. 合成")
        lay_comp_act = QVBoxLayout()
        self.btn_merge = QPushButton(" 开始合成")
        self.btn_merge.setStyleSheet("background-color: #1976D2; color: white; font-weight: bold; padding: 10px;")
        self.btn_merge.clicked.connect(self.run_merge)
        lay_comp_act.addWidget(self.btn_merge)
        grp_comp_act.setLayout(lay_comp_act)
        
        tab3_layout.addWidget(grp_comp_list); tab3_layout.addWidget(grp_comp_act); tab3_layout.addStretch()

        self.tabs.addTab(tab1, "1. 静态预处理")
        self.tabs.addTab(tab2, "2. 动态与参数")
        self.tabs.addTab(tab3, "3. 编队合成")
        left_layout.addWidget(self.tabs)
        
        # Right Panel
        right_panel = QWidget(); right_layout = QVBoxLayout(right_panel)
        self.fig = Figure(figsize=(5, 5), dpi=100, facecolor='black')
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d') 
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        self.setup_scene(mode='auto')
        
        self.log_box = QTextEdit(); self.log_box.setMaximumHeight(150); self.log_box.setReadOnly(True); self.log_box.setStyleSheet("background-color: #222; color: #0f0;")
        right_layout.addWidget(self.toolbar); right_layout.addWidget(self.canvas); right_layout.addWidget(self.log_box)
        splitter = QSplitter(Qt.Orientation.Horizontal); splitter.addWidget(left_panel); splitter.addWidget(right_panel); splitter.setStretchFactor(0, 3); splitter.setStretchFactor(1, 7); layout.addWidget(splitter)
        
        self.tabs.currentChanged.connect(self.on_tab_changed)

    def log(self, msg):
        self.log_box.append(msg)
        QApplication.processEvents()

    # (其他函数保持不变...)
    def reset_all_state(self):
        self.playing = False
        self.anim_timer.stop()
        self.anim_frames = {}
        self.current_frame = 0
        self.current_static_pts = None
        self.current_static_cols = None
        if self.scatter: self.scatter.remove(); self.scatter = None
        self.ax.clear(); self.ax.set_facecolor('black'); self.ax.axis('off')
        self.setup_scene(mode='auto')
        self.tabs.setCurrentIndex(0)

    def run_optimize(self):
        self.btn_optimize.setEnabled(False)
        self.btn_optimize.setText("计算中...请稍候")
        self.log(" 后台开始计算编队...")
        
        self.worker = OptimizationWorker(
            self.optimizer,
            self.combo_axis.currentIndex(), 
            self.spin_count.value(), 
            self.spin_safe_config.value()
        )
        self.worker.finished.connect(self.on_optimize_finished)
        self.worker.start()

    def on_optimize_finished(self, success, msg, pts, cols):
        self.btn_optimize.setEnabled(True)
        self.btn_optimize.setText("生成静态编队 (预览)")
        self.log(msg)
        if success:
            self.current_static_pts = pts
            self.current_static_cols = cols
            self.plot_static_memory(pts, cols, force_mode='auto')
            self.recommend_boundaries_smart(pts, self.spin_count.value(), self.spin_safe_config.value())

    def on_tab_changed(self, index):
        if index == 0:
            self.setup_scene(mode='auto')
            if self.current_static_pts is not None: self.plot_static_memory(self.current_static_pts, self.current_static_cols, force_mode='auto')
        else: self.setup_scene(mode='box')

    def setup_scene(self, mode='auto'):
        if self.ax is None: return 
        self.ax.clear(); self.ax.set_facecolor('black'); self.ax.axis('off')
        if mode == 'box':
            try:
                L, W, H = self.spin_L.value(), self.spin_W.value(), self.spin_H.value()
                axis_len = max(L, W, H) * 0.1
                self.ax.plot([0, axis_len], [0, 0], [0, 0], color='red')
                self.ax.plot([0, 0], [0, axis_len], [0, 0], color='green')
                self.ax.plot([0, 0], [0, 0], [0, axis_len], color='blue')
                min_x, max_x = -L/2, L/2; min_y, max_y = -W/2, W/2; min_z, max_z = 0, H
                corners = np.array([[min_x, min_y, min_z], [max_x, min_y, min_z], [max_x, max_y, min_z], [min_x, max_y, min_z], [min_x, min_y, max_z], [max_x, min_y, max_z], [max_x, max_y, max_z], [min_x, max_y, max_z]])
                edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
                for s, e in edges: self.ax.plot3D([corners[s][0], corners[e][0]], [corners[s][1], corners[e][1]], [corners[s][2], corners[e][2]], color='gray', linestyle='--', alpha=0.5)
                self.ax.set_xlim(min_x, max_x); self.ax.set_ylim(min_y, max_y); self.ax.set_zlim(min_z, max_z); self.ax.set_box_aspect((L, W, H))
            except: pass
        self.scatter = self.ax.scatter([], [], [], c=[], s=self.spin_pt.value())
        self.canvas.draw()

    def refresh_scene_if_needed(self):
        if self.tabs.currentIndex() > 0: self.setup_scene(mode='box')

    def load_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "选择 FBX", "", "FBX Files (*.fbx)")
        if fname: self.reset_all_state(); self.current_fbx = fname; self.lbl_file.setText(os.path.basename(fname)); self.log(f"已加载: {fname}")

    def run_extract(self):
        if not self.current_fbx: return self.log(" 请先选择文件")
        self.log(" 提取中...")
        try:
            success, msg = self.extractor.run(self.current_fbx, self.spin_scale.value())
            self.log(msg)
        except Exception as e: self.log(f" 提取崩溃: {e}"); print(traceback.format_exc())

    def recommend_boundaries_smart(self, pts, count, safe_dist):
        try:
            min_x, max_x = pts[:,0].min(), pts[:,0].max(); min_y, max_y = pts[:,1].min(), pts[:,1].max(); min_z, max_z = pts[:,2].min(), pts[:,2].max()
            curr_L = max_x - min_x; curr_W = max_y - min_y; curr_H = max_z - min_z
            if len(pts) > 1:
                tree = cKDTree(pts); dists, _ = tree.query(pts, k=2); avg_neighbor_dist = np.mean(dists[:, 1])
            else: avg_neighbor_dist = safe_dist
            if avg_neighbor_dist < 0.001: avg_neighbor_dist = 0.001
            ratio = 1.0
            if avg_neighbor_dist < safe_dist: ratio = safe_dist / avg_neighbor_dist
            sug_L = curr_L * ratio * 1.1; sug_W = curr_W * ratio * 1.1; sug_H = curr_H * ratio * 1.1
            self.log("-" * 35)
            self.log(f" 智能场地建议: 目标 {safe_dist}m, 当前 {avg_neighbor_dist:.2f}m")
            if ratio > 1.0: self.log(f"    建议放大 {ratio:.2f} 倍 -> {sug_L:.1f}x{sug_W:.1f}x{sug_H:.1f}")
            else: self.log(f"    当前尺寸合适")
            self.log("-" * 35)
        except Exception as e: print(e)

    def scan_animations(self):
        if not self.current_fbx: return
        self.log("扫描中..."); 
        try:
            self.animations = self.exporter.get_animations(self.current_fbx)
            self.combo_anim.blockSignals(True)
            self.combo_anim.clear()
            for anim in self.animations: self.combo_anim.addItem(anim['name'])
            self.combo_anim.blockSignals(False)
            if self.animations: 
                self.combo_anim.setCurrentIndex(0)
                self.on_anim_selected()
        except Exception as e: self.log(f"扫描崩溃: {e}")

    def on_anim_selected(self):
        idx = self.combo_anim.currentIndex()
        if idx < 0: return
        raw_dur = self.animations[idx]['duration']
        self.lbl_raw_dur.setText(f"{raw_dur:.2f} s")

    def run_export(self):
        idx = self.combo_anim.currentIndex()
        if idx < 0: return self.log("请先选择动画")
        default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CSV")
        if not os.path.exists(default_dir): os.makedirs(default_dir)
        default_name = os.path.join(default_dir, "drone_path.csv")
        save_path, _ = QFileDialog.getSaveFileName(self, "保存轨迹", default_name, "CSV Files (*.csv)")
        if not save_path: return

        self.btn_export.setEnabled(False); self.btn_export.setText(" 正在导出...")
        self.log(f"开始导出: {os.path.basename(save_path)}")
        L, W, H = self.spin_L.value(), self.spin_W.value(), self.spin_H.value()
        
        self.export_worker = ExportWorker(
            self.exporter, self.optimizer_traj, self.current_fbx, self.animations[idx]['index'], 
            20, 2.0, self.combo_axis.currentIndex(), save_path,
            self.spin_safe_config.value(), self.spin_max_vel.value(), L, W, H,
            self.spin_time_scale.value() if self.spin_time_scale.value() > 0 else None,
            self.spin_loop.value()
        )
        self.export_worker.finished.connect(self.on_export_finished); self.export_worker.start()

    def on_export_finished(self, success, msg, path, info):
        self.btn_export.setEnabled(True); self.btn_export.setText(" 生成并自动优化轨迹")
        self.log(msg)
        if success:
            self.current_csv = path; self.load_csv_for_play()
            QMessageBox.information(self, "成功", f"导出成功！\n文件: {path}")

    def run_safety_check(self):
        if not self.current_csv: return self.log("请先生成轨迹")
        self.log(" 安全分析..."); safety_analyzer.analyze_safety(csv_file=self.current_csv, safe_distance=self.spin_safe_config.value(), max_velocity=self.spin_max_vel.value()); self.log("✅ 图表已生成")

    def plot_static_memory(self, pts, cols, force_mode=None):
        try:
            pts_np = np.array(pts); cols_np = np.array(cols)
            if len(pts_np) == 0: return
            if force_mode: self.setup_scene(mode=force_mode)
            visual_pts = pts_np.copy()
            if force_mode == 'box' or (force_mode is None and self.tabs.currentIndex() == 1): visual_pts[:, 2] += self.spin_H.value() / 2.0
            self.scatter._offsets3d = (visual_pts[:,0], visual_pts[:,1], visual_pts[:,2])
            self.scatter.set_color(cols_np)
            self.canvas.draw()
        except Exception as e: self.log(f" 绘图出错: {e}")

    def load_csv_for_play(self, csv_file=None):
        if csv_file: self.current_csv = csv_file
        self.log(f"加载播放: {self.current_csv}"); self.anim_frames = {}; all_pos = []
        try:
            with open(self.current_csv, 'r') as f:
                reader = csv.reader(f); next(reader)
                for row in reader:
                    f_idx = int(row[0]); x,y,z = float(row[4]), float(row[5]), float(row[6]); r,g,b = float(row[7])/255, float(row[8])/255, float(row[9])/255
                    if f_idx not in self.anim_frames: self.anim_frames[f_idx] = []
                    self.anim_frames[f_idx].append([x,y,z,r,g,b])
            self.total_frames = len(self.anim_frames); self.current_frame = 0; 
            if self.tabs.currentIndex() == 0: self.tabs.setCurrentIndex(1)
            self.setup_scene(mode='box')
            self.playing = True; self.anim_timer.start(50)
        except Exception as e: self.log(f"加载失败: {e}")

    def toggle_play(self): self.playing = not self.playing; self.anim_timer.start() if self.playing else self.anim_timer.stop()
    def update_point_size(self): 
        if self.scatter: self.scatter.set_sizes([self.spin_pt.value()]); self.canvas.draw()
    def update_plot(self):
        if not self.anim_frames: return
        try:
            data = np.array(self.anim_frames.get(self.current_frame, []))
            if len(data) > 0:
                self.scatter._offsets3d = (data[:,0], data[:,1], data[:,2])
                self.scatter.set_color(data[:,3:])
                self.scatter.set_sizes([self.spin_pt.value()])
                self.current_frame = (self.current_frame + 1) % self.total_frames
                self.canvas.draw()
        except: pass

    # --- Tab 3: List handling ---
    def add_comp_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "添加 CSV", "", "CSV Files (*.csv)")
        if fname:
            ok, msg = self.composer.add_file(fname)
            if ok: self.log(msg); self.refresh_list()
            else: self.log(f" {msg}")
    def remove_comp_file(self):
        row = self.list_widget.currentRow()
        if row >= 0: self.composer.remove_file(row); self.refresh_list()
    
    def on_list_item_clicked(self, item):
        row = self.list_widget.row(item)
        if row >= 0 and row < len(self.composer.playlist):
            data = self.composer.playlist[row]
            # 阻断信号
            self.spin_trans_dur.blockSignals(True)
            self.spin_rot_x.blockSignals(True)
            self.spin_rot_y.blockSignals(True)
            self.spin_rot_z.blockSignals(True)
            self.spin_pos_x.blockSignals(True)
            self.spin_pos_y.blockSignals(True)
            self.spin_pos_z.blockSignals(True)
            
            self.spin_trans_dur.setValue(data.get('transition_dur', 5.0))
            rot = data.get('rotation', [0,0,0])
            self.spin_rot_x.setValue(rot[0])
            self.spin_rot_y.setValue(rot[1])
            self.spin_rot_z.setValue(rot[2])
            pos = data.get('position', [0,0,0])
            self.spin_pos_x.setValue(pos[0])
            self.spin_pos_y.setValue(pos[1])
            self.spin_pos_z.setValue(pos[2])
            
            # 恢复信号
            self.spin_trans_dur.blockSignals(False)
            self.spin_rot_x.blockSignals(False)
            self.spin_rot_y.blockSignals(False)
            self.spin_rot_z.blockSignals(False)
            self.spin_pos_x.blockSignals(False)
            self.spin_pos_y.blockSignals(False)
            self.spin_pos_z.blockSignals(False)

    def update_comp_params(self):
        row = self.list_widget.currentRow()
        if row >= 0:
            self.composer.set_transition_duration(row, self.spin_trans_dur.value())
            self.composer.set_rotation(row, self.spin_rot_x.value(), self.spin_rot_y.value(), self.spin_rot_z.value())
            self.composer.set_position(row, self.spin_pos_x.value(), self.spin_pos_y.value(), self.spin_pos_z.value())
            self.refresh_list()
            self.list_widget.setCurrentRow(row)

    def refresh_list(self):
        self.list_widget.clear()
        for i, item in enumerate(self.composer.playlist):
            name = os.path.basename(item['file'])
            dur = item['transition_dur']
            rot = item.get('rotation', [0,0,0])
            pos = item.get('position', [0,0,0])
            # 显示更详细的信息
            txt = f"[{i+1}] {name}\n    Rot:{rot} | Pos:{pos}"
            if i < len(self.composer.playlist) - 1:
                txt += f"\n    ( ↘ 过渡: {dur}s ↘ )"
            else:
                txt += "\n    (  结束 )"
            self.list_widget.addItem(txt)

    def run_merge(self):
        if len(self.composer.playlist) < 2: return self.log(" 至少需要两个文件才能合成")
        save_path, _ = QFileDialog.getSaveFileName(self, "保存合成文件", "Full_Show.csv", "CSV Files (*.csv)")
        if not save_path: return
        
        # 获取场地边界传递给后台
        L, W, H = self.spin_L.value(), self.spin_W.value(), self.spin_H.value()
        
        self.log(" 开始合成 ..."); QApplication.processEvents()
        ok, msg = self.composer.merge_shows(save_path, self.spin_safe_config.value(), (L, W, H))
        self.log(msg)
        if ok: self.current_csv = save_path; self.load_csv_for_play()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())