"""Runtime UI translation for the MEA Pipeline desktop application.

The existing GUI was built with English literals rather than Qt Designer
translation catalogs.  This module keeps those literals as stable source text
and translates visible Qt controls at runtime, including dialogs created after
the language changes.
"""

from __future__ import annotations

import re
from typing import Iterable

from PySide6.QtCore import QEvent, QObject, QSettings, QTimer, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QAbstractButton,
    QComboBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QTabWidget,
    QTableWidget,
    QTextEdit,
    QWidget,
)


LANGUAGES = {
    "en": "English",
    "zh_CN": "汉语",
    "ja_JP": "日本語",
}


_ZH = {
    "File": "文件", "Tools": "工具", "Language": "语言", "English": "英语",
    "Chinese": "汉语", "Japanese": "日语", "Exit": "退出", "Open Data Files": "打开数据文件",
    "Save File": "保存文件", "Clear Loaded Data": "清除已加载数据", "Channel Map": "电极图",
    "Sorting": "分选", "Analysis": "分析", "Custom Analysis": "自定义分析",
    "Channel Forecast Response": "通道响应预测", "Stimulus Generation": "刺激生成",
    "Closed Loop Control": "闭环控制", "Agent Custom Code": "Agent 自定义代码",
    "MEA Pipeline Studio": "MEA Pipeline 工作台", "MEA Pipeline\nStudio": "MEA Pipeline\n工作台",
    "Data Library": "数据区", "Loaded file database": "已加载文件数据库",
    "No data files loaded": "尚未加载数据文件", "Idle": "空闲", "Raw Data Raster": "原始数据 Raster",
    "Activity log": "运行日志", "Open": "打开", "Close": "关闭", "Cancel": "取消", "OK": "确定",
    "Yes": "是", "No": "否", "Apply": "应用", "Reset": "重置", "Reset View": "重置视图",
    "Save": "保存", "Save Plot": "保存图像", "Remove": "删除", "Add": "添加", "New": "新建",
    "Browse": "浏览", "Select": "选择", "Load": "加载", "Run": "运行", "Stop": "停止",
    "Start": "开始", "Pause": "暂停", "Play": "播放", "Preview": "预览", "Generate": "生成",
    "Refresh": "刷新", "Up": "上移", "Down": "下移", "Undo": "撤销", "Ready": "就绪",
    "Running": "运行中", "Stopped": "已停止", "Settings...": "设置...", "Advanced": "高级设置",
    "Show advanced": "显示高级设置", "Display settings...": "显示设置...", "View settings...": "视图设置...",
    "File": "文件", "Files": "文件", "Folder": "文件夹", "Name": "名称", "Type": "类型",
    "Kind": "类型", "Label": "标签", "Source": "来源", "Origin": "来源", "Condition": "条件",
    "Channels": "通道", "Channel": "通道", "Electrode": "电极", "Electrodes": "电极",
    "Spikes": "Spike 数", "Waveforms": "波形", "Samples": "样本", "Features": "特征",
    "Time": "时间", "Count": "数量", "Mode": "模式", "Group": "组", "Protocol": "Protocol",
    "Block": "Block", "Blocks": "Blocks", "Phase": "阶段", "Threshold": "阈值", "Window": "窗口",
    "Response": "响应", "Pre": "刺激前", "After": "刺激后", "Start s": "开始时间 (s)",
    "Stop s": "结束时间 (s)", "Latency ms": "延迟 (ms)", "Note": "备注", "Total": "总计",
    "Use": "使用", "Reference": "参考", "Selection": "选择", "Selected electrode": "已选电极",
    "All channels": "全部通道", "All files": "全部文件", "All electrodes with non-zero in/out degree": "所有出入度非零电极",
    "Open Main Raster": "打开主 Raster", "Open Raster": "打开 Raster", "Open Map": "打开电极图",
    "Open Result": "打开结果", "Plot": "绘图", "PSTH": "PSTH", "PSTH bin": "PSTH bin",
    "Raster window": "Raster 窗口", "Response window": "响应窗口", "Strong response window": "强响应窗口",
    "Stimulus Response Analysis": "刺激响应分析", "Stimulus Response Files": "刺激响应文件",
    "Stimulus Response Database": "刺激响应数据库", "Stimulus Response Comparison": "刺激响应对比",
    "Stimulus Response Group Statistics": "刺激响应组统计", "Stimulus PSTH": "刺激 PSTH",
    "Stimulus Activation Curve": "刺激激活概率", "Stimulus Channel Map": "刺激电极图",
    "Stable Delay Map": "稳定延迟图", "Dynamics Analysis": "动力学分析",
    "Dynamics Analysis Database": "动力学分析数据库", "Dynamics Analysis Filters": "动力学分析筛选",
    "Multi-file Factor Analysis": "多文件因子分析", "Temporal Coupling": "时间耦合",
    "Temporal Coupling Settings": "时间耦合设置", "Auto Sorting": "自动分选",
    "Sorting Results": "分选结果", "Pipeline Settings": "Pipeline 设置", "Pipeline Results": "Pipeline 结果",
    "Stimulus Generation": "刺激生成", "Experiment": "实验", "Settings": "设置",
    "Electrode group": "电极组", "Event group": "事件组", "Site library": "位点库",
    "Protocol library": "Protocol 库", "Event group library": "事件组库", "Block phases": "Block 阶段",
    "Center": "中心", "Center electrode ID": "中心电极编号", "Electrode count": "电极数量",
    "Stimulation": "刺激", "Stimulation mode": "刺激模式", "Stim electrodes": "刺激电极",
    "Stimulus electrode group": "刺激电极组", "Stimulus file": "刺激文件", "Zero stimulus": "零时刻刺激",
    "Selected trial preview": "已选 trial 预览", "Global response is calculated only for the current preview trial.": "全局响应曲线仅计算当前预览 trial。",
    "Custom Plot": "自定义绘图", "Custom Plot Result": "自定义绘图结果", "X data": "X 数据",
    "Y data": "Y 数据", "X label": "X 轴标签", "Y label": "Y 轴标签", "New Y data": "新增 Y 数据",
    "Plot processed data": "绘制处理后数据", "Select data and click Plot.": "选择数据后点击绘图。",
    "Spontaneous Waveform Amplitude": "自发波形幅值", "Spontaneous Spike Waveform Amplitude": "自发 Spike 波形幅值",
    "Spike Waveform Amplitude": "Spike 波形幅值", "Channel Forecast Response": "通道响应预测",
    "Live raster / heatmap": "实时 Raster / Heatmap", "Controller": "控制器",
    "External controller / network state": "外部控制器 / 网络状态", "Closed-loop rules": "闭环规则",
    "Add Rule": "添加规则", "Remove Rule": "删除规则", "Save Rules": "保存规则",
    "Agent Environment Check": "Agent 环境检测", "Agent Custom Code": "Agent 自定义代码",
    "Task": "任务", "Run log": "运行日志", "Result list": "结果列表", "Generated modules": "生成模块",
    "Step": "步骤", "Status": "状态", "Details": "详情", "Updated": "更新时间", "Summary": "摘要",
    "Description": "描述", "Raw file": "原始文件", "Processed data": "处理后数据", "Stim": "刺激",
    "Add Files": "添加文件", "Add Folder": "添加文件夹", "Clear files": "清除文件",
    "Inspect file": "查看文件", "Analyze": "分析", "Filters...": "筛选...", "Sort by": "排序方式",
    "Connection Degrees": "连接出入度", "Connections": "连接", "Out": "出度", "In": "入度",
}


_JA = {
    "File": "ファイル", "Tools": "ツール", "Language": "言語", "English": "英語",
    "Chinese": "中国語", "Japanese": "日本語", "Exit": "終了", "Open Data Files": "データファイルを開く",
    "Save File": "ファイルを保存", "Clear Loaded Data": "読込データをクリア", "Channel Map": "電極マップ",
    "Sorting": "ソーティング", "Analysis": "解析", "Custom Analysis": "カスタム解析",
    "Channel Forecast Response": "チャネル応答予測", "Stimulus Generation": "刺激生成",
    "Closed Loop Control": "閉ループ制御", "Agent Custom Code": "Agent カスタムコード",
    "MEA Pipeline Studio": "MEA Pipeline スタジオ", "MEA Pipeline\nStudio": "MEA Pipeline\nスタジオ",
    "Data Library": "データ", "Loaded file database": "読込ファイルデータベース",
    "No data files loaded": "データファイル未読込", "Idle": "待機中", "Raw Data Raster": "生データ Raster",
    "Activity log": "実行ログ", "Open": "開く", "Close": "閉じる", "Cancel": "キャンセル", "OK": "OK",
    "Yes": "はい", "No": "いいえ", "Apply": "適用", "Reset": "リセット", "Reset View": "表示をリセット",
    "Save": "保存", "Save Plot": "図を保存", "Remove": "削除", "Add": "追加", "New": "新規",
    "Browse": "参照", "Select": "選択", "Load": "読込", "Run": "実行", "Stop": "停止",
    "Start": "開始", "Pause": "一時停止", "Play": "再生", "Preview": "プレビュー", "Generate": "生成",
    "Refresh": "更新", "Up": "上へ", "Down": "下へ", "Undo": "元に戻す", "Ready": "準備完了",
    "Running": "実行中", "Stopped": "停止済み", "Settings...": "設定...", "Advanced": "詳細設定",
    "Show advanced": "詳細設定を表示", "Display settings...": "表示設定...", "View settings...": "ビュー設定...",
    "Files": "ファイル", "Folder": "フォルダ", "Name": "名前", "Type": "種類", "Kind": "種類",
    "Label": "ラベル", "Source": "ソース", "Origin": "由来", "Condition": "条件",
    "Channels": "チャネル", "Channel": "チャネル", "Electrode": "電極", "Electrodes": "電極",
    "Spikes": "スパイク", "Waveforms": "波形", "Samples": "サンプル", "Features": "特徴量",
    "Time": "時間", "Count": "数", "Mode": "モード", "Group": "グループ", "Protocol": "プロトコル",
    "Block": "ブロック", "Blocks": "ブロック", "Phase": "フェーズ", "Threshold": "しきい値",
    "Window": "ウィンドウ", "Response": "応答", "Pre": "刺激前", "After": "刺激後",
    "Start s": "開始 (s)", "Stop s": "終了 (s)", "Latency ms": "遅延 (ms)", "Note": "メモ",
    "Total": "合計", "Use": "使用", "Reference": "参照", "Selection": "選択",
    "Selected electrode": "選択電極", "All channels": "全チャネル", "All files": "全ファイル",
    "Open Main Raster": "メイン Raster を開く", "Open Raster": "Raster を開く", "Open Map": "マップを開く",
    "Open Result": "結果を開く", "Plot": "描画", "PSTH": "PSTH", "PSTH bin": "PSTH bin",
    "Raster window": "Raster ウィンドウ", "Response window": "応答ウィンドウ",
    "Strong response window": "強応答ウィンドウ", "Stimulus Response Analysis": "刺激応答解析",
    "Stimulus Response Files": "刺激応答ファイル", "Stimulus Response Database": "刺激応答データベース",
    "Stimulus Response Comparison": "刺激応答比較", "Stimulus Response Group Statistics": "刺激応答グループ統計",
    "Stimulus PSTH": "刺激 PSTH", "Stimulus Activation Curve": "刺激活性化確率",
    "Stimulus Channel Map": "刺激電極マップ", "Stable Delay Map": "安定遅延マップ",
    "Dynamics Analysis": "動力学解析", "Dynamics Analysis Database": "動力学解析データベース",
    "Dynamics Analysis Filters": "動力学解析フィルタ", "Multi-file Factor Analysis": "複数ファイル因子分析",
    "Temporal Coupling": "時間結合", "Temporal Coupling Settings": "時間結合設定",
    "Auto Sorting": "自動ソーティング", "Sorting Results": "ソーティング結果",
    "Pipeline Settings": "Pipeline 設定", "Pipeline Results": "Pipeline 結果",
    "Experiment": "実験", "Settings": "設定", "Electrode group": "電極グループ",
    "Event group": "イベントグループ", "Site library": "サイトライブラリ",
    "Protocol library": "プロトコルライブラリ", "Event group library": "イベントグループライブラリ",
    "Block phases": "ブロックフェーズ", "Center": "中心", "Center electrode ID": "中心電極 ID",
    "Electrode count": "電極数", "Stimulation": "刺激", "Stimulation mode": "刺激モード",
    "Stim electrodes": "刺激電極", "Stimulus electrode group": "刺激電極グループ",
    "Stimulus file": "刺激ファイル", "Zero stimulus": "基準刺激", "Selected trial preview": "選択 trial プレビュー",
    "Custom Plot": "カスタムプロット", "Custom Plot Result": "カスタムプロット結果",
    "X data": "X データ", "Y data": "Y データ", "X label": "X 軸ラベル", "Y label": "Y 軸ラベル",
    "New Y data": "Y データ追加", "Plot processed data": "処理済みデータを描画",
    "Spontaneous Waveform Amplitude": "自発波形振幅", "Spontaneous Spike Waveform Amplitude": "自発スパイク波形振幅",
    "Spike Waveform Amplitude": "スパイク波形振幅", "Live raster / heatmap": "リアルタイム Raster / Heatmap",
    "Controller": "コントローラ", "External controller / network state": "外部コントローラ / ネットワーク状態",
    "Closed-loop rules": "閉ループルール", "Add Rule": "ルール追加", "Remove Rule": "ルール削除",
    "Save Rules": "ルール保存", "Agent Environment Check": "Agent 環境チェック",
    "Task": "タスク", "Run log": "実行ログ", "Result list": "結果リスト", "Generated modules": "生成モジュール",
    "Step": "手順", "Status": "状態", "Details": "詳細", "Updated": "更新日時", "Summary": "概要",
    "Description": "説明", "Raw file": "生ファイル", "Processed data": "処理済みデータ", "Stim": "刺激",
    "Add Files": "ファイル追加", "Add Folder": "フォルダ追加", "Clear files": "ファイルをクリア",
    "Inspect file": "ファイルを確認", "Analyze": "解析", "Filters...": "フィルタ...", "Sort by": "並び順",
    "Connection Degrees": "接続次数", "Connections": "接続", "Out": "出次数", "In": "入次数",
}


_WORD_ZH = {
    "settings": "设置", "analysis": "分析", "data": "数据", "file": "文件", "files": "文件",
    "channel": "通道", "channels": "通道", "electrode": "电极", "electrodes": "电极",
    "stimulus": "刺激", "stimulation": "刺激", "response": "响应", "window": "窗口",
    "selected": "已选", "select": "选择", "group": "组", "library": "库", "database": "数据库",
    "result": "结果", "results": "结果", "preview": "预览", "display": "显示", "threshold": "阈值",
    "duration": "时长", "start": "开始", "stop": "停止", "save": "保存", "load": "加载",
    "open": "打开", "remove": "删除", "add": "添加", "clear": "清除", "default": "默认",
    "current": "当前", "all": "全部", "custom": "自定义", "automatic": "自动", "manual": "手动",
    "loaded": "已加载", "loading": "正在加载", "saved": "已保存", "completed": "已完成",
    "complete": "完成", "failed": "失败", "cancelled": "已取消", "canceled": "已取消",
    "skipped": "已跳过", "starting": "正在启动", "updated": "已更新", "selected": "已选择",
    "created": "已创建", "removed": "已删除", "produced": "已生成", "ready": "就绪",
    "running": "正在运行", "warnings": "警告", "warning": "警告", "error": "错误",
    "source": "来源", "output": "输出", "shape": "形状", "processed": "已处理", "dataset": "数据集",
    "datasets": "数据集", "records": "记录", "record": "记录", "workspace": "工作区",
    "agent": "Agent", "environment": "环境", "verified": "已验证", "verification": "验证",
    "using": "正在使用", "cached": "缓存的", "click": "点击", "retest": "重新测试", "check": "检查",
    "again": "再次", "probe": "探测", "timed": "计时", "timeout": "超时", "modified": "已修改",
    "module": "模块", "modules": "模块", "shell": "框架", "command": "命令", "found": "找到",
    "preflight": "预检测", "task": "任务", "prepared": "已准备", "another": "另一个", "already": "已经",
    "process": "进程", "stopping": "正在停止", "finished": "已结束", "code": "代码", "status": "状态",
    "timing": "耗时", "generation": "生成", "instructions": "指令", "missing": "缺失", "successfully": "成功",
    "summary": "摘要", "inspect": "检查", "generated": "已生成", "syntax": "语法", "details": "详情",
    "read": "读取", "write": "写入", "folder": "文件夹", "rows": "行", "row": "行",
}

_WORD_JA = {
    "settings": "設定", "analysis": "解析", "data": "データ", "file": "ファイル", "files": "ファイル",
    "channel": "チャネル", "channels": "チャネル", "electrode": "電極", "electrodes": "電極",
    "stimulus": "刺激", "stimulation": "刺激", "response": "応答", "window": "ウィンドウ",
    "selected": "選択", "select": "選択", "group": "グループ", "library": "ライブラリ", "database": "データベース",
    "result": "結果", "results": "結果", "preview": "プレビュー", "display": "表示", "threshold": "しきい値",
    "duration": "時間", "start": "開始", "stop": "停止", "save": "保存", "load": "読込",
    "open": "開く", "remove": "削除", "add": "追加", "clear": "クリア", "default": "既定",
    "current": "現在", "all": "全て", "custom": "カスタム", "automatic": "自動", "manual": "手動",
    "loaded": "読込済み", "loading": "読込中", "saved": "保存済み", "completed": "完了",
    "complete": "完了", "failed": "失敗", "cancelled": "キャンセル済み", "canceled": "キャンセル済み",
    "skipped": "スキップ", "starting": "開始中", "updated": "更新済み", "selected": "選択済み",
    "created": "作成済み", "removed": "削除済み", "produced": "生成済み", "ready": "準備完了",
    "running": "実行中", "warnings": "警告", "warning": "警告", "error": "エラー",
    "source": "ソース", "output": "出力", "shape": "形状", "processed": "処理済み", "dataset": "データセット",
    "datasets": "データセット", "records": "レコード", "record": "レコード", "workspace": "ワークスペース",
    "agent": "Agent", "environment": "環境", "verified": "検証済み", "verification": "検証",
    "using": "使用中", "cached": "キャッシュ済み", "click": "クリック", "retest": "再テスト", "check": "確認",
    "again": "再度", "probe": "プローブ", "timed": "計時", "timeout": "タイムアウト", "modified": "変更済み",
    "module": "モジュール", "modules": "モジュール", "shell": "シェル", "command": "コマンド", "found": "検出",
    "preflight": "事前確認", "task": "タスク", "prepared": "準備済み", "another": "別の", "already": "すでに",
    "process": "プロセス", "stopping": "停止中", "finished": "終了", "code": "コード", "status": "状態",
    "timing": "所要時間", "generation": "生成", "instructions": "指示", "missing": "不足", "successfully": "正常に",
    "summary": "概要", "inspect": "確認", "generated": "生成済み", "syntax": "構文", "details": "詳細",
    "read": "読み込み", "write": "書き込み", "folder": "フォルダー", "rows": "行", "row": "行",
}


_EXTRA_ZH = {
    "artifact": "\u4f2a\u8ff9", "average": "\u5e73\u5747", "amplitude": "\u5e45\u503c", "activation": "\u6fc0\u6d3b", "curve": "\u66f2\u7ebf",
    "burst": "\u7a81\u53d1", "bursts": "\u7a81\u53d1", "pulse": "\u8109\u51b2", "pulses": "\u8109\u51b2", "rate": "\u53d1\u653e\u7387", "firing": "\u53d1\u653e",
    "profile": "\u8f6e\u5ed3", "mean": "\u5747\u503c", "median": "\u4e2d\u4f4d\u6570", "maximum": "\u6700\u5927\u503c", "minimum": "\u6700\u5c0f\u503c",
    "valid": "\u6709\u6548", "defined": "\u5df2\u5b9a\u4e49", "choose": "\u9009\u62e9", "requires": "\u9700\u8981", "required": "\u5fc5\u586b", "available": "\u53ef\u7528",
    "input": "\u8f93\u5165", "view": "\u89c6\u56fe", "map": "\u56fe", "hide": "\u9690\u85cf", "show": "\u663e\u793a", "connection": "\u8fde\u63a5", "connections": "\u8fde\u63a5", "degree": "\u5ea6",
    "first": "\u7b2c\u4e00\u4e2a", "peak": "\u5cf0\u503c", "stable": "\u7a33\u5b9a", "latency": "\u5ef6\u8fdf", "trial": "\u8bd5\u6b21", "trials": "\u8bd5\u6b21", "bin": "\u5206\u7bb1", "bins": "\u5206\u7bb1",
    "typical": "\u5178\u578b", "range": "\u8303\u56f4", "within": "\u5728...\u5185", "around": "\u5468\u56f4", "each": "\u6bcf\u4e2a", "before": "\u4e4b\u524d", "after": "\u4e4b\u540e", "visible": "\u53ef\u89c1",
    "spatial": "\u7a7a\u95f4", "temporal": "\u65f6\u95f4", "latent": "\u6f5c\u5728", "model": "\u6a21\u578b", "models": "\u6a21\u578b", "fit": "\u62df\u5408", "fitting": "\u62df\u5408", "state": "\u72b6\u6001", "states": "\u72b6\u6001",
    "waveform": "\u6ce2\u5f62", "waveforms": "\u6ce2\u5f62", "embedding": "\u5d4c\u5165", "assignment": "\u5206\u914d", "duplicate": "\u91cd\u590d", "name": "\u540d\u79f0", "plotted": "\u5df2\u7ed8\u5236",
    "no": "\u6ca1\u6709", "available": "\u53ef\u7528", "not": "\u4e0d", "only": "\u4ec5", "total": "\u603b\u8ba1", "index": "\u7d22\u5f15", "sample": "\u6837\u672c", "samples": "\u6837\u672c",
}

_EXTRA_JA = {
    "artifact": "\u30a2\u30fc\u30c1\u30d5\u30a1\u30af\u30c8", "average": "\u5e73\u5747", "amplitude": "\u632f\u5e45", "activation": "\u6d3b\u6027\u5316", "curve": "\u66f2\u7dda",
    "burst": "\u30d0\u30fc\u30b9\u30c8", "bursts": "\u30d0\u30fc\u30b9\u30c8", "pulse": "\u30d1\u30eb\u30b9", "pulses": "\u30d1\u30eb\u30b9", "rate": "\u767a\u706b\u7387", "firing": "\u767a\u706b",
    "profile": "\u30d7\u30ed\u30d5\u30a1\u30a4\u30eb", "mean": "\u5e73\u5747\u5024", "median": "\u4e2d\u592e\u5024", "maximum": "\u6700\u5927", "minimum": "\u6700\u5c0f",
    "valid": "\u6709\u52b9", "defined": "\u5b9a\u7fa9\u6e08\u307f", "choose": "\u9078\u629e", "requires": "\u5fc5\u8981", "required": "\u5fc5\u9808", "available": "\u5229\u7528\u53ef\u80fd",
    "input": "\u5165\u529b", "view": "\u8868\u793a", "map": "\u30de\u30c3\u30d7", "hide": "\u975e\u8868\u793a", "show": "\u8868\u793a", "connection": "\u63a5\u7d9a", "connections": "\u63a5\u7d9a", "degree": "\u6b21\u6570",
    "first": "\u6700\u521d\u306e", "peak": "\u30d4\u30fc\u30af", "stable": "\u5b89\u5b9a", "latency": "\u9045\u5ef6", "trial": "\u8a66\u884c", "trials": "\u8a66\u884c", "bin": "\u30d3\u30f3", "bins": "\u30d3\u30f3",
    "typical": "\u4e00\u822c\u7684", "range": "\u7bc4\u56f2", "within": "\u4ee5\u5185", "around": "\u5468\u8fba", "each": "\u5404", "before": "\u524d", "after": "\u5f8c", "visible": "\u8868\u793a\u4e2d",
    "spatial": "\u7a7a\u9593", "temporal": "\u6642\u9593", "latent": "\u6f5c\u5728", "model": "\u30e2\u30c7\u30eb", "models": "\u30e2\u30c7\u30eb", "fit": "\u30d5\u30a3\u30c3\u30c8", "fitting": "\u30d5\u30a3\u30c3\u30c8", "state": "\u72b6\u614b", "states": "\u72b6\u614b",
    "waveform": "\u6ce2\u5f62", "waveforms": "\u6ce2\u5f62", "embedding": "\u57cb\u3081\u8fbc\u307f", "assignment": "\u5272\u308a\u5f53\u3066", "duplicate": "\u91cd\u8907", "name": "\u540d\u524d", "plotted": "\u30d7\u30ed\u30c3\u30c8\u6e08\u307f",
    "no": "\u306a\u3057", "not": "\u306a\u3044", "only": "\u306e\u307f", "total": "\u5408\u8a08", "index": "\u30a4\u30f3\u30c7\u30c3\u30af\u30b9", "sample": "\u30b5\u30f3\u30d7\u30eb", "samples": "\u30b5\u30f3\u30d7\u30eb",
}

_DYNAMIC_ZH = {
    "No data files loaded.": "\u5c1a\u672a\u52a0\u8f7d\u6570\u636e\u6587\u4ef6\u3002",
    "Run factor analysis first.": "\u8bf7\u5148\u8fd0\u884c\u56e0\u5b50\u5206\u6790\u3002",
    "No spike data is available.": "\u6ca1\u6709\u53ef\u7528\u7684 spike \u6570\u636e\u3002",
    "Generation failed": "\u751f\u6210\u5931\u8d25", "Run completed.": "\u8fd0\u884c\u5b8c\u6210\u3002",
    "Process finished": "\u8fdb\u7a0b\u5df2\u7ed3\u675f", "Process error": "\u8fdb\u7a0b\u9519\u8bef",
    "Run timed out.": "\u8fd0\u884c\u8d85\u65f6\u3002", "Stopping process...": "\u6b63\u5728\u505c\u6b62\u8fdb\u7a0b...",
    "Agent task prepared.": "Agent \u4efb\u52a1\u5df2\u51c6\u5907\u3002", "Agent generated module.py successfully.": "Agent \u5df2\u6210\u529f\u751f\u6210 module.py\u3002",
    "Select a saved module to modify.": "\u8bf7\u9009\u62e9\u8981\u4fee\u6539\u7684\u5df2\u4fdd\u5b58\u6a21\u5757\u3002", "Select or create a module first.": "\u8bf7\u5148\u9009\u62e9\u6216\u521b\u5efa\u6a21\u5757\u3002",
    "Another process is already running.": "\u53e6\u4e00\u4e2a\u8fdb\u7a0b\u5df2\u5728\u8fd0\u884c\u3002", "Another agent command is already running.": "\u53e6\u4e00\u4e2a Agent \u547d\u4ee4\u5df2\u5728\u8fd0\u884c\u3002",
    "Command failed": "\u547d\u4ee4\u5931\u8d25", "Command error": "\u547d\u4ee4\u9519\u8bef", "CFG path missing": "CFG \u8def\u5f84\u7f3a\u5931", "CFG not found": "\u672a\u627e\u5230 CFG",
    "Runner path missing": "Runner \u8def\u5f84\u7f3a\u5931", "Runner not found": "\u672a\u627e\u5230 Runner", "Runner error": "Runner \u9519\u8bef",
    "Auto sorting running...": "\u81ea\u52a8\u6392\u5e8f\u8fd0\u884c\u4e2d...", "Auto sorting complete": "\u81ea\u52a8\u6392\u5e8f\u5b8c\u6210", "Auto sorting failed": "\u81ea\u52a8\u6392\u5e8f\u5931\u8d25",
    "Total duration: --": "\u603b\u65f6\u957f\uff1a--", "Hide advanced": "\u9690\u85cf\u9ad8\u7ea7\u8bbe\u7f6e", "Show advanced": "\u663e\u793a\u9ad8\u7ea7\u8bbe\u7f6e",
}

_DYNAMIC_JA = {
    "No data files loaded.": "\u30c7\u30fc\u30bf\u30d5\u30a1\u30a4\u30eb\u306f\u8aad\u307f\u8fbc\u307e\u308c\u3066\u3044\u307e\u305b\u3093\u3002",
    "Run factor analysis first.": "\u5148\u306b\u56e0\u5b50\u5206\u6790\u3092\u5b9f\u884c\u3057\u3066\u304f\u3060\u3055\u3044\u3002", "No spike data is available.": "\u5229\u7528\u53ef\u80fd\u306a spike \u30c7\u30fc\u30bf\u304c\u3042\u308a\u307e\u305b\u3093\u3002",
    "Generation failed": "\u751f\u6210\u5931\u6557", "Run completed.": "\u5b9f\u884c\u5b8c\u4e86", "Process finished": "\u30d7\u30ed\u30bb\u30b9\u5b8c\u4e86", "Process error": "\u30d7\u30ed\u30bb\u30b9\u30a8\u30e9\u30fc",
    "Run timed out.": "\u5b9f\u884c\u30bf\u30a4\u30e0\u30a2\u30a6\u30c8", "Stopping process...": "\u30d7\u30ed\u30bb\u30b9\u3092\u505c\u6b62\u4e2d...",
    "Agent task prepared.": "Agent \u30bf\u30b9\u30af\u3092\u6e96\u5099\u3057\u307e\u3057\u305f", "Agent generated module.py successfully.": "Agent が module.py を正常に生成しました。",
    "Select a saved module to modify.": "\u5909\u66f4\u3059\u308b\u4fdd\u5b58\u6e08\u307f\u30e2\u30b8\u30e5\u30fc\u30eb\u3092\u9078\u629e\u3057\u3066\u304f\u3060\u3055\u3044。", "Select or create a module first.": "\u5148\u306b\u30e2\u30b8\u30e5\u30fc\u30eb\u3092\u9078\u629e\u307e\u305f\u306f\u4f5c\u6210\u3057\u3066\u304f\u3060\u3055\u3044。",
    "Another process is already running.": "\u5225\u306e\u30d7\u30ed\u30bb\u30b9\u304c\u3059\u3067\u306b\u5b9f\u884c\u4e2d\u3067\u3059", "Another agent command is already running.": "\u5225\u306e Agent \u30b3\u30de\u30f3\u30c9\u304c\u3059\u3067\u306b\u5b9f\u884c\u4e2d\u3067\u3059",
    "Command failed": "\u30b3\u30de\u30f3\u30c9\u5931\u6557", "Command error": "\u30b3\u30de\u30f3\u30c9\u30a8\u30e9\u30fc", "CFG path missing": "CFG \u30d1\u30b9\u304c\u3042\u308a\u307e\u305b\u3093", "CFG not found": "CFG \u304c\u898b\u3064\u304b\u308a\u307e\u305b\u3093",
    "Runner path missing": "Runner \u30d1\u30b9\u304c\u3042\u308a\u307e\u305b\u3093", "Runner not found": "Runner \u304c\u898b\u3064\u304b\u308a\u307e\u305b\u3093", "Runner error": "Runner \u30a8\u30e9\u30fc",
    "Auto sorting running...": "\u81ea\u52d5\u30bd\u30fc\u30c6\u30a3\u30f3\u30b0\u5b9f\u884c\u4e2d...", "Auto sorting complete": "\u81ea\u52d5\u30bd\u30fc\u30c6\u30a3\u30f3\u30b0\u5b8c\u4e86", "Auto sorting failed": "\u81ea\u52d5\u30bd\u30fc\u30c6\u30a3\u30f3\u30b0\u5931\u6557",
    "Total duration: --": "\u5408\u8a08\u6642\u9593: --", "Hide advanced": "\u8a73\u7d30\u8a2d\u5b9a\u3092\u975e\u8868\u793a", "Show advanced": "\u8a73\u7d30\u8a2d\u5b9a\u3092\u8868\u793a",
}

_PHRASE_ZH = {
    "Duplicate name": "\u540d\u79f0\u91cd\u590d", ". Choose a different name.": "\u8bf7\u9009\u62e9\u5176\u4ed6\u540d\u79f0\u3002", "New Map": "\u65b0\u5efa\u7535\u6781\u56fe", "Map name": "\u7535\u6781\u56fe\u540d\u79f0",
    "Auto Detect": "\u81ea\u52a8\u68c0\u6d4b", "Move To": "\u79fb\u52a8\u5230", "Setup Guide": "\u914d\u7f6e\u6307\u5357", "Agent Setup Guide": "Agent \u914d\u7f6e\u6307\u5357",
    "Artifact 0-3 ms": "\u4f2a\u8ff9 0-3 ms", "Total spike count / bin": "\u6bcf\u4e2a\u5206\u7bb1\u7684 Spike \u603b\u6570", "No valid time windows were defined": "\u672a\u5b9a\u4e49\u6709\u6548\u7684\u65f6\u95f4\u7a97\u53e3",
    "Normalized firing profile": "\u5f52\u4e00\u5316\u53d1\u653e\u8f6e\u5ed3", "Robust scaled firing profile": "\u9c81\u68d2\u7f29\u653e\u53d1\u653e\u8f6e\u5ed3", "Fitting aligned latent models...": "\u6b63\u5728\u62df\u5408\u5bf9\u9f50\u7684\u6f5c\u5728\u6a21\u578b...",
    "single pulse": "\u5355\u8109\u51b2", "Rate (Hz/ch)": "\u53d1\u653e\u7387 (Hz/\u901a\u9053)", "Linear Dynamical System (LDS)": "\u7ebf\u6027\u52a8\u529b\u5b66\u7cfb\u7edf (LDS)", "Per time total": "\u6bcf\u4e2a\u65f6\u95f4\u70b9\u603b\u548c",
    "Min total activity": "\u6700\u5c0f\u603b\u6d3b\u52a8\u91cf", "Min active bursts/windows": "\u6700\u5c0f\u6d3b\u52a8\u7a81\u53d1/\u7a97\u53e3\u6570", "Min variance": "\u6700\u5c0f\u65b9\u5dee", "Reset map view": "\u91cd\u7f6e\u7535\u6781\u56fe\u89c6\u56fe", "Apply text": "\u5e94\u7528\u6587\u672c", "Sample labels": "\u6837\u672c\u6807\u7b7e", "blank = auto": "\u7559\u7a7a = \u81ea\u52a8", "X step": "X \u6b65\u957f", "Activation curve": "\u6fc0\u6d3b\u66f2\u7ebf", "Reset / Apply": "\u91cd\u7f6e / \u5e94\u7528",
    "First spike latency (ms)": "\u9996\u4e2a Spike \u5ef6\u8fdf (ms)", "Trial-level activation probabilities": "\u8bd5\u6b21\u7ea7\u6fc0\u6d3b\u6982\u7387", "Waveform clustering (NEV)": "\u6ce2\u5f62\u805a\u7c7b (NEV)", "None (scaled waveform)": "\u65e0 (\u7f29\u653e\u6ce2\u5f62)", "Gaussian Mixture": "\u9ad8\u65af\u6df7\u5408\u6a21\u578b",
    "PCA components": "PCA \u6210\u5206", "ICA components": "ICA \u6210\u5206", "Max clusters": "\u6700\u5927\u805a\u7c7b\u6570", "Cluster count": "\u805a\u7c7b\u6570", "Average Spike Amplitude": "Spike \u5e73\u5747\u5e45\u503c", "Firing Rate": "\u53d1\u653e\u7387",
    "Sampling rate": "\u91c7\u6837\u7387", "Low cut": "\u4f4e\u622a\u6b62\u9891\u7387", "High cut": "\u9ad8\u622a\u6b62\u9891\u7387", "Spike waveforms": "Spike \u6ce2\u5f62", "Time (sample index)": "\u65f6\u95f4 (\u6837\u672c\u7d22\u5f15)", "Voltage (uV)": "\u7535\u538b (uV)", "Avg rate": "\u5e73\u5747\u53d1\u653e\u7387", "Time (s)": "\u65f6\u95f4 (s)", "Hide connections": "\u9690\u85cf\u8fde\u63a5", "Burst Trajectory": "\u7a81\u53d1\u8f68\u8ff9",
}

_PHRASE_JA = {
    "Duplicate name": "\u540d\u524d\u304c\u91cd\u8907\u3057\u3066\u3044\u307e\u3059", ". Choose a different name.": "\u5225\u306e\u540d\u524d\u3092\u9078\u629e\u3057\u3066\u304f\u3060\u3055\u3044", "New Map": "\u65b0\u3057\u3044\u96fb\u6975\u30de\u30c3\u30d7", "Map name": "\u30de\u30c3\u30d7\u540d",
    "Auto Detect": "\u81ea\u52d5\u691c\u51fa", "Move To": "\u79fb\u52d5\u5148", "Setup Guide": "\u8a2d\u5b9a\u30ac\u30a4\u30c9", "Agent Setup Guide": "Agent \u8a2d\u5b9a\u30ac\u30a4\u30c9",
    "Artifact 0-3 ms": "\u30a2\u30fc\u30c1\u30d5\u30a1\u30af\u30c8 0-3 ms", "Total spike count / bin": "\u30d3\u30f3\u3042\u305f\u308a\u306e\u30b9\u30d1\u30a4\u30af\u7dcf\u6570", "No valid time windows were defined": "\u6709\u52b9\u306a\u6642\u9593\u30a6\u30a3\u30f3\u30c9\u30a6\u304c\u5b9a\u7fa9\u3055\u308c\u3066\u3044\u307e\u305b\u3093",
    "Normalized firing profile": "\u6b63\u898f\u5316\u3055\u308c\u305f\u767a\u706b\u30d7\u30ed\u30d5\u30a1\u30a4\u30eb", "Robust scaled firing profile": "\u30ed\u30d0\u30b9\u30c8\u30b9\u30b1\u30fc\u30eb\u767a\u706b\u30d7\u30ed\u30d5\u30a1\u30a4\u30eb", "Fitting aligned latent models...": "\u30a2\u30e9\u30a4\u30f3\u6e08\u307f\u6f5c\u5728\u30e2\u30c7\u30eb\u3092\u30d5\u30a3\u30c3\u30c8\u4e2d...",
    "single pulse": "\u30b7\u30f3\u30b0\u30eb\u30d1\u30eb\u30b9", "Rate (Hz/ch)": "\u767a\u706b\u7387 (Hz/\u30c1\u30e3\u30f3\u30cd\u30eb)", "Linear Dynamical System (LDS)": "\u7dda\u5f62\u52d5\u529b\u5b66\u7cfb\u7d71 (LDS)", "Per time total": "\u6642\u523b\u3054\u3068\u306e\u5408\u8a08",
    "Min total activity": "\u6700\u5c0f\u7dcf\u6d3b\u52d5\u91cf", "Min active bursts/windows": "\u6700\u5c0f\u30d0\u30fc\u30b9\u30c8/\u30a6\u30a3\u30f3\u30c9\u30a6\u6570", "Min variance": "\u6700\u5c0f\u5206\u6563", "Reset map view": "\u30de\u30c3\u30d7\u8868\u793a\u3092\u30ea\u30bb\u30c3\u30c8", "Apply text": "\u30c6\u30ad\u30b9\u30c8\u3092\u9069\u7528", "Sample labels": "\u30b5\u30f3\u30d7\u30eb\u30e9\u30d9\u30eb", "blank = auto": "\u7a7a\u6b04 = \u81ea\u52d5", "X step": "X \u30b9\u30c6\u30c3\u30d7", "Activation curve": "\u6d3b\u6027\u5316\u66f2\u7dda", "Reset / Apply": "\u30ea\u30bb\u30c3\u30c8 / \u9069\u7528",
    "First spike latency (ms)": "\u521d\u56de\u30b9\u30d1\u30a4\u30af\u9045\u5ef6 (ms)", "Trial-level activation probabilities": "\u8a66\u884c\u30ec\u30d9\u30eb\u6d3b\u6027\u5316\u78ba\u7387", "Waveform clustering (NEV)": "\u6ce2\u5f62\u30af\u30e9\u30b9\u30bf\u30ea\u30f3\u30b0 (NEV)", "None (scaled waveform)": "\u306a\u3057 (\u30b9\u30b1\u30fc\u30eb\u6e08\u307f\u6ce2\u5f62)", "Gaussian Mixture": "\u30ac\u30a6\u30b9\u6df7\u5408\u30e2\u30c7\u30eb",
    "PCA components": "PCA \u6210\u5206", "ICA components": "ICA \u6210\u5206", "Max clusters": "\u6700\u5927\u30af\u30e9\u30b9\u30bf\u6570", "Cluster count": "\u30af\u30e9\u30b9\u30bf\u6570", "Average Spike Amplitude": "\u30b9\u30d1\u30a4\u30af\u5e73\u5747\u632f\u5e45", "Firing Rate": "\u767a\u706b\u7387",
    "Sampling rate": "\u30b5\u30f3\u30d7\u30ea\u30f3\u30b0\u30ec\u30fc\u30c8", "Low cut": "\u4f4e\u57df\u30ab\u30c3\u30c8", "High cut": "\u9ad8\u57df\u30ab\u30c3\u30c8", "Spike waveforms": "\u30b9\u30d1\u30a4\u30af\u6ce2\u5f62", "Time (sample index)": "\u6642\u9593 (\u30b5\u30f3\u30d7\u30eb\u30a4\u30f3\u30c7\u30c3\u30af\u30b9)", "Voltage (uV)": "\u96fb\u5727 (uV)", "Avg rate": "\u5e73\u5747\u767a\u706b\u7387", "Time (s)": "\u6642\u9593 (s)", "Hide connections": "\u63a5\u7d9a\u3092\u975e\u8868\u793a", "Burst Trajectory": "\u30d0\u30fc\u30b9\u30c8\u8ecc\u8de1",
}

_PHRASE_ZH.update({
    "Reference electrode": "参考电极",
    "Spikes": "Spike 数",
    "Single stimulus only": "仅单次刺激",
    "Preview cluster": "预览簇",
    "Selected trial preview": "选中试次预览",
    "All data windows": "全部数据窗口",
    "Channel z-score": "通道 z-score",
    "Log + channel z-score": "对数 + 通道 z-score",
    "None": "无",
    "Select loaded database files, choose FA, LDS, or pi-VAE, then run dynamics analysis.": "选择已加载的数据文件，选择 FA、LDS 或 pi-VAE，然后运行动力学分析。",
    "Select one loaded spike file to detect stable-delay electrodes and stable directed delay connections.": "选择一个已加载的 spike 文件，以检测稳定延迟电极和稳定有向延迟连接。",
    "Custom workflow": "自定义工作流",
    "Choose a basic function first. Data range, channels, and plotting are configured in the following steps.": "请先选择基础功能，后续步骤中再配置数据范围、通道和绘图。",
    "Run custom pipeline": "运行自定义流程",
    "Custom Data Selection": "自定义数据选择",
    "blank = all selected; e.g. chan1, chan2": "留空 = 全部已选；例如 chan1, chan2",
    "optional dataset prefix": "可选的数据集前缀",
    "Open Main Raster": "打开主 Raster",
    "Open Raster": "打开 Raster",
    "Save Selection + Analyze": "保存选择并分析",
    "Auto": "自动",
    "Channel / feature labels": "通道 / 特征标签",
    "Manual numeric values": "手动数值",
    "Line": "折线图",
    "Bar": "柱状图",
    "legend": "图例",
    "Group statistics": "分组统计",
    "Local response": "局部响应",
    "Electrode sequence": "电极顺序",
    "Lasso channels": "套索选择通道",
    "All sites": "全部位点",
    "ICA max iterations": "ICA 最大迭代次数",
    "GMM covariance": "GMM 协方差",
    "DBSCAN eps": "DBSCAN 邻域半径",
    "DBSCAN min samples": "DBSCAN 最小样本数",
    "Min silhouette": "最小轮廓系数",
    "Maxwell Footprint Analysis": "Maxwell 足迹分析",
    "Enable": "启用",
    "Run Footprint Analysis": "运行足迹分析",
    "Pipeline Settings": "Pipeline 设置",
    "Enable channel-wise z-score normalization": "启用逐通道 z-score 归一化",
    "Locate": "定位",
    "Linear": "线性",
    "RBF kernel ridge": "RBF 核岭回归",
    "kNN local average": "kNN 局部平均",
    "Polynomial ridge": "多项式岭回归",
    "Random forest": "随机森林",
    "Gradient boosting": "梯度提升",
    "RMSE high to low": "RMSE 从高到低",
    "RMSE low to high": "RMSE 从低到高",
    "Recon Metrics": "重建指标",
    "W Metrics": "W 指标",
    "Trajectory": "轨迹",
    "Normalized Time": "归一化时间",
    "Burst Trajectory Settings": "Burst 轨迹设置",
    "Hide stim tail": "隐藏刺激尾迹",
    "Raster display settings": "Raster 显示设置",
    "Burst detection settings": "Burst 检测设置",
    "MaxWell Closed Loop": "MaxWell 闭环",
    "Simulation": "仿真",
    "External network": "外部网络",
    "Detect ch": "检测通道",
    "Sequence": "序列",
    "Build C++": "构建 C++",
    "Dry-run Setup": "试运行配置",
    "Setup Hardware": "配置硬件",
    "Follow": "跟随",
    "Pipeline Results": "Pipeline 结果",
    "Run Auto Sorting": "运行自动 Sorting",
    "Run Channel Sorting": "运行通道 Sorting",
    "Footprint Analysis": "足迹分析",
    "Save Sorting": "保存 Sorting",
    "Compute Embedding": "计算嵌入",
    "Assign Cluster": "分配聚类",
    "Mark Tail/Noise": "标记尾迹/噪声",
    "Hide noise": "隐藏噪声",
    "Unsaved sorting": "未保存的 Sorting",
    "All clusters": "全部聚类",
    "High to low": "从高到低",
    "Low to high": "从低到高",
    "Target": "目标",
    "Lag ms": "延迟 ms",
    "Lag SD ms": "延迟标准差 ms",
    "Match": "匹配",
    "A trial is counted as strong response when this post-stimulus count exceeds the non-stimulus/non-burst baseline count mean by more than five standard deviations.": "当刺激后计数超过非刺激、非 Burst 基线计数均值五个标准差以上时，该试次被判定为强响应。",
    "Time bin for burst/activity vectors. Typical range: 5-20 ms.": "Burst/活动向量的时间分箱，典型范围为 5-20 ms。",
    "Analysis window per burst or per non-overlapping segment. Typical range: 100-500 ms.": "每个 Burst 或非重叠片段的分析窗口，典型范围为 100-500 ms。",
    "Use detected bursts, or split each file into non-overlapping windows.": "使用检测到的 Burst，或将每个文件划分为互不重叠的窗口。",
    "Preprocessing applied before fitting. Channel z-score is usually the safest default.": "拟合前应用的预处理。通道 z-score 通常是最稳妥的默认选项。",
    "Number of latent factors/states. Typical range: 8-32 for routine exploration.": "潜在因子/状态数量，常规探索的典型范围为 8-32。",
    "When enabled, drag on the channel map. Left drag selects; right drag removes.": "启用后可在通道图上拖动：左键拖动选择，右键拖动取消。",
    "Open the selected source file in the main Raw Data Raster window.": "在主 Raw Data Raster 窗口中打开选中的源文件。",
    "Compare population activation and PSTH similarity for each stimulus electrode group.": "比较各刺激电极组的群体激活和 PSTH 相似度。",
    "Visible raster window. Default shows pre-stimulus and post-stimulus together.": "可见 Raster 时间窗，默认同时显示刺激前和刺激后。",
    "Local response groups nearby and behaviorally similar channels together.": "局部响应模式将空间邻近且响应相似的通道归为一组。",
    "Visible time span on the raster. Typically set to cover pre + after response.": "Raster 的可见时间跨度，通常设置为覆盖刺激前和刺激后响应。",
    "Local response groups channels by spatial/response similarity; electrode sequence preserves map order.": "局部响应按空间和响应相似性分组；电极顺序模式保留 Map 顺序。",
    "Remove spikes within a short artifact window after each stimulation event.": "移除每次刺激事件后短伪迹窗口内的 Spike。",
    "How many channels are shown at once in the raster viewport.": "Raster 视口中同时显示的通道数量。",
    "Heatmap, artifact-tail and other raster display options.": "Heatmap、伪迹尾迹及其他 Raster 显示选项。",
    "Burst detection parameters used by burst-related analyses.": "Burst 相关分析使用的 Burst 检测参数。",
    "The current sorting result has not been saved. Save before closing?": "当前 Sorting 结果尚未保存，关闭前是否保存？",
})

_PHRASE_JA.update({
    "Reference electrode": "参照電極", "Spikes": "スパイク数", "Single stimulus only": "単一刺激のみ",
    "Preview cluster": "プレビュークラスタ", "Selected trial preview": "選択試行のプレビュー", "All data windows": "全データウィンドウ",
    "Channel z-score": "チャンネル z-score", "Log + channel z-score": "対数 + チャンネル z-score", "None": "なし",
    "Custom workflow": "カスタムワークフロー", "Run custom pipeline": "カスタムパイプラインを実行", "Custom Data Selection": "カスタムデータ選択",
    "Open Main Raster": "メイン Raster を開く", "Open Raster": "Raster を開く", "Save Selection + Analyze": "選択を保存して解析",
    "Auto": "自動", "Channel / feature labels": "チャンネル / 特徴ラベル", "Manual numeric values": "手動数値", "Line": "折れ線", "Bar": "棒グラフ", "legend": "凡例",
    "Group statistics": "グループ統計", "Local response": "局所応答", "Electrode sequence": "電極順序", "Lasso channels": "チャンネルを投げ縄選択", "All sites": "全サイト",
    "Enable": "有効", "Pipeline Settings": "Pipeline 設定", "Locate": "位置を表示", "Linear": "線形", "Random forest": "ランダムフォレスト",
    "Trajectory": "軌跡", "Normalized Time": "正規化時間", "Hide stim tail": "刺激テールを非表示", "Raster display settings": "Raster 表示設定",
    "Burst detection settings": "Burst 検出設定", "MaxWell Closed Loop": "MaxWell 閉ループ", "Simulation": "シミュレーション", "External network": "外部ネットワーク",
    "Detect ch": "検出チャンネル", "Sequence": "シーケンス", "Build C++": "C++ をビルド", "Dry-run Setup": "ドライラン設定", "Setup Hardware": "ハードウェア設定", "Follow": "追従",
    "Run Auto Sorting": "自動 Sorting を実行", "Run Channel Sorting": "チャンネル Sorting を実行", "Save Sorting": "Sorting を保存", "Compute Embedding": "埋め込みを計算",
    "Assign Cluster": "クラスタを割り当て", "Mark Tail/Noise": "テール/ノイズをマーク", "Hide noise": "ノイズを非表示", "Unsaved sorting": "未保存の Sorting",
    "All clusters": "全クラスタ", "High to low": "高い順", "Low to high": "低い順", "Target": "対象", "Lag ms": "遅延 ms", "Lag SD ms": "遅延標準偏差 ms", "Match": "一致",
    "A trial is counted as strong response when this post-stimulus count exceeds the non-stimulus/non-burst baseline count mean by more than five standard deviations.": "刺激後カウントが非刺激・非 Burst ベースライン平均を5標準偏差以上上回る試行を強応答と判定します。",
    "Time bin for burst/activity vectors. Typical range: 5-20 ms.": "Burst/活動ベクトルの時間ビンです。一般的な範囲は 5-20 ms です。",
    "Analysis window per burst or per non-overlapping segment. Typical range: 100-500 ms.": "各 Burst または非重複区間の解析窓です。一般的な範囲は 100-500 ms です。",
    "When enabled, drag on the channel map. Left drag selects; right drag removes.": "有効時はチャンネルマップ上でドラッグします。左ドラッグで選択、右ドラッグで解除します。",
    "Remove spikes within a short artifact window after each stimulation event.": "各刺激イベント後の短いアーチファクト窓内のスパイクを除去します。",
    "The current sorting result has not been saved. Save before closing?": "現在の Sorting 結果は未保存です。閉じる前に保存しますか？",
})

_PHRASE_ZH.update({
    "Protocol": "协议", "Block": "Block", "Switch": "切换", "Subgroup": "子组", "Off": "关闭", "On": "开启",
    "Choose the stimulation mode.": "选择刺激模式。",
    "Multiple electrodes": "多电极", "Balanced random groups": "均衡随机电极组", "Sequential groups": "按顺序使用电极组",
    "Use listed groups once": "按列表顺序使用一次", "New seed on save": "每次保存使用新种子", "Fixed seed": "固定种子",
    "Lambda mode": "Lambda 模式", "Custom sequence": "自定义序列", "Base event groups": "基础事件组",
    "Rest only (no recording)": "仅休息（不记录）", "Stimulate": "刺激", "Event subgroups": "事件子组",
    "Pipeline source": "Pipeline 数据源", "Pipeline spontaneous source *": "Pipeline 自发数据源 *",
    "Required. This value is used by the generated stimulation program.": "必填。该值会被生成的刺激程序使用。",
    "Set the experiment identity and output roots used by the generated MaxWell package. Hardware defaults are kept unless you edit them later in the generated config files.": "设置生成的 MaxWell 代码包所使用的实验标识和输出根目录。除非之后在配置文件中修改，否则保留硬件默认值。",
    "Spontaneous data for random stimulation is selected from the current pipeline database in the Stimulus tab.": "随机刺激使用的自发数据从 Stimulus 页面的当前 Pipeline 数据库中选择。",
    "Electrode groups are named sets used by experiment blocks. Defaults come from the current channel map when possible.": "电极组是实验 Block 使用的命名集合；条件允许时，默认值来自当前通道图。",
    "Define stimulation patterns. Random burst uses a selectable inter-burst distribution: Poisson uses lambda, while Uniform uses minimum and maximum intervals. The generated protocol records its complete duration.": "定义刺激模式。随机 Burst 可选择 Burst 间隔分布：Poisson 使用 lambda，Uniform 使用最小和最大间隔。生成的协议会记录完整刺激时长。",
    "Use a loaded spontaneous spike dataset as the rate template for random-electrode stimulation.": "使用已加载的自发 Spike 数据集作为随机电极刺激的发放率模板。",
    "Optional base event groups, one per line or separated by |": "可选基础事件组：每行一组，或使用 | 分隔",
    "Optional base groups: one group per line or separated by |. Counts are balanced from the protocol pulse/burst count.": "可选基础组：每行一组，或使用 | 分隔。数量根据协议中的 Pulse/Burst 总数均衡分配。",
    "Blocks combine one event group, one protocol, and the fixed pre/stim/post phases written into the generated run package.": "Block 将一个事件组、一个协议以及固定的 pre/stim/post 阶段组合后写入生成的运行代码包。",
    "Remove the selected block phase from the phase library.": "从阶段库中移除选中的 Block phase。",
    "Rest only: all three phase durations are quiet recovery waits; no recording or stimulation is performed.": "仅休息：三个阶段时长均为静息恢复等待，不进行记录或刺激。",
    "Configure the protocol and stimulation sites here, then inspect the generated sequence before building the package. Fields marked * are written into the stimulation program.": "在此配置协议和刺激位点，并在构建代码包前检查生成的序列。标有 * 的字段会写入刺激程序。",
    "1. Stimulus protocol": "1. 刺激协议", "Check one or more groups in Site library.": "请在位点库中勾选一个或多个电极组。",
    "Off: all events use selected electrode group": "关闭：所有事件使用同一个选中电极组",
    "On: switch electrode groups per event": "开启：每个事件切换电极组",
    "Checked site groups are added here automatically. Use Up/Down to control sequence order.": "勾选的位点组会自动加入此处；使用上移/下移调整顺序。",
    "Bursts uses detected burst intervals; All data windows splits the recording into consecutive windows.": "Burst 模式使用检测到的 Burst 区间；全部数据窗口模式将记录划分为连续窗口。",
    "FA estimates latent states independently; LDS adds a temporal latent-state model; pi-VAE fits a conditional Poisson VAE.": "FA 独立估计潜在状态；LDS 增加时间潜在状态模型；pi-VAE 拟合条件 Poisson VAE。",
    "Require a channel to participate in at least this many bursts/windows.": "要求通道至少参与指定数量的 Burst/窗口。",
    "Remove channels with nearly constant activity vectors. Typical range: 0-0.1.": "移除活动向量近乎恒定的通道，典型范围为 0-0.1。",
    "Upper bound on fitted channels for speed and numerical stability.": "为保证速度和数值稳定性而设置的拟合通道数上限。",
    "Remove spikes within +/- this range around stimulation artifacts.": "移除刺激伪迹前后该范围内的 Spike。",
    "Run a saved-style workflow with selected files, time range, channels, and one basic function.": "使用选中的文件、时间范围、通道和一个基础功能运行已保存形式的工作流。",
    "Use detected bursts, or split the whole recording into non-overlapping windows.": "使用检测到的 Burst，或将完整记录划分为互不重叠的窗口。",
    "Preprocessing for the state vector before FA fitting.": "FA 拟合前对状态向量进行的预处理。",
    "Model used to predict z(t) from previous latent-state bins.": "使用之前的潜在状态分箱预测 z(t) 的模型。",
    "Number of previous latent time bins used to predict the next latent state.": "预测下一潜在状态时使用的历史时间分箱数量。",
    "Weight for activity-pattern similarity when detecting spatial-temporal regions.": "检测时空区域时活动模式相似度的权重。",
    "Weight for physical distance affinity when detecting spatial-temporal regions.": "检测时空区域时物理距离邻近度的权重。",
    "Higher values keep only channels that are clearly regional members.": "值越高，保留的通道越倾向于明确属于该区域。",
    "Select which burst the comparison views show.": "选择比较视图中显示的 Burst。",
    "Controls the channel ordering in reconstruction views.": "控制重建视图中的通道排列顺序。",
    "Time bin used to convert spikes into channel activity vectors. Typical range: 5-20 ms.": "将 Spike 转换为通道活动向量时使用的时间分箱，典型范围为 5-20 ms。",
    "Analysis window after burst onset, or fixed window size in all-window mode.": "Burst 开始后的分析窗口；在全部窗口模式下表示固定窗口大小。",
    "Choose whether to fit only detected bursts or tile the full recording into non-overlapping windows.": "选择仅拟合检测到的 Burst，或将完整记录划分为互不重叠的窗口。",
    "Preprocessing applied before factor analysis. Per-burst is usually the best first pass.": "因子分析前的预处理。首次分析通常建议使用逐 Burst 模式。",
    "Number of latent factors used by the FA model. Higher values capture more structure but can overfit.": "FA 模型使用的潜在因子数。较大值可捕获更多结构，但可能过拟合。",
    "Temporal model used to explain latent-state evolution across bins.": "用于解释潜在状态随分箱演化的时间模型。",
    "How many previous latent-state bins are used by the temporal model.": "时间模型使用的历史潜在状态分箱数量。",
    "Burst index displayed in the reconstruction views.": "重建视图中显示的 Burst 索引。",
    "Number of grid bins shown in the current raster window. Typical range: 10-120.": "当前 Raster 窗口显示的网格分箱数量，典型范围为 10-120。",
    "Milliseconds represented by each grid bin. Smaller values show finer timing detail.": "每个网格分箱代表的毫秒数；值越小，时间细节越精细。",
    "Time window used to integrate the heatmap around the current playhead. Typical range: 50-300 ms.": "围绕当前播放位置积分 Heatmap 的时间窗，典型范围为 50-300 ms。",
    "Bin size used to compute the average firing-rate curve below the raster.": "计算 Raster 下方平均发放率曲线时使用的分箱大小。",
    "Population spike-count bin used for burst detection. Typical range: 5-20 ms.": "Burst 检测使用的群体 Spike 计数分箱，典型范围为 5-20 ms。",
    "Z-score threshold for calling a burst from the population rate. Higher values are more selective.": "根据群体发放率判定 Burst 的 z-score 阈值；值越高，筛选越严格。",
    "Minimum total spikes required inside a detected burst candidate.": "Burst 候选区间内所需的最小 Spike 总数。",
    "Adjust the display-only parameters here. These settings control how the raster and heatmap are shown, without changing the downstream burst-detection thresholds.": "在此调整仅影响显示的参数。这些设置控制 Raster 和 Heatmap 的显示方式，不改变后续 Burst 检测阈值。",
    "Integration window used to compute each heatmap frame.": "计算每帧 Heatmap 时使用的积分窗口。",
    "Remove spikes shortly after each stimulus marker.": "移除每个刺激标记后短时间内的 Spike。",
    "These parameters define how the population rate is converted into burst intervals. They affect burst overlays here and any burst-based analyses opened from this raster window.": "这些参数定义如何将群体发放率转换为 Burst 区间，并影响当前叠加显示及从此 Raster 窗口打开的 Burst 分析。",
    "Population spike-count bin. Smaller bins preserve onset detail; larger bins smooth the rate.": "群体 Spike 计数分箱。较小分箱保留起始细节，较大分箱使发放率更平滑。",
    "Z-score threshold above baseline population activity. Typical range: 6-10.": "高于群体基线活动的 z-score 阈值，典型范围为 6-10。",
    "Reject burst candidates with too few total spikes.": "剔除 Spike 总数过少的 Burst 候选。",
})

_PHRASE_JA.update({
    "Protocol": "プロトコル", "Block": "ブロック", "Switch": "切り替え", "Subgroup": "サブグループ", "Off": "オフ", "On": "オン",
    "Choose the stimulation mode.": "刺激モードを選択してください。",
    "Multiple electrodes": "複数電極", "Balanced random groups": "均等ランダムグループ", "Sequential groups": "順次グループ",
    "New seed on save": "保存ごとに新しいシード", "Fixed seed": "固定シード", "Custom sequence": "カスタムシーケンス",
    "Base event groups": "基本イベントグループ", "Rest only (no recording)": "休止のみ（記録なし）", "Stimulate": "刺激", "Event subgroups": "イベントサブグループ",
    "Required. This value is used by the generated stimulation program.": "必須です。この値は生成される刺激プログラムで使用されます。",
    "Check one or more groups in Site library.": "サイトライブラリで1つ以上のグループを選択してください。",
    "Off: all events use selected electrode group": "オフ：全イベントで選択した電極グループを使用",
    "On: switch electrode groups per event": "オン：イベントごとに電極グループを切り替え",
    "Checked site groups are added here automatically. Use Up/Down to control sequence order.": "選択したサイトグループは自動的に追加されます。上/下で順序を調整します。",
    "Rest only: all three phase durations are quiet recovery waits; no recording or stimulation is performed.": "休止のみ：3つのフェーズはすべて回復待機で、記録も刺激も行いません。",
    "Remove spikes shortly after each stimulus marker.": "各刺激マーカー直後のスパイクを除去します。",
    "Reject burst candidates with too few total spikes.": "総スパイク数が少なすぎる Burst 候補を除外します。",
})


def _word_fallback(text: str, replacements: dict[str, str]) -> str:
    """Translate common UI words while preserving numbers and domain tokens."""
    result = text
    changed = False
    for source in sorted(replacements, key=len, reverse=True):
        pattern = re.compile(rf"\b{re.escape(source)}\b", flags=re.IGNORECASE)
        result, count = pattern.subn(replacements[source], result)
        changed = changed or count > 0
    return result if changed else text


def translate_text(text: str, language: str) -> str:
    source = str(text or "")
    if not source or language == "en":
        return source
    lowered = source.lower()
    if re.search(r"(?:[a-zA-Z]:\\|/home/|/usr/|\\src\\|/src/|\.py:\d+)", source) or "traceback (most recent call last)" in lowered:
        return source
    dynamic = (_DYNAMIC_ZH if language == "zh_CN" else _DYNAMIC_JA).get(source)
    if dynamic is not None:
        return dynamic
    phrase = (_PHRASE_ZH if language == "zh_CN" else _PHRASE_JA).get(source)
    if phrase is not None:
        return phrase
    catalog = _ZH if language == "zh_CN" else _JA
    translated = catalog.get(source)
    if translated is not None:
        return translated
    base_words = _WORD_ZH if language == "zh_CN" else _WORD_JA
    extra_words = _EXTRA_ZH if language == "zh_CN" else _EXTRA_JA
    if len(source.split()) > 5 or any(mark in source for mark in (".", ";", ":")):
        return source
    return _word_fallback(source, {**base_words, **extra_words})


_TECHNICAL_LOG_PREFIXES = (
    "$ ", "stderr:", "stdout:", "traceback", "file \"", "exec ", "codex", "thinking",
)


def is_technical_log_message(message: str) -> bool:
    """Return True for command output that must remain diagnostically exact."""
    text = str(message or "").strip()
    lowered = text.lower()
    if any(lowered.startswith(prefix) for prefix in _TECHNICAL_LOG_PREFIXES):
        return True
    if "traceback (most recent call last)" in lowered:
        return True
    if re.search(r"(?:[a-zA-Z]:\\|/home/|/usr/|\\src\\|/src/|\.py:\d+)", text):
        return True
    if "module not found" in lowered or "syntaxerror" in lowered or "runtimeerror" in lowered:
        return True
    return False


def _translate_log_template(message: str, language: str) -> str | None:
    if language == "en":
        return None
    templates = (
        (r"^Database loaded:\s*(\d+) files?$", "已加载数据库：{0} 个文件", "データベースを読み込みました：{0} ファイル"),
        (r"^Loaded\s+(\d+) files?$", "已加载 {0} 个文件", "{0} ファイルを読み込みました"),
        (r"^Skipped\s+(\d+) files?(?::\s*(.*))?$", "已跳过 {0} 个文件{1}", "{0} ファイルをスキップしました{1}"),
        (r"^Selected module:\s*(.+)$", "已选择模块：{0}", "選択したモジュール：{0}"),
        (r"^Module saved:\s*(.+)$", "模块已保存：{0}", "モジュールを保存しました：{0}"),
        (r"^Module deleted:\s*(.+)$", "模块已删除：{0}", "モジュールを削除しました：{0}"),
        (r"^Result saved:\s*(.+)$", "结果已保存：{0}", "結果を保存しました：{0}"),
        (r"^Result deleted:\s*(.+)$", "结果已删除：{0}", "結果を削除しました：{0}"),
        (r"^Preparing file database\.?$", "正在准备文件数据库...", "ファイルデータベースを準備中..."),
        (r"^Adding file database rows\.?$", "正在添加文件数据库记录...", "ファイルデータベース行を追加中..."),
        (r"^Selecting loaded file and updating preview\.?$", "正在选择已加载文件并更新预览...", "読み込み済みファイルを選択してプレビューを更新中..."),
        (r"^Refreshing\.?$", "正在刷新...", "更新中..."),
        (r"^Run started\.?$", "运行已开始。", "実行を開始しました。"),
        (r"^Run failed(?::\s*(.*))?$", "运行失败{0}", "実行に失敗しました{0}"),
        (r"^Generation failed(?::\s*(.*))?$", "生成失败{0}", "生成に失敗しました{0}"),
    )
    for pattern, zh_template, ja_template in templates:
        match = re.match(pattern, message, flags=re.IGNORECASE)
        if match is None:
            continue
        values = list(match.groups())
        if pattern.startswith("^Skipped"):
            detail = values[1] if len(values) > 1 else ""
            values[1] = f"：{detail}" if detail else ""
        elif values and pattern.startswith(("^Run failed", "^Generation failed")):
            values[0] = f"：{values[0]}" if values[0] else ""
        template = zh_template if language == "zh_CN" else ja_template
        return template.format(*values)
    return None


def translate_log_message(message: str, language: str) -> str:
    """Translate a user-facing log line while preserving its timestamp."""
    source = str(message or "")
    match = re.match(r"^(\[\d{2}:\d{2}:\d{2}\]\s*)(.*)$", source, flags=re.DOTALL)
    prefix, body = (match.group(1), match.group(2)) if match else ("", source)
    if is_technical_log_message(body):
        return source
    templated = _translate_log_template(body, language)
    if templated is not None:
        return prefix + templated
    return prefix + translate_text(body, language)


class LocalizedLogTextEdit(QTextEdit):
    """QTextEdit that can re-render user-facing log lines after language changes."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._localized_log_entries: list[tuple[str, bool]] = []
        manager = language_manager()
        if manager is not None:
            manager.language_changed.connect(self._retranslate_entries)

    def append(self, text: str) -> None:  # noqa: A003 - Qt API name.
        source = str(text or "")
        localizable = not is_technical_log_message(source)
        self._localized_log_entries.append((source, localizable))
        self._trim_entries()
        super().append(self._render_entry(source, localizable))

    def append_technical(self, text: str) -> None:
        source = str(text or "")
        self._localized_log_entries.append((source, False))
        self._trim_entries()
        super().append(source)

    def clear(self) -> None:
        self._localized_log_entries.clear()
        super().clear()

    def _render_entry(self, source: str, localizable: bool) -> str:
        manager = language_manager()
        language = manager.language if manager is not None else "en"
        return translate_log_message(source, language) if localizable else source

    def _trim_entries(self) -> None:
        maximum = int(self.document().maximumBlockCount())
        if maximum > 0 and len(self._localized_log_entries) > maximum:
            del self._localized_log_entries[:-maximum]

    def _retranslate_entries(self, _language: str) -> None:
        entries = list(self._localized_log_entries)
        QTextEdit.clear(self)
        for source, localizable in entries:
            QTextEdit.append(self, self._render_entry(source, localizable))


class LanguageManager(QObject):
    language_changed = Signal(str)

    def __init__(self, app: QApplication):
        super().__init__(app)
        self.app = app
        self.settings = QSettings("MEA Pipeline", "MEA Pipeline Studio")
        stored = str(self.settings.value("ui/language", "en") or "en")
        self.language = stored if stored in LANGUAGES else "en"
        self.app.installEventFilter(self)
        self._sync_timer = QTimer(self)
        self._sync_timer.setInterval(400)
        self._sync_timer.timeout.connect(self._sync_visible_windows)
        self._sync_timer.start()

    def set_language(self, language: str, *, persist: bool = True) -> None:
        if language not in LANGUAGES:
            language = "en"
        # Capture application-updated English text while the old language is
        # still known, then translate from those stable source strings.
        for widget in self.app.topLevelWidgets():
            self._capture_tree(widget)
        self.language = language
        if persist:
            self.settings.setValue("ui/language", language)
            self.settings.sync()
        for widget in self.app.topLevelWidgets():
            self.translate_widget(widget)
        self.language_changed.emit(language)

    def _sync_visible_windows(self) -> None:
        for widget in self.app.topLevelWidgets():
            if widget.isVisible():
                self.translate_widget(widget)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Show and isinstance(watched, QWidget) and watched.isWindow():
            self.translate_widget(watched)
        return False

    def _recover_source(self, current: str) -> str:
        value = str(current or "")
        for catalog in (_ZH, _JA):
            for source, translated in catalog.items():
                if value == translated:
                    return source
        for catalog in (_DYNAMIC_ZH, _DYNAMIC_JA):
            for source, translated in catalog.items():
                if value == translated:
                    return source
        return value

    def _source(self, obj: QObject, key: str, current: str) -> str:
        property_key = f"_i18n_source_{key}"
        rendered_key = f"_i18n_rendered_{key}"
        cached = obj.property(property_key)
        if cached is None:
            cached = self._recover_source(current)
            obj.setProperty(property_key, cached)
            return str(cached)
        cached = str(cached)
        rendered = obj.property(rendered_key)
        # A later setText/setTitle call is application data, not a translation.
        # Capture it as the new source so dynamic status text cannot remain in
        # the previous language.
        if current and rendered is not None and str(current) != str(rendered) and str(current) != cached:
            cached = self._recover_source(str(current))
            obj.setProperty(property_key, cached)
        return cached

    @staticmethod
    def _set_rendered(obj: QObject, key: str, value: str) -> None:
        obj.setProperty(f"_i18n_rendered_{key}", value)

    def _capture_text(self, obj: QObject, key: str, current: str) -> None:
        self._source(obj, key, current)

    def _capture_tree(self, root: QObject) -> None:
        self._capture_object(root)
        for child in root.findChildren(QObject):
            self._capture_object(child)

    def _capture_object(self, obj: QObject) -> None:
        if bool(obj.property("_i18n_skip")):
            return
        if isinstance(obj, QWidget):
            self._capture_text(obj, "window_title", obj.windowTitle())
        if isinstance(obj, QAction):
            self._capture_text(obj, "text", obj.text())
        if isinstance(obj, QAbstractButton):
            self._capture_text(obj, "text", obj.text())
        elif isinstance(obj, QLabel):
            self._capture_text(obj, "text", obj.text())
        if isinstance(obj, QGroupBox):
            self._capture_text(obj, "title", obj.title())
        if isinstance(obj, QLineEdit):
            self._capture_text(obj, "placeholder", obj.placeholderText())
        elif isinstance(obj, QTextEdit):
            self._capture_text(obj, "placeholder", obj.placeholderText())
        if isinstance(obj, (QWidget, QAction)):
            self._capture_text(obj, "tooltip", obj.toolTip())
        if isinstance(obj, QComboBox):
            self._capture_combo_items(obj)

    def _capture_combo_items(self, combo: QComboBox) -> None:
        current = [combo.itemText(index) for index in range(combo.count())]
        cached = combo.property("_i18n_item_sources")
        rendered = combo.property("_i18n_item_rendered")
        if not isinstance(cached, list) or len(cached) != len(current):
            combo.setProperty("_i18n_item_sources", [self._recover_source(value) for value in current])
            return
        if isinstance(rendered, list):
            sources = list(cached)
            changed = False
            for index, value in enumerate(current):
                if index < len(rendered) and value != rendered[index] and value != sources[index]:
                    sources[index] = self._recover_source(value)
                    changed = True
            if changed:
                combo.setProperty("_i18n_item_sources", sources)

    def translate_widget(self, widget: QWidget) -> None:
        self.translate_object(widget)
        for child in widget.findChildren(QObject):
            self.translate_object(child)

    def translate_object(self, obj: QObject) -> None:
        if bool(obj.property("_i18n_skip")):
            return
        if isinstance(obj, QWidget):
            source = self._source(obj, "window_title", obj.windowTitle())
            if source:
                value = translate_text(source, self.language)
                if obj.windowTitle() != value:
                    obj.setWindowTitle(value)
                self._set_rendered(obj, "window_title", value)
        if isinstance(obj, QAction):
            source = self._source(obj, "text", obj.text())
            if source:
                value = translate_text(source, self.language)
                if obj.text() != value:
                    obj.setText(value)
                self._set_rendered(obj, "text", value)
        if isinstance(obj, QAbstractButton):
            source = self._source(obj, "text", obj.text())
            if source:
                value = translate_text(source, self.language)
                if obj.text() != value:
                    obj.setText(value)
                self._set_rendered(obj, "text", value)
        elif isinstance(obj, QLabel):
            source = self._source(obj, "text", obj.text())
            if source and "<" not in source:
                value = translate_text(source, self.language)
                if obj.text() != value:
                    obj.setText(value)
                self._set_rendered(obj, "text", value)
        if isinstance(obj, QGroupBox):
            source = self._source(obj, "title", obj.title())
            if source:
                value = translate_text(source, self.language)
                if obj.title() != value:
                    obj.setTitle(value)
                self._set_rendered(obj, "title", value)
        if isinstance(obj, QLineEdit):
            source = self._source(obj, "placeholder", obj.placeholderText())
            value = translate_text(source, self.language)
            if obj.placeholderText() != value:
                obj.setPlaceholderText(value)
            self._set_rendered(obj, "placeholder", value)
        elif isinstance(obj, QTextEdit):
            source = self._source(obj, "placeholder", obj.placeholderText())
            value = translate_text(source, self.language)
            if obj.placeholderText() != value:
                obj.setPlaceholderText(value)
            self._set_rendered(obj, "placeholder", value)
        if isinstance(obj, (QWidget, QAction)):
            source = self._source(obj, "tooltip", obj.toolTip())
            value = translate_text(source, self.language)
            if obj.toolTip() != value:
                obj.setToolTip(value)
            self._set_rendered(obj, "tooltip", value)
        if isinstance(obj, QComboBox):
            self._capture_combo_items(obj)
            sources = obj.property("_i18n_item_sources")
            if isinstance(sources, list) and len(sources) == obj.count():
                rendered = []
                for index, source in enumerate(sources):
                    value = translate_text(str(source), self.language)
                    if obj.itemText(index) != value:
                        obj.setItemText(index, value)
                    rendered.append(value)
                obj.setProperty("_i18n_item_rendered", rendered)
        if isinstance(obj, QTabWidget):
            self._translate_tabs(obj)
        if isinstance(obj, QTableWidget):
            self._translate_headers(obj)

    def _translate_tabs(self, tabs: QTabWidget) -> None:
        current = [tabs.tabText(index) for index in range(tabs.count())]
        sources = tabs.property("_i18n_tab_sources")
        rendered = tabs.property("_i18n_tab_rendered")
        if not isinstance(sources, list) or len(sources) != len(current):
            sources = [self._recover_source(value) for value in current]
            tabs.setProperty("_i18n_tab_sources", sources)
        elif isinstance(rendered, list):
            sources = list(sources)
            for index, value in enumerate(current):
                if index < len(rendered) and value != rendered[index] and value != sources[index]:
                    sources[index] = self._recover_source(value)
            tabs.setProperty("_i18n_tab_sources", sources)
        rendered_values = []
        for index, source in enumerate(sources):
            value = translate_text(str(source), self.language)
            if tabs.tabText(index) != value:
                tabs.setTabText(index, value)
            rendered_values.append(value)
        tabs.setProperty("_i18n_tab_rendered", rendered_values)

    def _translate_headers(self, table: QTableWidget) -> None:
        current = [table.horizontalHeaderItem(index).text() if table.horizontalHeaderItem(index) else "" for index in range(table.columnCount())]
        sources = table.property("_i18n_header_sources")
        rendered = table.property("_i18n_header_rendered")
        if not isinstance(sources, list) or len(sources) != len(current):
            sources = [self._recover_source(value) for value in current]
            table.setProperty("_i18n_header_sources", sources)
        elif isinstance(rendered, list):
            sources = list(sources)
            for index, value in enumerate(current):
                if index < len(rendered) and value != rendered[index] and value != sources[index]:
                    sources[index] = self._recover_source(value)
            table.setProperty("_i18n_header_sources", sources)
        rendered_values = []
        for index, source in enumerate(sources):
            item = table.horizontalHeaderItem(index)
            value = translate_text(str(source), self.language)
            if item is not None:
                item.setText(value)
            rendered_values.append(value)
        table.setProperty("_i18n_header_rendered", rendered_values)


_MANAGER: LanguageManager | None = None


def language_manager(app: QApplication | None = None) -> LanguageManager | None:
    global _MANAGER
    application = app or QApplication.instance()
    if application is None:
        return None
    if _MANAGER is None or _MANAGER.app is not application:
        _MANAGER = LanguageManager(application)
    return _MANAGER


def install_language_manager(app: QApplication) -> LanguageManager:
    manager = language_manager(app)
    assert manager is not None
    return manager
