"""Colours and the Qt style sheet every page of the app shares."""

THEME = {
    "bg": "#f3f4f7",
    "card": "#ffffff",
    "ink": "#1f2933",
    "muted": "#7b8794",
    "line": "#e1e5ea",
    "accent": "#3b6fd4",
    "raw": "#6b7785",
    "inhale": "#2f9e63",
    "exhale": "#d9534f",
    "span": "#f0a53a",
    "keep": "#1a9fa0",
    "bad": "#d9534f",
    "placed": "#0a7a28",
    "auto": "#999999",
}
SERIES = ("#3b6fd4", "#d9534f", "#2f9e63", "#8e5bd0", "#f0a53a", "#1a9fa0")
"""Line colours, in order, for plots with several series."""

STYLE = f"""
QWidget {{ background: {THEME["bg"]}; color: {THEME["ink"]}; font-family: "Segoe UI"; font-size: 13px; }}
QFrame#card {{ background: {THEME["card"]}; border-radius: 6px; }}
QFrame#card QLabel {{ background: transparent; }}
QLabel#title {{ color: {THEME["muted"]}; font-size: 9px; }}
QLabel#value {{ font-size: 19px; font-weight: 600; }}
QLabel#sub {{ color: {THEME["muted"]}; font-size: 10px; }}
QLabel#heading {{ font-size: 17px; font-weight: 600; }}
QLabel#bad {{ color: {THEME["bad"]}; }}
QListWidget, QTableView, QTreeView, QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox,
QPlainTextEdit {{
    background: {THEME["card"]}; border: 1px solid {THEME["line"]}; border-radius: 4px; }}
QPlainTextEdit {{ font-family: "Consolas", "Menlo", monospace; font-size: 11px; }}
QListWidget, QTableView, QTreeView {{ font-size: 12px; }}
QListWidget::item {{ padding: 1px 4px; }}
QTableView::item {{ padding: 0px 4px; }}
QListWidget::item:selected, QTableView::item:selected, QTreeView::item:selected {{
    background: {THEME["accent"]}; color: white; }}
QListWidget#nav {{ background: {THEME["card"]}; border: none; font-size: 14px; }}
QListWidget#nav::item {{ padding: 10px 14px; }}
QHeaderView::section {{ background: {THEME["bg"]}; border: none; padding: 3px; color: {THEME["muted"]}; }}
QPushButton {{ background: {THEME["line"]}; border: none; border-radius: 4px; padding: 6px 14px; }}
QPushButton:hover {{ background: #d3d9e0; }}
QPushButton#accent {{ background: {THEME["accent"]}; color: white; font-weight: 600; }}
QPushButton#accent:hover {{ background: #2f5bb5; }}
QPushButton:disabled {{ color: {THEME["muted"]}; }}
QSplitter::handle {{ background: {THEME["line"]}; }}
QGroupBox {{ border: 1px solid {THEME["line"]}; border-radius: 6px; margin-top: 14px; padding: 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; color: {THEME["muted"]}; }}
"""
