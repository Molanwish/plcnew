import tkinter as tk
from tkinter import ttk

def main():
    root = tk.Tk()
    root.title("按钮测试")
    root.geometry("800x600")
    
    # 主框架
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # 配置网格
    main_frame.grid_columnconfigure(0, weight=1)
    main_frame.grid_rowconfigure(0, weight=1)  # 参数区域
    main_frame.grid_rowconfigure(1, weight=0)  # 按钮区域
    
    # 参数区域 - 用一个简单的标签代替
    param_area = ttk.LabelFrame(main_frame, text="参数区域")
    param_area.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
    
    # 按钮区域框架
    button_frame = ttk.Frame(main_frame, relief="groove", borderwidth=1)
    button_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
    
    # 配置按钮区域
    button_frame.grid_columnconfigure(0, weight=1)
    button_frame.grid_columnconfigure(1, weight=1)
    button_frame.grid_columnconfigure(2, weight=1)
    
    # PLC操作按钮组
    plc_button_frame = ttk.LabelFrame(button_frame, text="PLC操作")
    plc_button_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
    
    # 文件操作按钮组
    file_button_frame = ttk.LabelFrame(button_frame, text="文件操作")
    file_button_frame.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
    
    # 其他操作按钮组
    other_button_frame = ttk.LabelFrame(button_frame, text="其他操作")
    other_button_frame.grid(row=0, column=2, padx=10, pady=5, sticky="ew")
    
    # 配置每个按钮框架的列
    plc_button_frame.grid_columnconfigure(0, weight=1)
    plc_button_frame.grid_columnconfigure(1, weight=1)
    
    file_button_frame.grid_columnconfigure(0, weight=1)
    file_button_frame.grid_columnconfigure(1, weight=1)
    
    other_button_frame.grid_columnconfigure(0, weight=1)
    
    # 创建按钮
    read_button = ttk.Button(plc_button_frame, text="读取PLC参数")
    write_button = ttk.Button(plc_button_frame, text="写入参数到PLC")
    
    save_button = ttk.Button(file_button_frame, text="保存参数到文件")
    load_button = ttk.Button(file_button_frame, text="从文件加载参数")
    
    reset_button = ttk.Button(other_button_frame, text="重置参数")
    
    # 放置按钮
    read_button.grid(row=0, column=0, padx=12, pady=8, sticky="ew")
    write_button.grid(row=0, column=1, padx=12, pady=8, sticky="ew")
    
    save_button.grid(row=0, column=0, padx=12, pady=8, sticky="ew")
    load_button.grid(row=0, column=1, padx=12, pady=8, sticky="ew")
    
    reset_button.grid(row=0, column=0, padx=12, pady=8, sticky="ew")
    
    # 显示窗口
    root.mainloop()

if __name__ == "__main__":
    main() 