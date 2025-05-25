"""
H-CAAN多智能体系统主入口 - 增强版
"""
import os
import subprocess
import sys
import time
import socket
import signal
import psutil

def check_port(port):
    """检查端口是否被占用"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result == 0

def find_available_port(start_port=8501, max_attempts=10):
    """查找可用端口"""
    for i in range(max_attempts):
        port = start_port + i
        if not check_port(port):
            return port
    return None

def kill_process_on_port(port):
    """杀死占用指定端口的进程"""
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                for conn in proc.connections():
                    if conn.laddr.port == port:
                        print(f"发现进程 {proc.info['name']} (PID: {proc.info['pid']}) 占用端口 {port}")
                        proc.terminate()
                        time.sleep(1)
                        if proc.is_running():
                            proc.kill()
                        print(f"已终止进程 PID: {proc.info['pid']}")
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        print(f"无法终止占用端口的进程: {e}")
    return False

def create_directories():
    """创建必要的目录结构"""
    directories = [
        'data/raw',
        'data/processed', 
        'data/models',
        'data/reports',
        'data/papers'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ 创建目录: {directory}")

def main():
    """主函数 - 智能端口管理和启动"""
    # 创建必要的目录
    create_directories()
    
    print("\n" + "="*50)
    print("启动 H-CAAN 多智能体系统...")
    print("="*50 + "\n")
    
    # 默认端口
    default_port = 8501
    
    # 检查端口是否被占用
    if check_port(default_port):
        print(f"⚠️  端口 {default_port} 已被占用")
        
        # 提供选项
        print("\n请选择操作:")
        print("1. 终止占用端口的进程并使用默认端口")
        print("2. 自动查找可用端口")
        print("3. 手动输入端口号")
        print("4. 退出")
        
        choice = input("\n请输入选项 (1-4): ").strip()
        
        if choice == '1':
            print(f"\n正在终止占用端口 {default_port} 的进程...")
            if kill_process_on_port(default_port):
                time.sleep(2)
                port = default_port
            else:
                print("无法终止进程，将查找新端口...")
                port = find_available_port(default_port + 1)
        elif choice == '2':
            port = find_available_port(default_port)
            if port:
                print(f"\n找到可用端口: {port}")
            else:
                print("未找到可用端口")
                sys.exit(1)
        elif choice == '3':
            try:
                port = int(input("请输入端口号 (如 8502): "))
                if check_port(port):
                    print(f"端口 {port} 已被占用")
                    sys.exit(1)
            except ValueError:
                print("无效的端口号")
                sys.exit(1)
        else:
            print("退出程序")
            sys.exit(0)
    else:
        port = default_port
    
    # Streamlit启动命令
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "streamlit_ui/Home.py",
        "--server.port", str(port),
        "--server.address", "localhost",
        "--server.maxUploadSize", "200",
        "--theme.primaryColor", "#FF6B6B",
        "--theme.backgroundColor", "#FFFFFF", 
        "--theme.secondaryBackgroundColor", "#F0F2F6",
        "--theme.textColor", "#262730"
    ]
    
    try:
        # 使用subprocess启动Streamlit
        print(f"\n正在端口 {port} 上启动Streamlit服务器...")
        print(f"命令: {' '.join(cmd)}")
        print(f"\n✅ 访问 http://localhost:{port} 使用系统")
        print("按 Ctrl+C 停止服务器\n")
        
        # 启动进程
        process = subprocess.Popen(cmd)
        
        # 等待进程结束
        process.wait()
        
    except KeyboardInterrupt:
        print("\n\n正在关闭服务器...")
        process.terminate()
        time.sleep(1)
        if process.poll() is None:
            process.kill()
        print("服务器已关闭")
        
    except Exception as e:
        print(f"启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()