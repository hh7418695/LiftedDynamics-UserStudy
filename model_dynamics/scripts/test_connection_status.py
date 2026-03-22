"""
test_connection_status.py - 实时显示 UDP 连接状态和设备检测

Windows 版本 - 用于检测真实触觉设备连接

使用方法 / Usage:
    python test_connection_status.py

功能 / Features:
1. 显示 UDP 端口绑定状态
2. 实时显示接收到的数据包
3. 检测触笔按钮按下
4. 显示位置和力数据
"""

import socket
import json
import time
import sys
from datetime import datetime

SERVER_IP = "127.0.0.1"  # Windows 本地
SERVER_PORT = 12312
BUFFER_SIZE = 512

def print_header():
    print("=" * 70)
    print("触觉设备连接测试 / Haptic Device Connection Test")
    print("=" * 70)
    print(f"时间 / Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"服务器 / Server: {SERVER_IP}:{SERVER_PORT}")
    print("=" * 70)

def print_status(message, status="INFO"):
    timestamp = datetime.now().strftime('%H:%M:%S')
    symbols = {
        "INFO": "ℹ",
        "SUCCESS": "✓",
        "ERROR": "✗",
        "WAITING": "⏳",
        "DATA": "📊"
    }
    symbol = symbols.get(status, "•")
    print(f"[{timestamp}] {symbol} {message}")

def main():
    print_header()

    # 1. 尝试绑定 UDP socket
    print_status("正在绑定 UDP socket... / Binding UDP socket...", "INFO")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((SERVER_IP, SERVER_PORT))
        print_status(f"UDP socket 绑定成功 / Successfully bound to {SERVER_IP}:{SERVER_PORT}", "SUCCESS")
    except Exception as e:
        print_status(f"UDP socket 绑定失败 / Bind failed: {e}", "ERROR")
        print("\n可能的原因 / Possible reasons:")
        print("  1. 端口已被占用 / Port already in use")
        print("  2. 权限不足 / Insufficient permissions")
        print("  3. 防火墙阻止 / Firewall blocking")
        input("\n按 Enter 退出 / Press Enter to exit...")
        return

    print("\n" + "=" * 70)
    print_status("等待 C++ 客户端连接... / Waiting for C++ client...", "WAITING")
    print("=" * 70)
    print("\n操作步骤 / Instructions:")
    print("  1. 启动 C++ 程序 (udp_client.exe)")
    print("  2. 按下触笔按钮开始发送数据 / Press stylus button to start")
    print("  3. 观察下方数据流 / Watch data stream below")
    print("  4. 按 Ctrl+C 退出 / Press Ctrl+C to exit")
    print("\n" + "-" * 70 + "\n")

    packet_count = 0
    start_time = time.time()
    last_packet_time = start_time
    button_pressed = False

    try:
        sock.settimeout(3.0)  # 3秒超时，用于显示等待状态

        while True:
            try:
                data_byte, address = sock.recvfrom(BUFFER_SIZE)
                current_time = time.time()
                packet_count += 1

                # 第一个包到达
                if packet_count == 1:
                    print_status(f"检测到 C++ 客户端！/ C++ client detected! 来自 From: {address}", "SUCCESS")
                    button_pressed = True

                # 解析数据
                try:
                    data_json = data_byte.decode('utf-8')
                    data_dict = json.loads(data_json)
                    position = data_dict.get('position', [])
                    timestamp = data_dict.get('timestamp', 0)

                    # 检测停止信号
                    if timestamp < 0:
                        print_status("收到停止信号 / Stop signal received", "SUCCESS")
                        print_status("触笔按钮已释放 / Stylus button released", "INFO")
                        button_pressed = False
                        print("\n" + "-" * 70)
                        print_status("等待下一次按钮按下... / Waiting for next button press...", "WAITING")
                        print("-" * 70 + "\n")
                        packet_count = 0
                        continue

                    # 显示数据（每10个包显示一次，避免刷屏）
                    if packet_count % 10 == 0:
                        elapsed = current_time - start_time
                        rate = packet_count / elapsed if elapsed > 0 else 0
                        print_status(
                            f"包 Packet #{packet_count} | "
                            f"位置 Pos: [{position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}] mm | "
                            f"速率 Rate: {rate:.1f} Hz",
                            "DATA"
                        )

                    last_packet_time = current_time

                except json.JSONDecodeError as e:
                    print_status(f"数据解析失败 / Parse failed: {e}", "ERROR")

            except socket.timeout:
                elapsed = time.time() - start_time
                if packet_count == 0:
                    # 还没收到任何数据
                    print_status(
                        f"仍在等待... / Still waiting... ({int(elapsed)}s) "
                        "请确认 C++ 程序正在运行 / Please confirm C++ program is running",
                        "WAITING"
                    )
                else:
                    # 收到过数据但现在超时了
                    idle_time = time.time() - last_packet_time
                    if idle_time > 5:
                        print_status(
                            f"已 {int(idle_time)}秒 无数据 / No data for {int(idle_time)}s. "
                            "C++ 程序可能已停止 / C++ may have stopped",
                            "WAITING"
                        )

    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print_status("用户中断 / User interrupted", "INFO")

    finally:
        sock.close()
        elapsed = time.time() - start_time
        print("=" * 70)
        print(f"\n统计信息 / Statistics:")
        print(f"  总运行时间 / Total time: {elapsed:.1f}s")
        print(f"  收到数据包 / Packets received: {packet_count}")
        if elapsed > 0 and packet_count > 0:
            print(f"  平均速率 / Average rate: {packet_count/elapsed:.1f} Hz")
        print(f"\nUDP socket 已关闭 / Socket closed")
        print("=" * 70)
        input("\n按 Enter 退出 / Press Enter to exit...")

if __name__ == "__main__":
    main()
