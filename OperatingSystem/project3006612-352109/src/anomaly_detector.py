#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linux 系统异常检测工具
支持测试模式和生产模式
"""

import re
import json
import yaml
import argparse
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import signal
import sys

@dataclass
class AnomalyEvent:
    """异常事件数据类"""
    timestamp: str
    event_type: str
    severity: str
    message: str
    source: str
    
    def to_dict(self):
        return asdict(self)

class AnomalyDetector:
    """系统异常检测器"""
    
    # 默认检测规则
    DEFAULT_RULES = {
        'oom': {
            'patterns': [
                r'Out of memory',
                r'Kill process \d+ \(.*\) score',
                r'oom-killer',
                r'Killed process \d+'
            ],
            'severity': 'critical',
            'description': '内存不足(OOM)'
        },
        'oops': {
            'patterns': [
                r'kernel BUG at',
                r'Oops:',
                r'general protection fault',
                r'unable to handle kernel'
            ],
            'severity': 'critical',
            'description': '内核错误(Oops)'
        },
        'panic': {
            'patterns': [
                r'Kernel panic',
                r'kernel panic - not syncing',
                r'panic occurred'
            ],
            'severity': 'critical',
            'description': '系统崩溃(Panic)'
        },
        'deadlock': {
            'patterns': [
                r'possible deadlock',
                r'INFO: task .* blocked for more than',
                r'hung_task_timeout_secs',                
                r'Task .* blocked for more than \d+ seconds'
            ],
            'severity': 'high',
            'description': '进程死锁'
        },
        'reboot': {
            'patterns': [
                r'system is going down',
                r'reboot: machine restart',
                r'systemd.*Shutting down',
                r'kernel: \[.*\] Linux version',
                r'systemd-shutdown.*Sending SIGTERM'                 
            ],
            'severity': 'warning',
            'description': '系统重启'
        },
        'fs_exception': {
            'patterns': [
                r'EXT4-fs error',
                r'XFS.*error',
                r'I/O error.*dev',
                r'Read-only file system',
                r'filesystem read-only',                  
                r'Remounting filesystem read-only',        
                r'filesystem error',
                r'BTRFS error'
            ],
            'severity': 'high',
            'description': '文件系统异常'
        }
    }
    
    # 生产环境日志路径
    PRODUCTION_LOGS = [
        '/var/log/syslog',
        '/var/log/messages',
        '/var/log/kern.log'
    ]
    
    # 测试环境日志路径
    TEST_LOGS = [
        'test_logs/test_oom.log',
            'test_logs/test_oops.log',
            'test_logs/test_panic.log',
            'test_logs/test_deadlock.log',
            'test_logs/test_reboot.log',
            'test_logs/test_fs_exception.log',
            'test_logs/test_mixed_with_normal.log',
            'test_logs/test_mixed_anomalies_only.log',
            'test_logs/test_high_frequency.log'
    ]
    
    def __init__(self, mode='production', config_file: Optional[str] = None):
        """
        初始化检测器
        
        参数:
            mode: 运行模式 ('test' 或 'production')
            config_file: 配置文件路径（可选）
        """
        self.mode = mode
        
        # 加载配置（如果提供）
        if config_file:
            self.config = self._load_config(config_file)
            self.rules = self.config.get('rules', self.DEFAULT_RULES)
            self.log_sources = self.config.get('log_sources', [])
        else:
            self.rules = self.DEFAULT_RULES
            self.log_sources = []
        
        # 如果没有指定日志源，根据模式选择默认日志
        if not self.log_sources:
            if mode == 'test':
                self.log_sources = self.TEST_LOGS
                print(f"运行模式: 测试模式")
                print(f"使用测试日志: {len(self.TEST_LOGS)} 个文件")
            else:
                self.log_sources = self.PRODUCTION_LOGS
                print(f"运行模式: 生产模式")
                print(f"监控系统日志: {', '.join(self.PRODUCTION_LOGS)}")
        
        self.events: List[AnomalyEvent] = []
        self.statistics = defaultdict(int)
        self.last_positions = {}
        self.running = False
    
    def _load_config(self, config_file: str) -> Dict:
        """加载配置文件"""
        config_path = Path(config_file)
        if not config_path.exists():
            print(f"警告: 配置文件 {config_file} 不存在，使用默认配置")
            return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif config_path.suffix == '.json':
                    return json.load(f)
        except Exception as e:
            print(f"警告: 加载配置文件失败 - {e}")
            return {}
    
    def _check_log_exists(self, log_path: str) -> bool:
        """检查日志文件是否存在"""
        path = Path(log_path)
        exists = path.exists() and path.is_file()
        if not exists and self.mode == 'test':
            print(f"警告: 测试日志 {log_path} 不存在")
            print(f"提示: 请先运行 'python3 generate_test_logs.py' 生成测试日志")
        return exists
    
    def _read_log_lines(self, log_path: str, from_position: int = 0) -> tuple:
        """读取日志文件"""
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(from_position)
                lines = f.readlines()
                new_position = f.tell()
                return lines, new_position
        except PermissionError:
            if self.mode == 'production':
                print(f"错误: 没有权限读取 {log_path}")
                print(f"提示: 请使用 'sudo python3 anomaly_detector.py --scan'")
            return [], from_position
        except Exception as e:
            print(f"错误: 读取日志文件 {log_path} 失败 - {e}")
            return [], from_position
    
    def _match_patterns(self, line: str, patterns: List[str]) -> bool:
        """匹配日志行与规则模式"""
        for pattern in patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True
        return False
    
    def _parse_log_line(self, line: str, source: str) -> Optional[AnomalyEvent]:
        """解析日志行，检测异常"""
        for event_type, rule in self.rules.items():
            if self._match_patterns(line, rule['patterns']):
                event = AnomalyEvent(
                    timestamp=datetime.now().isoformat(),
                    event_type=event_type,
                    severity=rule['severity'],
                    message=line.strip(),
                    source=source
                )
                return event
        return None
    
    def scan_logs(self, real_time: bool = False) -> List[AnomalyEvent]:
        """扫描日志文件"""
        new_events = []
        
        for log_path in self.log_sources:
            if not self._check_log_exists(log_path):
                continue
            
            from_pos = self.last_positions.get(log_path, 0)
            lines, new_pos = self._read_log_lines(log_path, from_pos)
            
            for line in lines:
                event = self._parse_log_line(line, log_path)
                if event:
                    new_events.append(event)
                    self.events.append(event)
                    self.statistics[event.event_type] += 1
                    
                    if real_time:
                        self._print_event(event)
            
            self.last_positions[log_path] = new_pos
        
        return new_events
    
    def _print_event(self, event: AnomalyEvent):
        """打印异常事件"""
        colors = {
            'critical': '\033[91m',
            'high': '\033[93m',
            'warning': '\033[94m',
        }
        reset = '\033[0m'
        
        color = colors.get(event.severity, '')
        print(f"{color}[{event.timestamp}] [{event.severity.upper()}] "
              f"{event.event_type.upper()}{reset}")
        print(f"  来源: {event.source}")
        print(f"  消息: {event.message[:200]}...")
        print()
    
    def monitor_daemon(self, interval: int = 5):
        """守护进程模式：实时监控"""
        print(f"\n启动实时监控 (每 {interval} 秒扫描一次)")
        print(f"监控日志文件: {', '.join([Path(p).name for p in self.log_sources])}")
        print("按 Ctrl+C 停止监控\n")
        
        self.running = True
        
        def signal_handler(sig, frame):
            print("\n\n停止监控...")
            self.running = False
            self.print_statistics()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # 初始扫描
        self.scan_logs(real_time=False)
        
        while self.running:
            time.sleep(interval)
            self.scan_logs(real_time=True)
    
    def print_statistics(self):
        """打印统计报告"""
        print("\n" + "="*60)
        print("异常统计报告")
        print("="*60)
        
        if not self.statistics:
            print("未检测到任何异常事件")
            return
        
        total = sum(self.statistics.values())
        print(f"总异常事件数: {total}\n")
        
        print(f"{'异常类型':<20} {'描述':<20} {'次数':<10} {'占比':<10}")
        print("-"*60)
        
        for event_type, count in sorted(self.statistics.items(), 
                                       key=lambda x: x[1], reverse=True):
            desc = self.rules.get(event_type, {}).get('description', event_type)
            percentage = (count / total) * 100
            print(f"{event_type:<20} {desc:<20} {count:<10} {percentage:.1f}%")
        
        print("\n" + "="*60)
    
    def export_events(self, output_file: str, format: str = 'json'):
        """导出异常事件"""
        output_path = Path(output_file)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                if format == 'json':
                    data = {
                        'mode': self.mode,
                        'total_events': len(self.events),
                        'statistics': dict(self.statistics),
                        'events': [e.to_dict() for e in self.events]
                    }
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                elif format == 'yaml':
                    data = {
                        'mode': self.mode,
                        'total_events': len(self.events),
                        'statistics': dict(self.statistics),
                        'events': [e.to_dict() for e in self.events]
                    }
                    yaml.dump(data, f, allow_unicode=True)
                
                elif format == 'text':
                    f.write(f"系统异常检测报告\n")
                    f.write(f"运行模式: {self.mode}\n")
                    f.write(f"生成时间: {datetime.now()}\n")
                    f.write(f"总事件数: {len(self.events)}\n\n")
                    
                    for event in self.events:
                        f.write(f"[{event.timestamp}] {event.event_type.upper()}\n")
                        f.write(f"  严重程度: {event.severity}\n")
                        f.write(f"  来源: {event.source}\n")
                        f.write(f"  消息: {event.message}\n\n")
                
                elif format == 'html':
                    # 生成 HTML 报告
                    f.write(self._generate_html_report())
            
            # 只在导出时显示简单提示
            print(f"\n✓ 已导出报告到 {output_file}")
        except Exception as e:
            print(f"导出失败: {e}")

    def _generate_html_report(self):
        """生成 HTML 格式报告"""
        total = sum(self.statistics.values())
        
        # 颜色映射
        severity_colors = {
            'critical': '#dc3545',
            'high': '#ffc107',
            'warning': '#17a2b8'
        }
        
        html = f"""<!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>系统异常检测报告</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ 
                font-family: 'Segoe UI', Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                line-height: 1.6;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            .header h1 {{ font-size: 32px; margin-bottom: 10px; }}
            .header p {{ opacity: 0.9; font-size: 14px; }}
            .summary {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                padding: 30px;
                background: #f8f9fa;
            }}
            .summary-card {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                text-align: center;
                transition: transform 0.3s;
            }}
            .summary-card:hover {{ transform: translateY(-5px); }}
            .summary-card .number {{
                font-size: 36px;
                font-weight: bold;
                color: #667eea;
                margin-bottom: 5px;
            }}
            .summary-card .label {{
                color: #6c757d;
                font-size: 14px;
            }}
            .statistics {{
                padding: 30px;
            }}
            .statistics h2 {{
                color: #333;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 3px solid #667eea;
            }}
            .stat-item {{
                display: flex;
                align-items: center;
                padding: 15px;
                margin-bottom: 10px;
                background: #f8f9fa;
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }}
            .stat-label {{
                flex: 1;
                font-weight: 600;
                color: #333;
            }}
            .stat-count {{
                font-size: 24px;
                font-weight: bold;
                color: #667eea;
                margin: 0 20px;
            }}
            .stat-bar {{
                flex: 2;
                height: 30px;
                background: #e9ecef;
                border-radius: 15px;
                overflow: hidden;
                position: relative;
            }}
            .stat-bar-fill {{
                height: 100%;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                display: flex;
                align-items: center;
                justify-content: flex-end;
                padding-right: 10px;
                color: white;
                font-size: 12px;
                font-weight: bold;
                transition: width 1s ease;
            }}
            .events {{
                padding: 30px;
                background: #f8f9fa;
            }}
            .events h2 {{
                color: #333;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 3px solid #667eea;
            }}
            .event-item {{
                background: white;
                padding: 20px;
                margin-bottom: 15px;
                border-radius: 8px;
                border-left: 4px solid #667eea;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            .event-header {{
                display: flex;
                align-items: center;
                margin-bottom: 10px;
            }}
            .event-type {{
                font-weight: bold;
                font-size: 16px;
                text-transform: uppercase;
                margin-right: 10px;
            }}
            .event-severity {{
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: bold;
                color: white;
            }}
            .severity-critical {{ background-color: #dc3545; }}
            .severity-high {{ background-color: #ffc107; color: #333; }}
            .severity-warning {{ background-color: #17a2b8; }}
            .event-time {{
                color: #6c757d;
                font-size: 13px;
                margin-left: auto;
            }}
            .event-source {{
                color: #6c757d;
                font-size: 13px;
                margin-bottom: 8px;
            }}
            .event-message {{
                background: #f8f9fa;
                padding: 12px;
                border-radius: 6px;
                font-family: monospace;
                font-size: 13px;
                color: #333;
                word-break: break-all;
            }}
            .footer {{
                text-align: center;
                padding: 20px;
                color: #6c757d;
                font-size: 13px;
                border-top: 1px solid #dee2e6;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 系统异常检测报告</h1>
                <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>运行模式: {self.mode.upper()}</p>
            </div>
            
            <div class="summary">
                <div class="summary-card">
                    <div class="number">{len(self.events)}</div>
                    <div class="label">检测到的异常事件</div>
                </div>
                <div class="summary-card">
                    <div class="number">{len(self.statistics)}</div>
                    <div class="label">异常类型数量</div>
                </div>
                <div class="summary-card">
                    <div class="number">{len(self.log_sources)}</div>
                    <div class="label">扫描的日志文件</div>
                </div>
            </div>
            
            <div class="statistics">
                <h2>📊 异常统计分析</h2>
    """
        
        # 添加统计数据
        for event_type, count in sorted(self.statistics.items(), key=lambda x: x[1], reverse=True):
            desc = self.rules.get(event_type, {}).get('description', event_type)
            percentage = (count / total) * 100 if total > 0 else 0
            
            html += f"""
                <div class="stat-item">
                    <div class="stat-label">{desc}</div>
                    <div class="stat-count">{count}</div>
                    <div class="stat-bar">
                        <div class="stat-bar-fill" style="width: {percentage}%">
                            {percentage:.1f}%
                        </div>
                    </div>
                </div>
    """
        
        html += """
            </div>
            
            <div class="events">
                <h2>📋 详细事件列表</h2>
    """
        
        # 添加事件列表
        for event in self.events:
            severity_class = f"severity-{event.severity}"
            html += f"""
                <div class="event-item">
                    <div class="event-header">
                        <div class="event-type">{event.event_type}</div>
                        <span class="event-severity {severity_class}">{event.severity.upper()}</span>
                        <div class="event-time">{event.timestamp}</div>
                    </div>
                    <div class="event-source">📁 {event.source}</div>
                    <div class="event-message">{event.message}</div>
                </div>
    """
        
        html += """
            </div>
            
            <div class="footer">
                <p>Linux 系统异常检测工具 | Made with ❤️ for Linux System Administrators</p>
            </div>
        </div>
    </body>
    </html>
    """
        return html

def generate_default_config(output_file: str = 'config.yaml'):
    """生成默认配置文件"""
    config = {
        'log_sources': [
            '/var/log/syslog',
            '/var/log/messages',
            '/var/log/kern.log'
        ],
        'monitor_interval': 5,
        'rules': AnomalyDetector.DEFAULT_RULES
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    print(f"已生成默认配置文件: {output_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Linux 系统异常检测工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
运行模式:
  --test-mode          使用测试日志（无需 sudo）
  --production-mode    扫描真实系统日志（需要 sudo）

示例用法:
  # 测试模式（演示用）
  python3 anomaly_detector.py --test-mode --scan
  python3 anomaly_detector.py --test-mode --monitor

  # 生产模式（实际使用）
  sudo python3 anomaly_detector.py --production-mode --scan
  sudo python3 anomaly_detector.py --production-mode --monitor

  # 导出报告
  python3 anomaly_detector.py --test-mode --scan --export report.json

  # 使用自定义配置
  sudo python3 anomaly_detector.py --config config.yaml --scan

  # 生成默认配置
  python3 anomaly_detector.py --generate-config
        """
    )
    
    # 运行模式选择
    parser.add_argument('--test-mode', action='store_true',
                       help='测试模式（使用测试日志）')
    parser.add_argument('--production-mode', action='store_true',
                       help='生产模式（扫描真实系统日志）')
    
    # 功能选项
    parser.add_argument('--config', '-c', help='配置文件路径')
    parser.add_argument('--scan', '-s', action='store_true',
                       help='扫描日志文件')
    parser.add_argument('--monitor', '-m', action='store_true',
                       help='实时监控模式')
    parser.add_argument('--interval', '-i', type=int, default=5,
                       help='监控间隔（秒），默认 5 秒')
    parser.add_argument('--export', '-e', help='导出结果到文件')
    parser.add_argument('--format', '-f', 
                       choices=['json', 'yaml', 'text', 'html'],
                       default='json',
                       help='导出格式')
    parser.add_argument('--generate-config', action='store_true',
                       help='生成默认配置文件')
    parser.add_argument('--stats', action='store_true',
                       help='显示统计信息')
    
    args = parser.parse_args()
    
    # 生成配置文件
    if args.generate_config:
        generate_default_config()
        return
    
    # 检查是否有操作
    if not (args.scan or args.monitor or args.stats):
        parser.print_help()
        return
    
    # 确定运行模式
    if args.test_mode:
        mode = 'test'
    else:
        # 如果没有指定测试模式，默认使用生产模式
        mode = 'production'
    
    # 创建检测器
    detector = AnomalyDetector(mode=mode, config_file=args.config)
    
    # 监控模式
    if args.monitor:
        detector.monitor_daemon(args.interval)
        return
    
    # 扫描模式
    if args.scan:
        print("\n开始扫描日志...")
        events = detector.scan_logs()
        print(f"\n检测到 {len(events)} 个异常事件")
        
        if events:
            print("\n最近的异常事件:")
            for event in events[-10:]:
                detector._print_event(event)
    
    # 显示统计
    if args.stats or args.scan:
        detector.print_statistics()
    
    # 导出结果
    if args.export:
        detector.export_events(args.export, args.format)

if __name__ == '__main__':
    main()
