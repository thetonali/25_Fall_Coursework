#!/usr/bin/env python3
# 测试日志生成器
# 用于生成包含多种系统异常的模拟日志文件，配合异常检测工具进行测试

import os
from datetime import datetime, timedelta
import random

# 测试日志生成类
class TestLogGenerator:
    
    # 各种异常的示例日志
    SAMPLE_LOGS = {
        'oom': [
            'Dec 29 10:23:45 server kernel: [12345.678901] Out of memory: Kill process 1234 (chrome) score 856 or sacrifice child',
            'Dec 29 10:23:46 server kernel: [12345.678902] Killed process 1234 (chrome) total-vm:4123456kB, anon-rss:2345678kB, file-rss:0kB',
            'Dec 29 11:15:30 server kernel: [13456.789012] oom-killer: gfp_mask=0x6200ca invoked oom-killer',
            'Dec 29 14:20:15 server kernel: [18765.432109] Memory cgroup out of memory: Kill process 5678 (java)',
        ],
        'oops': [
            'Dec 29 12:34:56 server kernel: [15678.901234] BUG: unable to handle kernel NULL pointer dereference at 0000000000000008',
            'Dec 29 12:34:57 server kernel: [15678.901235] Oops: 0002 [#1] SMP',
            'Dec 29 13:45:20 server kernel: [16789.012345] kernel BUG at fs/ext4/inode.c:1234!',
            'Dec 29 15:22:10 server kernel: [19234.567890] general protection fault: 0000 [#1] PREEMPT SMP',
        ],
        'panic': [
            'Dec 29 09:15:30 server kernel: [10234.567890] Kernel panic - not syncing: Fatal exception',
            'Dec 29 09:15:31 server kernel: [10234.567891] Kernel panic - not syncing: Attempted to kill init!',
            'Dec 29 16:30:45 server kernel: [20345.678901] panic occurred, switching back to text console',
        ],
        'deadlock': [
            'Dec 29 08:45:12 server kernel: [8765.432109] INFO: task mysqld:1234 blocked for more than 120 seconds.',
            'Dec 29 08:45:13 server kernel: [8765.432110] "echo 0 > /proc/sys/kernel/hung_task_timeout_secs" disables this message.',
            'Dec 29 10:30:20 server kernel: [12345.098765] possible deadlock detected in ext4_da_write_begin',
            'Dec 29 17:20:30 server kernel: [21234.567890] Task apache2:5678 blocked for more than 180 seconds',
        ],
        'reboot': [
            'Dec 29 06:00:00 server systemd[1]: Shutting down.',
            'Dec 29 06:00:01 server kernel: [5432.109876] reboot: machine restart',
            'Dec 29 06:00:10 server kernel: [0.000000] Linux version 5.15.0-91-generic (buildd@lcy02-amd64-010)',
            'Dec 29 18:45:00 server systemd-shutdown[1]: Sending SIGTERM to remaining processes...',
        ],
        'fs_exception': [
            'Dec 29 11:30:45 server kernel: [13678.901234] EXT4-fs error (device sda1): ext4_find_entry:1234: inode #12345: comm apache2: reading directory lblock 0',
            'Dec 29 11:30:46 server kernel: [13678.901235] EXT4-fs (sda1): Remounting filesystem read-only',
            'Dec 29 14:15:20 server kernel: [17890.123456] I/O error, dev sda1, sector 123456789',
            'Dec 29 15:45:30 server kernel: [19345.678901] XFS (sdb1): metadata I/O error: block 0x123456 ("xlog_iodone") error 5',
            'Dec 29 16:20:10 server kernel: [20123.456789] BTRFS error (device sdc1): bdev /dev/sdc1 errs: wr 1, rd 0, flush 0, corrupt 0, gen 0',
        ],
        'normal': [
            'Dec 29 09:00:00 server systemd[1]: Started Daily apt download activities.',
            'Dec 29 09:30:15 server cron[1234]: (root) CMD (test -x /usr/sbin/anacron || ( cd / && run-parts --report /etc/cron.daily ))',
            'Dec 29 10:00:00 server sshd[5678]: Accepted publickey for user from 192.168.1.100 port 54321 ssh2',
            'Dec 29 11:00:00 server systemd[1]: Starting Cleanup of Temporary Directories...',
            'Dec 29 12:00:00 server kernel: [14567.890123] usb 2-1: new high-speed USB device number 5 using xhci_hcd',
        ]
    }
    # 初始化测试日志生成器
    def __init__(self, output_dir='test_logs'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    # 生成单一类型的测试日志文件 
    def generate_single_log(self, log_type, filename='test_syslog.log'):
  
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            if log_type in self.SAMPLE_LOGS:
                for log in self.SAMPLE_LOGS[log_type]:
                    f.write(log + '\n')
                print(f"✓ 生成 {log_type} 测试日志: {filepath}")
            else:
                print(f"✗ 未知日志类型: {log_type}")
        
        return filepath
    # 生成包含多种异常的混合测试日志
    def generate_mixed_log(self, filename='test_mixed.log', include_normal=True):
        filepath = os.path.join(self.output_dir, filename)
        
        all_logs = []
        
        # 收集所有异常日志
        for log_type, logs in self.SAMPLE_LOGS.items():
            if log_type == 'normal' and not include_normal:
                continue
            all_logs.extend(logs)
        
        # 打乱顺序
        random.shuffle(all_logs)
        
        with open(filepath, 'w') as f:
            for log in all_logs:
                f.write(log + '\n')
        
        print(f"✓ 生成混合测试日志: {filepath} (共 {len(all_logs)} 条)")
        return filepath
    # 生成完整的测试日志套件
    def generate_complete_test_suite(self):
        print("="*60)
        print("生成测试日志套件")
        print("="*60)
        
        # 1. 为每种异常类型生成单独的日志文件
        for log_type in self.SAMPLE_LOGS.keys():
            if log_type != 'normal':
                self.generate_single_log(log_type, f'test_{log_type}.log')
        
        # 2. 生成混合日志
        self.generate_mixed_log('test_mixed_with_normal.log', include_normal=True)
        self.generate_mixed_log('test_mixed_anomalies_only.log', include_normal=False)
        
        # 3. 生成高频异常日志
        self.generate_high_frequency_log()
        
        print("="*60)
        print(f"所有测试日志已生成到目录: {self.output_dir}/")
        print("="*60)

      # 生成高频异常日志，用于测试统计功能
    def generate_high_frequency_log(self, filename='test_high_frequency.log'):
        filepath = os.path.join(self.output_dir, filename)
        
        logs = []
        # 生成多个 OOM 事件
        for i in range(5):
            logs.append(f'Dec 29 10:{20+i}:00 server kernel: Out of memory: Kill process {1000+i} (app{i})')
        
        # 生成多个文件系统错误
        for i in range(3):
            logs.append(f'Dec 29 11:{10+i}:00 server kernel: EXT4-fs error (device sda1): error {i}')
        
        # 生成多个死锁
        for i in range(2):
            logs.append(f'Dec 29 12:{15+i}:00 server kernel: INFO: task process{i}:1234 blocked for more than 120 seconds.')
        
        with open(filepath, 'w') as f:
            for log in logs:
                f.write(log + '\n')
        
        print(f"✓ 生成高频测试日志: {filepath}")
        return filepath

def main():
    generator = TestLogGenerator()
    generator.generate_complete_test_suite()
    
    print("\n使用方法:")
    print("1. 测试单个异常类型:")
    print("   python3 anomaly_detector.py --scan --config config.yaml")
    print("   # 修改 config.yaml 中的 log_sources 指向 test_logs/test_oom.log")
    print()
    print("2. 测试混合日志:")
    print("   python3 test_anomaly_detector.py")
    print()
    print("3. 手动测试:")
    print("   cat test_logs/test_oom.log")

if __name__ == '__main__':
    main()

