# Linux 系统异常检测工具

2025年全国大学生计算机系统能力大赛 - 操作系统设计赛 - 西北区域赛

[![Python](https://img.shields.io/badge/Python-3.6%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux-lightgrey)](https://www.kernel.org/)

---

## 所选题目

- **ID:** proj407
- **内容：** 一个功能完善的 Linux 系统日志异常检测工具，支持实时监控、统计分析和多种异常类型检测。
- **演示视频：** [链接](#)
- **项目设计文档：** [链接](#)
---

## 项目信息

**项目名称：** 操作系统异常信息检测工具

**比赛：** 2025年全国大学生计算机系统能力大赛 - 操作系统设计赛 - 西北区域赛

**参赛队伍名称：** 地表最速考拉

**参赛队伍编号：** #待填写

**队伍成员：** 程安冉，官瑞琪，易明秀

**指导老师：** 刘刚（andyliu@lzu.edu.cn）

---
##  目录

- [功能特性](#功能特性)
- [支持的异常类型](#支持的异常类型)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [使用说明](#使用说明)
  - [测试模式](#测试模式)
  - [生产模式](#生产模式)
  - [系统服务](#系统服务)
- [配置文件](#配置文件)
- [输出示例](#输出示例)
- [常见问题](#常见问题)
- [项目结构](#项目结构)

---
## 项目结构
```
linux-anomaly-detector/
project-root/
├── config/
│   ├── config.yaml                         # 默认配置文件
│   ├── test_deadlock_config.yaml           # 死锁检测配置
│   ├── test_fs_exception_config.yaml       # 文件系统异常配置
│   ├── test_high_frequency_config.yaml     # 高频异常配置
│   ├── test_mixed_anomalies_only_config.yaml # 仅异常日志混合配置
│   ├── test_mixed_with_normal_config.yaml  # 正常与异常混合配置
│   ├── test_oom_config.yaml                # 内存不足（OOM）配置
│   ├── test_oops_config.yaml               # 内核错误（Oops）配置
│   ├── test_panic_config.yaml              # 系统崩溃（Panic）配置
│   ├── test_reboot_config.yaml             # 系统重启配置
├── src/
│   ├── anomaly_detector.py                 # 异常检测工具主程序
│   ├── generate_test_logs.py               # 生成测试日志工具
│
├──  test_logs/                   # 测试日志目录（运行后生成）
|   ├── test_oom.log
|   ├── test_oops.log
|   ├── test_panic.log
|   ├── test_deadlock.log
|   ├── test_reboot.log
|   ├── test_fs_exception.log
|   ├── test_mixed_with_normal.log
|   ├── test_mixed_anomalies_only.log
|   └── test_high_frequency.log
|── install.sh                              # 安装脚本
|── 项目设计文档.md                          # 项目功能实现与分析
├── README.md                               # 项目说明文档
├── requirements.txt                        # 项目依赖文件
```

---

## 功能特性

-  **双模式运行**
   - **测试模式**：使用预生成的测试日志，无需 root 权限，适合演示和功能验证
    - **生产模式**：扫描真实系统日志，需要 root 权限，用于实际监控

-  **6 种异常检测**
    - OOM (内存不足)
    - Oops (内核错误)
    - Panic (系统崩溃)
    - Deadlock (进程死锁)
    - Reboot (系统重启)
    - FS Exception (文件系统异常)

-  **实时监控**
    - 守护进程模式持续监控
    - 可配置扫描间隔
    - 支持 Ctrl+C 退出

-  **统计分析**
    - 按异常类型统计
    - 显示频率和占比
    - 自动生成统计报告

-  **多格式导出**
    - JSON 格式（机器可读）
    - YAML 格式（配置友好）
    - TEXT 格式（人类可读）
    - HTML 格式（可读性高）

-  **systemd 集成**
    - 支持作为系统服务运行
    - 开机自启动
    - 崩溃自动重启
    - 统一日志管理

-  **灵活配置**
    - YAML 配置文件
    - 自定义检测规则
    - 正则表达式支持

---

## 支持的异常类型

| 类型 | 说明 | 严重级别 | 检测关键词 |
|------|------|---------|-----------|
| **OOM** | 内存不足导致进程被杀 | Critical | `Out of memory`, `oom-killer` |
| **Oops** | 内核错误但未崩溃 | Critical | `kernel BUG`, `Oops:` |
| **Panic** | 内核无法恢复的错误 | Critical | `Kernel panic` |
| **Deadlock** | 进程长时间阻塞 | High | `blocked for more than` |
| **Reboot** | 系统重启事件 | Warning | `system is going down` |
| **FS Exception** | 文件系统错误 | High | `EXT4-fs error`, `I/O error` |

---

## 环境要求

### 操作系统
- **推荐**：Ubuntu 22.04 LTS
- **兼容**：Debian 10+, CentOS 7+, RHEL 7+

### 软件依赖
- **Python**：3.6 或更高版本
- **PyYAML**：用于配置文件解析

### 权限要求
- **测试模式**：普通用户权限
- **生产模式**：需要 root 权限（读取系统日志）

---

## 快速开始

### 1. 克隆项目
```bash
git clone -b v2 https://gitlab.eduxiji.net/T202510730997675/project3006612-352109.git
cd project3006612-352109
```

### 2. 运行安装脚本
```bash
# 运行安装脚本（需要 root 权限）
sudo bash install.sh
```
### 3.测试系统命令

安装后可以直接使用 `anomaly-detector` 命令：
```bash
# 测试模式
anomaly-detector --test-mode --scan

# 生产模式
sudo anomaly-detector --production-mode --scan

# 查看帮助
anomaly-detector --help
```


## 使用说明

### 测试模式

测试模式使用预生成的测试日志文件，**无需 root 权限**，适合：
- 功能演示
- 测试验证

#### 生成测试日志
```bash
python3 src/generate_test_logs.py
```

这会在 `test_logs/` 目录生成 9 个测试日志文件。

#### 基本功能
##### 1.扫描日志文件+自动生成统计报告
```bash
# 扫描所有测试日志
python3 src/anomaly_detector.py --test-mode --scan

# 按照配置文件扫描日志
python3 src/anomaly_detector.py --test-mode --config config/test_deadlock_config.yaml --scan

```
##### 2.实时监控
```bash

# 实时监控（每 5 秒扫描一次）
python3 src/anomaly_detector.py --test-mode --monitor

# 自定义监控间隔（10 秒）
python3 src/anomaly_detector.py --test-mode --monitor --interval 10

# 按照配置文件实时监控
python3 src/anomaly_detector.py --test-mode --config config/test_deadlock_config.yaml --monitor 

# 按 Ctrl+C 停止监控,自动生成统计报告
```

##### 3.导出报告
```bash
# JSON 格式
python3 src/anomaly_detector.py --test-mode --scan --export test_report.json

# YAML 格式
python3 src/anomaly_detector.py --test-mode --scan --export test_report.yaml --format yaml

# 文本格式
python3 src/anomaly_detector.py --test-mode --scan --export test_report.txt --format text

# HTML格式
python3 src/anomaly_detector.py --test-mode --scan --export test_report.html --format html

# 查看报告
cat test_report.json
# 查看HTML报告
python3 -m http.server 8000
#  获取虚拟机IP地址
ifconfig
# 虚拟机所在的物理机浏览器上执行
http://<虚拟机的IP地址>:8000/test_report.html

```

---

### 生产模式

生产模式默认扫描真实系统日志，**需要 sudo 权限**，用于：
- 实际系统监控
- 生产环境部署
- 异常排查
- 安全审计

#### 基本功能
##### 1.扫描日志文件+自动生成统计报告
```bash
# 扫描真实系统日志
sudo python3 src/anomaly_detector.py --scan
或
sudo python3 src/anomaly_detector.py --production-mode --scan

```
##### 2.实时监控
```bash
# 实时监控系统日志
sudo python3 src/anomaly_detector.py [--production-mode] --monitor
或
sudo python3 src/anomaly_detector.py [--production-mode] --monitor

# 自定义监控间隔（3 秒）
sudo python3 src/anomaly_detector.py --monitor --interval 3
或
sudo python3 src/anomaly_detector.py --production-mode --monitor --interval 3

# 使用自定义配置文件
sudo python3 src/anomaly_detector.py --config <自定义配置文件> --scan
或
sudo python3 src/anomaly_detector.py --config <自定义配置文件> --production-mode --scan

```
##### 3.导出系统报告
```bash
# 扫描并导出json格式
sudo python3 src/anomaly_detector.py --scan --export system_report.json
或
sudo python3 src/anomaly_detector.py --production-mode --scan --export system_report.json

# YAML 格式
sudo python3 src/anomaly_detector.py --scan --export system_report.yaml --format yaml
或
sudo python3 src/anomaly_detector.py --production-mode --scan --export system_report.yaml --format yaml

# 文本格式
sudo python3 src/anomaly_detector.py --scan --export system_report.txt --format text
或
sudo python3 src/anomaly_detector.py --production-mode --scan --export system_report.txt --format text

# HTML格式
sudo python3 src/anomaly_detector.py --scan --export system_report.html --format html
或
sudo python3 src/anomaly_detector.py --production-mode --scan --export system_report.html --format html

# 查看报告
cat system_report.json
或
cat system_report.yaml
或
cat system_report.txt

# 查看HTML报告
python3 -m http.server 8000
#  获取虚拟机IP地址
ifconfig
# 虚拟机所在的物理机浏览器上执行
http://<虚拟机的IP地址>:8000/system_report.html
```

#### 监控的日志文件

生产模式默认监控以下系统日志：
- `/var/log/syslog` (Ubuntu/Debian)
- `/var/log/messages` (CentOS/RHEL)
- `/var/log/kern.log` (内核日志)
---

### 系统服务

将工具安装为 systemd 服务，支持开机自启和后台运行。


#### 服务管理命令
```bash
# 启动服务（生产模式运行）
sudo systemctl start anomaly-detector

# 停止服务
sudo systemctl stop anomaly-detector

# 重启服务
sudo systemctl restart anomaly-detector

# 查看服务状态
sudo systemctl status anomaly-detector

# 设置开机自启
sudo systemctl enable anomaly-detector

# 取消开机自启
sudo systemctl disable anomaly-detector
```

#### 查看服务日志
```bash
# 方法 1：直接查看日志文件
sudo tail -f /var/log/anomaly-detector/anomaly-detector.log

# 方法 2：使用 journalctl（推荐）
sudo journalctl -u anomaly-detector -f

# 查看最近 50 条日志
sudo journalctl -u anomaly-detector -n 50

# 查看今天的日志
sudo journalctl -u anomaly-detector --since today

# 查看错误日志
sudo journalctl -u anomaly-detector -p err
```
---

## 配置文件

配置文件使用 YAML 格式，支持自定义日志路径和检测规则。

### 生成默认配置
```bash
python3 anomaly_detector.py --generate-config
```

### 配置文件示例
```yaml
# Linux 系统异常检测工具 - 配置文件

# 日志文件源（生产模式使用）
log_sources:
  - /var/log/syslog
  - /var/log/messages
  - /var/log/kern.log

# 监控间隔（秒）
monitor_interval: 5

# 检测规则配置
rules:
  # OOM (内存不足)
  oom:
    patterns:
      - 'Out of memory'
      - 'Kill process \d+ \(.*\) score'
      - 'oom-killer'
      - 'Killed process \d+'
    severity: critical
    description: 内存不足(OOM)
  
  # Oops (内核错误)
  oops:
    patterns:
      - 'kernel BUG at'
      - 'Oops:'
      - 'general protection fault'
      - 'unable to handle kernel'
    severity: critical
    description: 内核错误(Oops)
  
  # Panic (系统崩溃)
  panic:
    patterns:
      - 'Kernel panic'
      - 'kernel panic - not syncing'
      - 'panic occurred'
    severity: critical
    description: 系统崩溃(Panic)
  
  # Deadlock (进程死锁)
  deadlock:
    patterns:
      - 'possible deadlock'
      - 'INFO: task .* blocked for more than'
      - 'hung_task_timeout_secs'
      - 'Task .* blocked for more than \d+ seconds'
    severity: high
    description: 进程死锁
  
  # Reboot (系统重启)
  reboot:
    patterns:
      - 'system is going down'
      - 'reboot: machine restart'
      - 'systemd.*Shutting down'
      - 'kernel: \[.*\] Linux version'
      - 'systemd-shutdown.*Sending SIGTERM'
    severity: warning
    description: 系统重启
  
  # FS Exception (文件系统异常)
  fs_exception:
    patterns:
      - 'EXT4-fs error'
      - 'XFS.*error'
      - 'I/O error.*dev'
      - 'Read-only file system'
      - 'filesystem read-only'
      - 'Remounting filesystem read-only'
      - 'filesystem error'
      - 'BTRFS error'
    severity: high
    description: 文件系统异常
```

### 使用自定义配置
```bash
# 生产模式
sudo python3 src/anomaly_detector.py --config config/config.yaml --production-mode --scan

# 测试模式
#  运行 test_deadlock_config.yaml 配置文件
python3 src/anomaly_detector.py --test-mode --config config/test_deadlock_config.yaml --scan

```

---

## 输出示例

### 扫描输出
```
开始扫描日志...

检测到 4 个异常事件

最近的异常事件:
[2025-12-30T12:38:21.292736] [HIGH] DEADLOCK
  来源: test_logs/test_deadlock.log
  消息: Dec 29 08:45:12 server kernel: [8765.432109] INFO: task mysqld:1234 blocked for more than 120 seconds....

[2025-12-30T12:38:21.293007] [HIGH] DEADLOCK
  来源: test_logs/test_deadlock.log
  消息: Dec 29 08:45:13 server kernel: [8765.432110] "echo 0 > /proc/sys/kernel/hung_task_timeout_secs" disables this message....

[2025-12-30T12:38:21.293022] [HIGH] DEADLOCK
  来源: test_logs/test_deadlock.log
  消息: Dec 29 10:30:20 server kernel: [12345.098765] possible deadlock detected in ext4_da_write_begin...

[2025-12-30T12:38:21.293210] [HIGH] DEADLOCK
  来源: test_logs/test_deadlock.log
  消息: Dec 29 17:20:30 server kernel: [21234.567890] Task apache2:5678 blocked for more than 180 seconds...


============================================================
异常统计报告
============================================================
总异常事件数: 4

异常类型                 描述                   次数         占比        
------------------------------------------------------------
deadlock             进程死锁                 4          100.0%

============================================================

```

### JSON 报告格式
```json
{
  "mode": "test",
  "total_events": 4,
  "statistics": {
    "deadlock": 4
  },
  "events": [
    {
      "timestamp": "2025-12-30T12:39:57.455501",
      "event_type": "deadlock",
      "severity": "high",
      "message": "Dec 29 08:45:12 server kernel: [8765.432109] INFO: task mysqld:1234 blocked for more than 120 seconds.",
      "source": "test_logs/test_deadlock.log"
    },
    {
      "timestamp": "2025-12-30T12:39:57.455630",
      "event_type": "deadlock",
      "severity": "high",
      "message": "Dec 29 08:45:13 server kernel: [8765.432110] \"echo 0 > /proc/sys/kernel/hung_task_timeout_secs\" disables this message.",
      "source": "test_logs/test_deadlock.log"
    },
    {
      "timestamp": "2025-12-30T12:39:57.455642",
      "event_type": "deadlock",
      "severity": "high",
      "message": "Dec 29 10:30:20 server kernel: [12345.098765] possible deadlock detected in ext4_da_write_begin",
      "source": "test_logs/test_deadlock.log"
    },
    {
      "timestamp": "2025-12-30T12:39:57.455937",
      "event_type": "deadlock",
      "severity": "high",
      "message": "Dec 29 17:20:30 server kernel: [21234.567890] Task apache2:5678 blocked for more than 180 seconds",
      "source": "test_logs/test_deadlock.log"
    }
  ]
}
```

---

## 常见问题

### Q1: 提示 "Permission denied" 权限不足？

**原因**：生产模式需要 root 权限读取系统日志。

**解决**：
```bash
sudo python3 src/anomaly_detector.py --production-mode --scan
```

---

### Q2: 测试模式提示找不到日志文件？

**原因**：测试日志未生成。

**解决**：
```bash
python3 src/generate_test_logs.py
ls test_logs/  # 验证是否生成
```

---

### Q3: 提示 "No module named 'yaml'"？

**原因**：PyYAML 未安装。

**解决**：
```bash
# 方法 1
pip3 install pyyaml

# 方法 2
sudo apt-get install python3-yaml
```

---

### Q4: 生产模式未检测到任何异常是正常的吗？

**是的！** 如果系统运行正常，没有异常事件是完全正常的。

可以用测试模式验证程序是否正常工作：
```bash
python3 src/generate_test_logs.py
python3 src/anomaly_detector.py --test-mode --scan
```

---

### Q5: 如何查看服务是否在运行？
```bash
# 方法 1：查看服务状态
sudo systemctl status anomaly-detector

# 方法 2：查看进程
ps aux | grep anomaly_detector

# 方法 3：查看日志文件
sudo tail -f /var/log/anomaly-detector/anomaly-detector.log
```

---

### Q6: 如何卸载系统服务？
```bash
# 停止并禁用服务
sudo systemctl stop anomaly-detector
sudo systemctl disable anomaly-detector

# 删除服务文件
sudo rm /etc/systemd/system/anomaly-detector.service
sudo systemctl daemon-reload

# 删除程序文件（可选）
sudo rm -rf /opt/anomaly-detector
sudo rm -rf /etc/anomaly-detector
sudo rm -rf /var/log/anomaly-detector
sudo rm /usr/local/bin/anomaly-detector
```
---

## 命令速查表

| 操作 | 测试模式 | 生产模式 |
|------|---------|---------|
| 扫描日志 | `python3 src/anomaly_detector.py --test-mode --scan` | `sudo python3 src/anomaly_detector.py --production-mode --scan` |
| 实时监控 | `python3 src/anomaly_detector.py --test-mode --monitor` | `sudo python3 src/anomaly_detector.py --production-mode --monitor` |
| 导出报告 | `python3 src/anomaly_detector.py --test-mode --scan --export report.json` | `sudo python3 src/anomaly_detector.py --production-mode --scan --export report.json` |
| 自定义间隔 | `--monitor --interval 10` | `--monitor --interval 10` |
| 查看帮助 | `python3 src/anomaly_detector.py --help` | `python3 src/anomaly_detector.py --help` |

---

## 演示视频录制建议

推荐的演示流程：
```bash
# 1. 展示项目文件
ls -lh

# 2. 生成测试日志
python3 src/src/generate_test_logs.py

# 3. 测试模式扫描
python3 src/src/anomaly_detector.py --test-mode --scan

# 4. 查看统计报告（已包含在扫描输出中）

# 5. 实时监控演示（运行几秒后 Ctrl+C）
python3 src/src/anomaly_detector.py --test-mode --monitor

# 6. 导出报告
python3 src/src/anomaly_detector.py --test-mode --scan --export demo.json
cat demo.json | head -30

# 7. 生产模式演示（可选）
sudo python3 src/src/anomaly_detector.py --production-mode --scan
```

---


### 开发环境设置
```bash
https://gitlab.eduxiji.net/T202510730997675/project3006612-352109
cd linux-anomaly-detector
pip3 install -r requirements.txt
python3 src/generate_test_logs.py
python3 src/anomaly_detector.py --test-mode --scan
```

---



## 联系方式

- **项目地址**：https://gitlab.eduxiji.net/T202510730997675/project3006612-352109
- **问题反馈**：https://gitlab.eduxiji.net/T202510730997675/project3006612-352109/-/issues

---
