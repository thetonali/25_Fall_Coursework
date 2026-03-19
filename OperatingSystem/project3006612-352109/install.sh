#!/bin/bash
#
# Linux 系统异常检测工具安装脚本
# 用途: 自动安装工具到系统
#

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印函数
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否为 root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        print_error "此脚本需要 root 权限运行"
        echo "请使用: sudo bash install.sh"
        exit 1
    fi
}

# 检查 Python 版本
check_python() {
    print_info "检查 Python 版本..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "未找到 Python3，请先安装 Python 3.6+"
        exit 1
    fi
    
    python_version=$(python3 --version | cut -d' ' -f2)
    print_info "Python 版本: $python_version"
}

# 安装依赖
install_dependencies() {
    print_info "安装依赖包..."
    
    # 尝试安装 PyYAML
    if python3 -c "import yaml" 2>/dev/null; then
        print_info "PyYAML 已安装"
    else
        print_warn "PyYAML 未安装，正在安装..."
        
        # 尝试使用 pip
        if command -v pip3 &> /dev/null; then
            pip3 install pyyaml
        else
            # 使用系统包管理器
            if command -v apt-get &> /dev/null; then
                apt-get update
                apt-get install -y python3-yaml
            elif command -v yum &> /dev/null; then
                yum install -y python3-pyyaml
            else
                print_error "无法自动安装 PyYAML，请手动安装"
                exit 1
            fi
        fi
    fi
     # 安装 dos2unix
    if ! command -v dos2unix &> /dev/null; then
        print_warn "未找到 dos2unix，正在安装..."
        sudo apt-get install -y dos2unix
    else
        print_info "dos2unix 已安装"
    fi
    
    print_info "依赖安装完成"
}

# 创建安装目录
create_directories() {
    print_info "创建安装目录..."
    
    INSTALL_DIR="/opt/anomaly-detector"
    CONFIG_DIR="/etc/anomaly-detector"
    LOG_DIR="/var/log/anomaly-detector"
    
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$CONFIG_DIR"
    mkdir -p "$LOG_DIR"
    
    print_info "目录创建完成"
}

# 复制文件
copy_files() {
    print_info "复制程序文件..."
    
    # 复制主程序
    cp src/anomaly_detector.py "$INSTALL_DIR/"
    chmod +x "$INSTALL_DIR/anomaly_detector.py"
    
    # 复制配置文件
    if [ -f "config/config.yaml" ]; then
        cp config/config.yaml "$CONFIG_DIR/config.yaml.example"
        if [ ! -f "$CONFIG_DIR/config.yaml" ]; then
            cp config/config.yaml "$CONFIG_DIR/config.yaml"
        fi
    fi
        # 修复换行符问题
    dos2unix "$INSTALL_DIR/anomaly_detector.py"
    
    print_info "文件复制完成"
}

# 创建命令行快捷方式
create_symlink() {
    print_info "创建命令行快捷方式..."
    
    ln -sf "$INSTALL_DIR/anomaly_detector.py" /usr/local/bin/anomaly-detector
    
    print_info "快捷方式创建完成"
}

# 创建 systemd 服务
create_systemd_service() {
    print_info "创建 systemd 服务..."
    
    cat > /etc/systemd/system/anomaly-detector.service << EOF
[Unit]
Description=System Anomaly Detector
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$INSTALL_DIR
ExecStart=/usr/bin/python3 $INSTALL_DIR/anomaly_detector.py --monitor --config $CONFIG_DIR/config.yaml  # 确保配置文件路径正确
Restart=always
RestartSec=10
StandardOutput=append:$LOG_DIR/anomaly-detector.log
StandardError=append:$LOG_DIR/anomaly-detector.error.log

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    
    print_info "systemd 服务创建完成"
}

# 显示安装后信息
show_post_install_info() {
    echo ""
    echo "================================================================"
    print_info "安装完成！"
    echo "================================================================"
    echo ""
    echo "安装位置:"
    echo "  程序目录: $INSTALL_DIR"
    echo "  配置目录: $CONFIG_DIR"
    echo "  日志目录: $LOG_DIR"
    echo ""
    echo "使用方法:"
    echo "  1. 扫描日志:     anomaly-detector --scan"
    echo "  2. 实时监控:     anomaly-detector --monitor"
    echo "  3. 查看帮助:     anomaly-detector --help"
    echo ""
    echo "作为服务运行:"
    echo "  启动服务:       systemctl start anomaly-detector"
    echo "  停止服务:       systemctl stop anomaly-detector"
    echo "  开机自启:       systemctl enable anomaly-detector"
    echo "  查看状态:       systemctl status anomaly-detector"
    echo "  查看日志:       journalctl -u anomaly-detector -f"
    echo ""
    echo "配置文件: $CONFIG_DIR/config.yaml"
    echo "================================================================"
}

# 主函数
main() {
    echo "================================================================"
    echo "    Linux 系统异常检测工具 - 安装程序"
    echo "================================================================"
    echo ""
    
    check_root
    check_python
    install_dependencies
    create_directories
    copy_files
    create_symlink
    create_systemd_service
    show_post_install_info
}

# 运行主函数
main
