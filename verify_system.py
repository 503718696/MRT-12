#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MRT-12 系统完整性验证脚本

Author: 罗兵 (Luo Bing)
Email: <2712179753@qq.com>
WeChat: 18368870543
Douyin: 1918705950
"""

import os
import sys
import importlib.util


def check_python_version():
    """检查 Python 版本"""
    print("🔍 检查 Python 版本...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python 版本：{version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python 版本过低：{version.major}.{version.minor}.{version.micro}")
        return False


def check_required_packages():
    """检查必需的 Python 包"""
    print("\n🔍 检查必需的 Python 包...")
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'), 
        ('torchaudio', 'TorchAudio'),
        ('numpy', 'NumPy'),
        ('psutil', 'PSUtil')
    ]
    
    all_good = True
    for package, name in required_packages:
        try:
            spec = importlib.util.find_spec(package)
            if spec is not None:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                print(f"✅ {name}: {version}")
            else:
                print(f"❌ {name}: 未安装")
                all_good = False
        except Exception as e:
            print(f"❌ {name}: 导入失败 - {e}")
            all_good = False
    
    return all_good


def check_gpu_availability():
    """检查 GPU 可用性"""
    print("\n🔍 检查 GPU 可用性...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ CUDA 可用，检测到 {gpu_count} 个 GPU:")
            for i in range(gpu_count):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {name} ({memory:.1f}GB)")
            return True
        else:
            print("⚠️  CUDA 不可用，将使用 CPU 训练")
            return False
    except Exception as e:
        print(f"❌ GPU 检查失败：{e}")
        return False


def check_project_structure():
    """检查项目目录结构"""
    print("\n🔍 检查项目结构...")
    required_files = [
        'core/__init__.py',
        'core/manifold_ops.py', 
        'core/model_mrt12.py',
        'data/__init__.py',
        'data/dataset.py',
        'utils/__init__.py',
        'utils/logger.py',
        'utils/checkpoint.py',
        'train_foundation.py',
        'tune_logic.py',
        'evaluate.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (缺失)")
            missing_files.append(file_path)
    
    return len(missing_files) == 0


def main():
    """主验证函数"""
    print("="*60)
    print("MRT-12 系统完整性验证")
    print("="*60)
    
    checks = [
        ("Python 版本", check_python_version()),
        ("依赖包", check_required_packages()),
        ("GPU 可用性", check_gpu_availability()),
        ("项目结构", check_project_structure())
    ]
    
    print("\n" + "="*60)
    print("验证总结")
    print("="*60)
    
    all_passed = all(result for _, result in checks)
    
    for name, result in checks:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
    
    if all_passed:
        print("\n🎉 所有检查通过！系统已就绪")
    else:
        print("\n⚠️  部分检查未通过，请修复后重试")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)