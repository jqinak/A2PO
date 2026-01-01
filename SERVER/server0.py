import os
import sys
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
import traceback
import resource

app = Flask(__name__)

# 配置参数
ALLOWED_PORT = 1234
BASE_DIR = "/project/peilab/qjl/CODE/SERVER/tmp"
READ_ONLY_PATHS = ["/usr", "/lib", "/bin", "/etc", os.path.expanduser("~")]  # 只读路径

# 确保工作目录存在
os.makedirs(BASE_DIR, exist_ok=True)

def setup_sandbox_environment(temp_dir):
    """设置沙盒环境限制"""
    # 设置资源限制
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))  # 2秒CPU时间
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))  # 256MB内存
    resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))  # 10MB文件大小
    resource.setrlimit(resource.RLIMIT_NPROC, (20, 20))  # 最多20个进程
    
    # 更改工作目录
    os.chdir(temp_dir)
    
    # 设置环境变量
    os.environ['PYTHONPATH'] = temp_dir
    os.environ['HOME'] = temp_dir

def is_write_allowed(path):
    """检查是否允许写入路径"""
    path = os.path.abspath(path)
    base_dir = os.path.abspath(BASE_DIR)
    
    # 检查是否在允许写入的目录或其子目录中
    if path.startswith(base_dir + os.sep) or path == base_dir:
        return True
    
    # 检查是否在只读路径中
    for read_only_path in READ_ONLY_PATHS:
        read_only_path_abs = os.path.abspath(read_only_path)
        if path.startswith(read_only_path_abs + os.sep) or path == read_only_path_abs:
            return False
    
    # 默认不允许写入
    return False

def safe_open(path, mode='r', *args, **kwargs):
    """安全的文件打开函数"""
    if 'w' in mode or 'a' in mode or 'x' in mode:
        if not is_write_allowed(path):
            raise PermissionError(f"不允许写入路径: {path}")
    
    # 对于读取，检查文件是否存在且可读
    if 'r' in mode and not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    
    return open(path, mode, *args, **kwargs)

def run_python_code(code, temp_dir):
    """在沙盒环境中运行Python代码"""
    # 创建临时文件
    temp_file = os.path.join(temp_dir, "user_code.py")
    
    try:
        # 写入用户代码
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(code)
        
        # 准备运行环境
        env = os.environ.copy()
        env['PYTHONPATH'] = temp_dir
        
        # 重写内置的open函数
        code_wrapper = f'''
import sys
import os
import builtins

# 保存原始的open函数
_original_open = builtins.open

def _safe_open(path, mode='r', *args, **kwargs):
    """安全的文件打开函数"""
    path = os.path.abspath(path)
    base_dir = "{BASE_DIR}"
    
    # 检查是否允许写入
    if 'w' in mode or 'a' in mode or 'x' in mode:
        if not (path.startswith(base_dir + os.sep) or path == base_dir):
            raise PermissionError(f"不允许写入路径: {{path}}")
    
    return _original_open(path, mode, *args, **kwargs)

# 替换内置的open函数
builtins.open = _safe_open

# 导入用户代码
with open("{temp_file}", 'r', encoding='utf-8') as f:
    user_code = f.read()

# 执行用户代码
exec(compile(user_code, "{temp_file}", 'exec'))
'''
        
        # 创建包装脚本
        wrapper_file = os.path.join(temp_dir, "wrapper.py")
        with open(wrapper_file, 'w', encoding='utf-8') as f:
            f.write(code_wrapper)
        
        # 运行代码
        result = subprocess.run(
            [sys.executable, wrapper_file],
            capture_output=True,
            text=True,
            timeout=5,  # 5秒超时
            cwd=temp_dir,
            env=env
        )
        
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
        
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": "代码执行超时（超过5秒）",
            "returncode": -1
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": f"执行出错: {str(e)}",
            "returncode": -1
        }

@app.route('/execute', methods=['POST'])
def execute_code():
    """执行代码的API端点"""
    try:
        # 验证请求
        if not request.is_json:
            raise BadRequest("请求必须是JSON格式")
        
        data = request.get_json()
        
        if 'code' not in data:
            raise BadRequest("请求中必须包含'code'字段")
        
        code = data['code']
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp(dir=BASE_DIR)
        
        try:
            # 运行代码
            result = run_python_code(code, temp_dir)
            
            response = {
                "success": result["returncode"] == 0,
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "returncode": result["returncode"],
                "temp_dir": temp_dir
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({
                "success": False,
                "stdout": "",
                "stderr": f"服务器内部错误: {str(e)}\n{traceback.format_exc()}",
                "returncode": -1
            }), 500
            
        finally:
            # 清理临时目录（可选，根据需求调整）
            # shutil.rmtree(temp_dir, ignore_errors=True)
            pass
            
    except BadRequest as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({
            "success": False, 
            "error": f"服务器错误: {str(e)}"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({"status": "healthy", "port": ALLOWED_PORT})

def main():
    """主函数"""
    print(f"启动沙盒服务器在端口 {ALLOWED_PORT}")
    print(f"允许写入的目录: {BASE_DIR}")
    print(f"只读路径: {READ_ONLY_PATHS}")
    print("服务器已启动，等待请求...")
    
    # 注意：在生产环境中，不要使用debug=True
    app.run(host='0.0.0.0', port=ALLOWED_PORT, debug=False, threaded=True)

if __name__ == '__main__':
    main()

# curl -X POST http://localhost:1234/execute \
#   -H "Content-Type: application/json" \
#   -d '{"code": "print(\"Hello, World!\")\nfor i in range(5):\n    print(f\"Number: {i}\")"}'

# curl -X POST http://localhost:1234/execute \
#   -H "Content-Type: application/json" \
#   -d '{
#     "code": "# Pillow测试\nfrom PIL import Image\nimport os\n\nsrc = \"/project/peilab/qjl/CODE/DATA/images/000.png\"\ndst = \"pil_cropped.png\"\n\n# 裁剪并保存\nimg = Image.open(src)\nw, h = img.size\ncropped = img.crop((w//4, h//4, 3*w//4, 3*h//4))\ncropped.save(dst)\n\nprint(f\"原始: {w}x{h}\")\nprint(f\"裁剪: {cropped.size}\")\nprint(f\"保存到: {os.path.abspath(dst)}\")\nprint(f\"文件存在: {os.path.exists(dst)}\")",
#     "timeout": 5
#   }'
