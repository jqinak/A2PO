import os
import sys
import json
import asyncio
import tempfile
import shutil
import aiofiles
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import traceback
from typing import Optional, Dict, Any
import uuid
import time
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import resource
import subprocess

# 配置参数
ALLOWED_PORT = 1234
BASE_DIR = "/project/peilab/qjl/CODE/SERVER/tmp"
READ_ONLY_PATHS = ["/usr", "/lib", "/bin", "/etc", os.path.expanduser("~")]

# 确保工作目录存在
os.makedirs(BASE_DIR, exist_ok=True)

# 全局变量
execution_tasks: Dict[str, Dict] = {}
thread_pool: Optional[ThreadPoolExecutor] = None
process_pool: Optional[ProcessPoolExecutor] = None

# 数据模型
class CodeExecutionRequest(BaseModel):
    code: str
    timeout: Optional[int] = 10  # 默认10秒超时
    memory_limit: Optional[int] = 256  # MB
    cpu_limit: Optional[int] = 5  # 秒

class ExecutionResult(BaseModel):
    task_id: str
    status: str  # pending, running, completed, failed
    result: Optional[Dict[str, Any]] = None
    created_at: float
    completed_at: Optional[float] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """生命周期管理器"""
    # 启动时
    print(f"异步沙盒服务器启动在端口 {ALLOWED_PORT}")
    print(f"允许写入的目录: {BASE_DIR}")
    
    global thread_pool, process_pool
    
    # 初始化线程池和进程池
    thread_pool = ThreadPoolExecutor(max_workers=50)
    process_pool = ProcessPoolExecutor(max_workers=10)
    
    app.startup_time = time.time()
    
    print(f"线程池大小: {thread_pool._max_workers}")
    print(f"进程池大小: {process_pool._max_workers}")
    print("服务器已启动，等待请求...")
    
    yield  # 运行应用程序
    
    # 关闭时
    print("服务器正在关闭，清理资源...")
    
    if thread_pool:
        thread_pool.shutdown(wait=False)
    if process_pool:
        process_pool.shutdown(wait=False)
    
    # 清理所有临时目录
    for task_id, task in execution_tasks.items():
        if 'temp_dir' in task and task['temp_dir']:
            try:
                shutil.rmtree(task['temp_dir'], ignore_errors=True)
            except:
                pass

# 创建FastAPI应用
app = FastAPI(
    title="Python沙盒服务器",
    description="异步Python代码执行沙盒",
    version="1.0.0",
    lifespan=lifespan
)

def is_write_allowed(path):
    """检查是否允许写入路径"""
    path = os.path.abspath(path)
    base_dir = os.path.abspath(BASE_DIR)
    
    if path.startswith(base_dir + os.sep) or path == base_dir:
        return True
    
    for read_only_path in READ_ONLY_PATHS:
        read_only_path_abs = os.path.abspath(read_only_path)
        if path.startswith(read_only_path_abs + os.sep) or path == read_only_path_abs:
            return False
    
    return False

def _execute_in_process(code: str, temp_dir: str, timeout: int, 
                       memory_limit: int, cpu_limit: int) -> Dict[str, Any]:
    """
    在子进程中执行代码
    """
    import tempfile
    import os
    import sys
    import subprocess
    import time
    import hashlib
    
    start_time = time.time()
    
    # 创建临时文件
    temp_file = os.path.join(temp_dir, "user_code.py")
    
    try:
        # 写入用户代码
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(code)
        
        # 构建安全执行环境
        env = os.environ.copy()
        env['PYTHONPATH'] = temp_dir
        
                # 修复后的包装脚本 - 修正f-string引号问题
        wrapper_code = f'''
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
        
        wrapper_file = os.path.join(temp_dir, "wrapper.py")
        with open(wrapper_file, 'w', encoding='utf-8') as f:
            f.write(wrapper_code)
        
        # 运行代码
        result = subprocess.run(
            [sys.executable, wrapper_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=temp_dir,
            env=env
        )
        
        execution_time = time.time() - start_time
        
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "execution_time": execution_time
        }
        
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        return {
            "stdout": "",
            "stderr": f"代码执行超时（超过{timeout}秒）",
            "returncode": -1,
            "execution_time": execution_time
        }
    except Exception as e:
        execution_time = time.time() - start_time
        return {
            "stdout": "",
            "stderr": f"执行出错: {str(e)}",
            "returncode": -1,
            "execution_time": execution_time
        }

async def run_code_in_process(code: str, temp_dir: str, timeout: int = 10, 
                             memory_limit: int = 256, cpu_limit: int = 5) -> Dict[str, Any]:
    """
    在独立的进程中运行代码
    """
    loop = asyncio.get_event_loop()
    
    try:
        result = await loop.run_in_executor(
            process_pool,
            _execute_in_process,
            code, temp_dir, timeout, memory_limit, cpu_limit
        )
        
        return {
            "success": result["returncode"] == 0,
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "returncode": result["returncode"],
            "execution_time": result.get("execution_time", 0)
        }
        
    except asyncio.TimeoutError:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"代码执行超时（超过{timeout}秒）",
            "returncode": -1,
            "execution_time": timeout
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"执行出错: {str(e)}",
            "returncode": -1,
            "execution_time": 0
        }

async def execute_code_task(task_id: str, code: str, timeout: int, 
                           memory_limit: int, cpu_limit: int):
    """
    异步执行代码任务
    """
    # 更新任务状态为运行中
    execution_tasks[task_id]['status'] = "running"
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(dir=BASE_DIR)
    execution_tasks[task_id]['temp_dir'] = temp_dir
    
    try:
        # 异步执行代码
        result = await run_code_in_process(
            code, temp_dir, timeout, memory_limit, cpu_limit
        )
        
        # 更新任务结果
        execution_tasks[task_id]['status'] = "completed"
        execution_tasks[task_id]['result'] = result
        execution_tasks[task_id]['completed_at'] = time.time()
        execution_tasks[task_id]['tmp_path'] = temp_dir
        
    except Exception as e:
        # 任务失败
        execution_tasks[task_id]['status'] = "failed"
        execution_tasks[task_id]['result'] = {
            "success": False,
            "stdout": "",
            "stderr": f"任务执行失败: {str(e)}",
            "returncode": -1
        }
        execution_tasks[task_id]['completed_at'] = time.time()
        traceback.print_exc()
        
    finally:
        # 延迟清理临时目录
        async def cleanup():
            await asyncio.sleep(300)  # 5分钟后清理
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                if 'temp_dir' in execution_tasks[task_id]:
                    del execution_tasks[task_id]['temp_dir']
            except:
                pass
        
        asyncio.create_task(cleanup())

@app.post("/execute")
async def execute_code(request: CodeExecutionRequest, background_tasks: BackgroundTasks):
    """
    提交代码执行任务（异步）
    """
    # 生成任务ID
    task_id = str(uuid.uuid4())
    
    # 创建任务记录
    execution_tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "created_at": time.time(),
        "code": request.code[:100] + "..." if len(request.code) > 100 else request.code
    }
    
    # 在后台执行任务
    background_tasks.add_task(
        execute_code_task,
        task_id,
        request.code,
        request.timeout,
        request.memory_limit,
        request.cpu_limit
    )
    
    return {
        "task_id": task_id,
        "status": "pending",
        "message": "任务已提交，正在处理",
        "created_at": execution_tasks[task_id]['created_at'],
        "check_status_url": f"/task/{task_id}/status",
        "get_result_url": f"/task/{task_id}/result"
    }

@app.get("/task/{task_id}/status")
async def get_task_status(task_id: str):
    """
    获取任务状态
    """
    if task_id not in execution_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = execution_tasks[task_id]
        # 添加tmp_path字段（如果存在且任务还在运行中）
    response = {
        "task_id": task_id,
        "status": task['status'],
        "created_at": task['created_at'],
        "completed_at": task.get('completed_at'),
        "elapsed_time": time.time() - task['created_at'] if task.get('completed_at') is None else task['completed_at'] - task['created_at']
    }

    if 'tmp_path' in task and task['status'] in ["pending", "running"]:
        response['tmp_path'] = task['tmp_path']

    return response

@app.get("/task/{task_id}/result")
async def get_task_result(task_id: str):
    """
    获取任务结果
    """
    if task_id not in execution_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = execution_tasks[task_id]
    
    if task['status'] == "pending":
        raise HTTPException(status_code=202, detail="任务还在处理中")
    
    if task['status'] == "running":
        raise HTTPException(status_code=202, detail="任务正在执行中")
    
    if 'result' not in task:
        raise HTTPException(status_code=500, detail="任务结果为空")
    
    # 返回结果
    result = task['result'].copy()
    if 'tmp_path' in task:
        result['tmp_path'] = task['tmp_path']
    # 清理过期的任务记录（30分钟后）
    async def cleanup_task():
        await asyncio.sleep(1800)  # 30分钟
        if task_id in execution_tasks:
            del execution_tasks[task_id]
    
    asyncio.create_task(cleanup_task())
    
    return result

@app.post("/execute/sync")
async def execute_code_sync(request: CodeExecutionRequest):
    """
    同步执行代码（兼容旧接口）
    """
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(dir=BASE_DIR)
    
    try:
        # 异步执行代码
        result = await run_code_in_process(
            request.code, temp_dir, request.timeout, 
            request.memory_limit, request.cpu_limit
        )
        
        # 延迟清理临时目录
        async def cleanup():
            await asyncio.sleep(60)  # 1分钟后清理
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass
        
        asyncio.create_task(cleanup())
        
        return result
        
    except Exception as e:
        # 立即清理
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"执行失败: {str(e)}")

@app.get("/health")
async def health_check():
    """
    健康检查
    """
    pending = len([t for t in execution_tasks.values() if t['status'] in ["pending", "running"]])
    completed = len([t for t in execution_tasks.values() if t['status'] == "completed"])
    
    status = {
        "status": "healthy",
        "port": ALLOWED_PORT,
        "base_dir": BASE_DIR,
        "pending_tasks": pending,
        "completed_tasks": completed,
        "thread_pool_workers": thread_pool._max_workers if thread_pool else 0,
        "process_pool_workers": process_pool._max_workers if process_pool else 0,
        "uptime": time.time() - app.startup_time if hasattr(app, 'startup_time') else 0,
        "timestamp": time.time()
    }
    
    return status

@app.get("/stats")
async def get_stats():
    """
    获取服务器统计信息
    """
    tasks_by_status = {
        "pending": 0,
        "running": 0,
        "completed": 0,
        "failed": 0
    }
    
    for task in execution_tasks.values():
        status = task['status']
        if status in tasks_by_status:
            tasks_by_status[status] += 1
    
    stats = {
        "total_tasks": len(execution_tasks),
        "tasks_by_status": tasks_by_status,
        "uptime": time.time() - app.startup_time if hasattr(app, 'startup_time') else 0
    }
    
    return stats

@app.get("/")
async def root():
    """
    根路径
    """
    return {
        "message": "Python沙盒服务器",
        "version": "1.0.0",
        "endpoints": {
            "POST /execute": "异步执行代码",
            "POST /execute/sync": "同步执行代码",
            "GET /task/{task_id}/status": "获取任务状态",
            "GET /task/{task_id}/result": "获取任务结果",
            "GET /health": "健康检查",
            "GET /stats": "服务器统计"
        }
    }

def main():
    """主函数"""
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=ALLOWED_PORT,
        log_level="info"
    )

if __name__ == "__main__":
    main()



# curl -X POST http://localhost:1234/execute   -H "Content-Type: application/json"   -d '{
#     "code": "# OpenCV读取视频帧\nimport cv2\nimport os\n\nvideo_path = \"/project/peilab/qjl/CODE/DATA/videos/0AGCS.mp4\"\noutput_path = \"frame_20s.jpg\"\n\n# 检查文件是否存在\nif not os.path.exists(video_path):\n    print(f\"视频文件不存在: {video_path}\")\n    exit(1)\n\nprint(f\"视频文件大小: {os.path.getsize(video_path)} bytes\")\n\n# 打开视频\ncap = cv2.VideoCapture(video_path)\nfps = cap.get(cv2.CAP_PROP_FPS)\ntotal_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\nprint(f\"FPS: {fps}, 总帧数: {total_frames}\")\n\n# 计算第20秒的帧位置\nframe_20s = int(20 * fps)\nprint(f\"第20秒帧位置: {frame_20s}\")\n\n# 跳转到第20秒\ncap.set(cv2.CAP_PROP_POS_FRAMES, frame_20s)\nret, frame = cap.read()\n\nif ret:\n    h, w = frame.shape[:2]\n    print(f\"帧尺寸: {w}x{h}\")\n    \n    # 裁剪中心区域\n    cropped = frame[h//4:3*h//4, w//4:3*w//4]\n    cv2.imwrite(output_path, cropped)\n    print(f\"保存裁剪帧: {os.path.abspath(output_path)}\")\n    print(f\"文件存在: {os.path.exists(output_path)}\")\nelse:\n    print(\"读取帧失败\")\n\ncap.release()",
#     "timeout": 30
#   }'


# curl -X GET http://0.0.0.0:1234/task/5519e62d-f41f-46ad-b59c-ad992a1593b6/result   -H "Content-Type: application/json"   -d '{
#     "code": "# OpenCV读取视频帧\nimport cv2\nimport os\n\nvideo_path = \"/project/peilab/qjl/CODE/DATA/videos/0AGCS.mp4\"\noutput_path = \"frame_20s.jpg\"\n\n# 检查文件是否存在\nif not os.path.exists(video_path):\n    print(f\"视频文件不存在: {video_path}\")\n    exit(1)\n\nprint(f\"视频文件大小: {os.path.getsize(video_path)} bytes\")\n\n# 打开视频\ncap = cv2.VideoCapture(video_path)\nfps = cap.get(cv2.CAP_PROP_FPS)\ntotal_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\nprint(f\"FPS: {fps}, 总帧数: {total_frames}\")\n\n# 计算第20秒的帧位置\nframe_20s = int(20 * fps)\nprint(f\"第20秒帧位置: {frame_20s}\")\n\n# 跳转到第20秒\ncap.set(cv2.CAP_PROP_POS_FRAMES, frame_20s)\nret, frame = cap.read()\n\nif ret:\n    h, w = frame.shape[:2]\n    print(f\"帧尺寸: {w}x{h}\")\n    \n    # 裁剪中心区域\n    cropped = frame[h//4:3*h//4, w//4:3*w//4]\n    cv2.imwrite(output_path, cropped)\n    print(f\"保存裁剪帧: {os.path.abspath(output_path)}\")\n    print(f\"文件存在: {os.path.exists(output_path)}\")\nelse:\n    print(\"读取帧失败\")\n\ncap.release()",
#     "timeout": 30
#   }'