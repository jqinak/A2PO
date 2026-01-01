# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import requests
import logging
import os
import threading
from contextlib import ExitStack
from enum import Enum
from math import ceil, floor
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4
import cv2
import numpy as np
from math import ceil
from typing import Union
import ray
import ray.actor
from qwen_vl_utils import fetch_image, fetch_video

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


# Adapted from verl/tools/sandbox_fusion_tools.py
class PoolMode(Enum):
    """Execution pool mode enumeration."""

    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """Ray actor for rate limiting using token bucket algorithm."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0  # For observability
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        """Acquire a token from the bucket."""
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        """Release a token back to the bucket."""
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        """Get current number of acquired tokens."""
        return self.current_count


class VisualExecutionWorker:
    """Worker for executing visual processing operations with optional rate limiting."""

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        """Initialize singleton rate limiter."""
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        """Health check method."""
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        """Execute function with optional rate limiting."""
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return fn(*fn_args, **fn_kwargs)
                except Exception as e:
                    # TODO we should make this available to the tool caller
                    logger.warning(f"Error when executing visual processing: {e}")
        else:
            return fn(*fn_args, **fn_kwargs)


def init_visual_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode
):
    """Initialize visual execution pool."""
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(VisualExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")


class MultiModalCode_tool(BaseTool):
    """A tool for zooming in on an image by cropping it based on a bounding box.

    This tool provides a zoom-in functionality by cropping a region from an image,
    with rate limiting and concurrent execution support through Ray.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the zoom-in operation
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    MIN_DIMENSION = 28

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "multimodalcode_tool",
                "description": (
                    "Perform deep research for image and video via code excution"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bbox_2d": {
                            "type": "array",
                            "items":{"type":"number"},
                            "minItems":4,
                            "maxItems":4,
                            "description": (
                                "The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is "
                                "the top-left corner and (x2, y2) is the bottom-right corner."
                            ),
                        },
                        "label": {
                            "type": "string",
                            "description": "The name or label of the object in the specified bounding box (optional).",
                        },
                    },
                    "required": ["bbox_2d"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Worker and rate limiting configuration
        self.num_workers = config.get("num_workers", 20)
        self.rate_limit = config.get("rate_limit", 50)
        self.timeout = config.get("timeout", 30)
        self.code_sandbox_url = "http://0.0.0.0:1234/excute/sync"
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_visual_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )
        logger.info(f"Initialized multimodalcode_tool with config: {config}")


    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """
        Creates a new instance for image zoom-in tool.

        This method initializes a new session for an image, which can then be used
        for operations like zooming. It fetches the image from various sources
        and stores it internally.

        Args:
            instance_id: An optional unique identifier for the instance. If not
                provided, a new UUID will be generated.
            **kwargs: Should contain 'image' key with image data, or 'create_kwargs'
                containing {'image': image_data}. Image can be one of the following:
                - A PIL.Image.Image object.
                - A string containing an HTTP or HTTPS URL.
                - A string containing a local file path.
                - A string containing a file URI (e.g., "file:///path/to/image.jpg").
                - A string containing a base64-encoded image in the format of "data:image/jpeg;base64,..."

        Returns:
            Tuple of (instance_id, ToolResponse)
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Handle create_kwargs parameter if passed
        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)

        # Get image from kwargs
        image = kwargs.get("image")

        video = kwargs.get("video")

        if image:
            img = fetch_image({"image": image})
        elif video:
            video = fetch_video({"video": video})
        else:
            raise ValueError("Missing required 'image' or 'video' parameter in kwargs")

        self._instance_dict[instance_id] = {
            "image": img,
            "video": video,
            "type": "image" if image else "video",
            "response": "",
            "reward": 0.0,
        }

        return instance_id, ToolResponse()

    # def resize_min_image_opencv(self, image: np.ndarray) -> np.ndarray:
    #     """
    #     Qwen-VL raises an error for images with height or width less than 32 pixels.
    #     使用OpenCV格式处理图像。
        
    #     Args:
    #         image: OpenCV格式的图像，类型为numpy.ndarray，形状为(H, W, C)或(H, W)
    #             其中H为高度，W为宽度，C为通道数(1, 3或4)
    #             数据类型应为uint8或float32/float64
            
    #     Returns:
    #         调整大小后的图像，类型与输入相同，为numpy.ndarray
    #     """
    #     # 类型检查
    #     if not isinstance(image, np.ndarray):
    #         raise TypeError(f"Input image must be a numpy.ndarray, got {type(image)}")
        
    #     # 维度检查
    #     if image.ndim not in [2, 3]:
    #         raise ValueError(f"Image must have 2 or 3 dimensions, got {image.ndim}")
        
    #     # 获取图像尺寸 (OpenCV中是先高后宽)
    #     height, width = image.shape[:2]
        
    #     # 处理长宽比过大的情况
    #     if max(height, width) / min(height, width) > 200:
    #         max_val = max(height, width)
    #         min_val = min(height, width)

    #         old_scale = max_val / min_val
    #         max_ratio = min(150, old_scale / 2)
    #         target_max = int(min_val * max_ratio)

    #         if height > width:
    #             new_height = target_max
    #             new_width = int(width * old_scale / max_ratio)
    #         else:
    #             new_width = target_max
    #             new_height = int(height * old_scale / max_ratio)
            
    #         # 确保尺寸为正数
    #         new_width = max(1, new_width)
    #         new_height = max(1, new_height)
            
    #         # 使用OpenCV的resize函数
    #         image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    #         height, width = image.shape[:2]

    #     # 处理最小边小于32像素的情况
    #     if min(height, width) >= 32:
    #         return image

    #     ratio = 32 / min(height, width)
    #     new_height = ceil(height * ratio)
    #     new_width = ceil(width * ratio)
        
    #     # 确保尺寸为正数
    #     new_width = max(1, new_width)
    #     new_height = max(1, new_height)
        
    #     # 使用OpenCV的resize函数
    #     new_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    #     return new_image

    async def execute(self, instance_id: str, **kwargs) -> tuple[ToolResponse, float, dict]:
        '''
        Docstring for execute
            instance_id: url for code result

        :rtype: tuple[ToolResponse, float, dict]
        '''
        try:
            resjson = requests.get(instance_id)
            # {"success":true,
            # "stdout":"视频文件大小: 1348854 bytes\nFPS: 24.0, 总帧数: 657\n第20秒帧位置: 480\n帧尺寸: 480x270\n保存裁剪帧: /project/peilab/qjl/CODE/SERVER/tmp/tmp3xi_ezcy/frame_20s.jpg\n文件存在: True\n",
            # "stderr":"",
            # "returncode":0,
            # "execution_time":1.5079731941223145,
            # "tmp_path": absolute_dir}
            stdout = resjson['stdout']
            stderr = resjson['stderr']
            tool_response_text = f"[Sand_Box_Server]: {{success:{resjson['success']} stdout:{stdout}, stderr:{stderr}}}"
            tmp_path = resjson['tmp_path']
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v'}
            # 获取所有文件路径
            image_list = []
            video_list = []
            for root, dirs, files in os.walk(tmp_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    ext = os.path.splitext(file)[1].lower()
                    if ext in image_extensions:
                        image_list.append(file_path)
                    elif ext in video_extensions:
                        video_list.append(file_path)
            return ToolResponse(text=tool_response_text, image=image_list, video=video_list), 0.0 if resjson['success'] else -0.05, {"success": resjson['success']}
# return ToolResponse(text=tool_response_text, image=image_list if image_list else None, video=video_list if video_list else None), 0.0 if resjson['success'] else -0.05, {"success": resjson['success']}

        except Exception as err:
            tool_response_text = f' [ERROR code] Request to Sand_Box_Server failed: {err}'
            logging.error(tool_response_text)
            return ToolResponse(text=tool_response_text, image=image_list, video=video_list), 0.0 if resjson['success'] else -0.05, {"success": resjson['success']}

        
    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
