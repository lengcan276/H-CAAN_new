"""
基础代理类模块，定义了所有专业代理的通用接口和功能。
提供日志记录、错误处理、状态跟踪、通信协议和资源管理功能。
"""

import os
import sys
import time
import json
import logging
import traceback
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from abc import ABC, abstractmethod
import threading
import uuid

class BaseAgent(ABC):
    """
    所有专业代理的抽象基类。
    定义了代理之间通信和协作的标准接口。
    """
    
    def __init__(self, agent_manager=None, name: str = None):
        """
        初始化基础代理。
        
        Args:
            agent_manager: 代理管理器实例
            name: 代理名称，如果未提供则使用类名
        """
        self.agent_manager = agent_manager
        self.name = name or self.__class__.__name__
        self.id = str(uuid.uuid4())
        
        # 设置日志记录器
        self.logger = self._setup_logger()
        
        # 代理状态
        self.status = "initialized"
        self.busy = False
        self.last_activity = time.time()
        
        # 任务和结果跟踪
        self.tasks = {}
        self.results_cache = {}
        
        # 线程安全锁
        self.lock = threading.RLock()
        
        self.logger.info(f"Agent {self.name} ({self.id}) initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """
        设置代理日志记录器。
        
        Returns:
            配置好的日志记录器
        """
        logger = logging.getLogger(f"agent.{self.name}")
        
        # 如果日志记录器已经有处理程序，则不添加新的处理程序
        if logger.handlers:
            return logger
            
        logger.setLevel(logging.INFO)
        
        # 创建控制台处理程序
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建格式化程序
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # 添加处理程序到日志记录器
        logger.addHandler(console_handler)
        
        # 创建日志目录（如果不存在）
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建文件处理程序
        file_handler = logging.FileHandler(os.path.join(log_dir, f"{self.name.lower()}.log"))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # 添加文件处理程序到日志记录器
        logger.addHandler(file_handler)
        
        return logger
    
    def set_agent_manager(self, agent_manager) -> None:
        """
        设置代理管理器。
        
        Args:
            agent_manager: 代理管理器实例
        """
        self.agent_manager = agent_manager
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取代理当前状态。
        
        Returns:
            包含代理状态信息的字典
        """
        with self.lock:
            return {
                "id": self.id,
                "name": self.name,
                "status": self.status,
                "busy": self.busy,
                "last_activity": self.last_activity,
                "tasks_count": len(self.tasks),
                "cached_results_count": len(self.results_cache)
            }
    
    def update_status(self, status: str) -> None:
        """
        更新代理状态。
        
        Args:
            status: 新状态
        """
        with self.lock:
            self.status = status
            self.last_activity = time.time()
            self.logger.info(f"Status updated to: {status}")
    
    def execute_task(self, task_type: str, **task_args) -> Dict[str, Any]:
        """
        执行代理任务，提供错误处理和结果缓存。
        
        Args:
            task_type: 任务类型
            **task_args: 任务参数
            
        Returns:
            任务执行结果
            
        Raises:
            ValueError: 如果任务类型无效
            Exception: 任务执行中的其他错误
        """
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        with self.lock:
            # 将任务添加到任务列表
            self.tasks[task_id] = {
                "type": task_type,
                "args": task_args,
                "status": "pending",
                "start_time": time.time(),
                "end_time": None,
                "result": None,
                "error": None
            }
            # 更新代理状态
            self.busy = True
            self.update_status(f"executing_{task_type}")
        
        try:
            # 检查和获取已缓存结果
            cache_key = self._generate_cache_key(task_type, **task_args)
            if cache_key in self.results_cache:
                self.logger.info(f"Using cached result for task {task_type}")
                result = self.results_cache[cache_key]
                
                with self.lock:
                    self.tasks[task_id]["status"] = "completed"
                    self.tasks[task_id]["end_time"] = time.time()
                    self.tasks[task_id]["result"] = "cached_result"
                    self.busy = False
                    self.update_status("idle")
                
                return result
            
            # 查找对应的方法
            method_name = f"_{task_type}"
            if not hasattr(self, method_name):
                raise ValueError(f"Invalid task type: {task_type}")
            
            # 执行任务
            method = getattr(self, method_name)
            result = method(**task_args)
            
            # 缓存结果
            self.results_cache[cache_key] = result
            
            # 更新任务状态
            with self.lock:
                self.tasks[task_id]["status"] = "completed"
                self.tasks[task_id]["end_time"] = time.time()
                self.tasks[task_id]["result"] = "success"
                self.busy = False
                self.update_status("idle")
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            
            self.logger.error(f"Error executing task {task_type}: {error_msg}")
            self.logger.error(f"Stack trace: {stack_trace}")
            
            # 更新任务状态
            with self.lock:
                self.tasks[task_id]["status"] = "error"
                self.tasks[task_id]["end_time"] = time.time()
                self.tasks[task_id]["error"] = {
                    "message": error_msg,
                    "traceback": stack_trace
                }
                self.busy = False
                self.update_status("error")
            
            # 重新抛出异常以便上游处理
            raise
    
    def execute_async_task(self, task_type: str, callback: Callable = None, **task_args) -> str:
        """
        异步执行代理任务。
        
        Args:
            task_type: 任务类型
            callback: 任务完成时调用的回调函数
            **task_args: 任务参数
            
        Returns:
            任务ID
        """
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        # 创建线程执行任务
        def thread_func():
            try:
                result = self.execute_task(task_type, **task_args)
                
                # 调用回调
                if callback:
                    try:
                        callback(result, None)
                    except Exception as callback_error:
                        self.logger.error(f"Error in callback for task {task_id}: {str(callback_error)}")
            except Exception as e:
                # 调用回调传递错误
                if callback:
                    try:
                        callback(None, e)
                    except Exception as callback_error:
                        self.logger.error(f"Error in error callback for task {task_id}: {str(callback_error)}")
        
        thread = threading.Thread(target=thread_func)
        thread.daemon = True
        thread.start()
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        获取任务状态。
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务状态字典
        """
        with self.lock:
            return self.tasks.get(task_id, {"status": "unknown"})
    
    def request_from_agent(self, agent_name: str, task_type: str, **task_args) -> Any:
        """
        向其他代理请求执行任务。
        
        Args:
            agent_name: 目标代理名称
            task_type: 任务类型
            **task_args: 任务参数
            
        Returns:
            任务执行结果
            
        Raises:
            ValueError: 如果代理不存在或不可用
        """
        if not self.agent_manager:
            raise ValueError("No agent manager available")
            
        return self.agent_manager.execute_task(agent_name, task_type, **task_args)
    
    def broadcast_message(self, message_type: str, content: Any) -> Dict[str, Any]:
        """
        向所有代理广播消息。
        
        Args:
            message_type: 消息类型
            content: 消息内容
            
        Returns:
            包含各代理处理结果的字典
        """
        if not self.agent_manager:
            return {"error": "No agent manager available"}
            
        return self.agent_manager.broadcast_message(message_type, content, sender=self.name)
    
    def receive_message(self, message_type: str, content: Any, sender: str) -> Any:
        """
        接收来自其他代理的消息。
        
        Args:
            message_type: 消息类型
            content: 消息内容
            sender: 发送者名称
            
        Returns:
            消息处理结果
        """
        self.logger.info(f"Received message of type {message_type} from {sender}")
        
        # 查找对应的消息处理方法
        handler_name = f"_handle_{message_type}"
        if hasattr(self, handler_name):
            handler = getattr(self, handler_name)
            return handler(content, sender)
        else:
            self.logger.warning(f"No handler for message type {message_type}")
            return {"status": "unhandled", "message": f"No handler for message type {message_type}"}
    
    def clear_cache(self) -> None:
        """清除结果缓存"""
        with self.lock:
            self.results_cache = {}
            self.logger.info("Cache cleared")
    
    def save_state(self, file_path: Optional[str] = None) -> str:
        """
        保存代理状态到文件。
        
        Args:
            file_path: 保存状态的文件路径（如果未提供，则使用默认路径）
            
        Returns:
            保存状态的文件路径
        """
        if file_path is None:
            # 创建状态目录（如果不存在）
            state_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'state')
            os.makedirs(state_dir, exist_ok=True)
            
            file_path = os.path.join(state_dir, f"{self.name.lower()}_state.json")
        
        with self.lock:
            # 准备要保存的状态
            state = {
                "id": self.id,
                "name": self.name,
                "status": self.status,
                "last_activity": self.last_activity,
                "tasks": self.tasks,
                # 不保存可再生成的缓存结果
                "timestamp": time.time()
            }
            
            # 保存状态
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.logger.info(f"State saved to {file_path}")
            
            return file_path
    
    def load_state(self, file_path: Optional[str] = None) -> bool:
        """
        从文件加载代理状态。
        
        Args:
            file_path: 状态文件路径（如果未提供，则使用默认路径）
            
        Returns:
            加载是否成功
        """
        if file_path is None:
            state_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'state')
            file_path = os.path.join(state_dir, f"{self.name.lower()}_state.json")
        
        if not os.path.exists(file_path):
            self.logger.warning(f"State file {file_path} does not exist")
            return False
        
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            with self.lock:
                # 恢复状态
                self.id = state.get("id", self.id)
                self.status = state.get("status", "initialized")
                self.last_activity = state.get("last_activity", time.time())
                self.tasks = state.get("tasks", {})
            
            self.logger.info(f"State loaded from {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading state from {file_path}: {str(e)}")
            return False
    
    def _generate_cache_key(self, task_type: str, **task_args) -> str:
        """
        为任务生成缓存键。
        
        Args:
            task_type: 任务类型
            **task_args: 任务参数
            
        Returns:
            缓存键
        """
        # 将任务参数转换为JSON字符串作为缓存键的一部分
        args_str = json.dumps(task_args, sort_keys=True)
        return f"{task_type}:{args_str}"
    
    def _handle_ping(self, content: Any, sender: str) -> Dict[str, Any]:
        """
        处理ping消息。
        
        Args:
            content: 消息内容
            sender: 发送者名称
            
        Returns:
            包含状态信息的响应
        """
        return {
            "status": "ok",
            "agent": self.name,
            "timestamp": time.time(),
            "message": f"Hello {sender}, this is {self.name}"
        }
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        获取代理功能列表。
        
        Returns:
            代理支持的任务类型列表
        """
        pass
    
    @abstractmethod
    def execute(self, task_type: str, **kwargs) -> Any:
        """
        执行特定任务的入口点方法。
        
        Args:
            task_type: 任务类型
            **kwargs: 任务参数
            
        Returns:
            任务执行结果
        """
        pass