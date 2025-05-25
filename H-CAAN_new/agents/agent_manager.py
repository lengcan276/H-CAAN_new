"""
代理管理器模块，用于协调和管理H-CAAN项目中的多个专业代理。
提供统一的接口进行任务执行、消息传递和错误处理。
"""

import os
import sys
import time
import json
import logging
import traceback
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import threading
import uuid

# 确保base_agent可用
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from .base_agent import BaseAgent

class AgentManager:
    """
    管理和协调多个专业代理的中心系统。
    处理代理注册、任务分发、消息传递和错误恢复。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化代理管理器。
        
        Args:
            config: 配置字典，包含日志级别、错误处理策略等
        """
        # 默认配置
        self.config = {
            "log_level": "INFO",
            "error_retry_attempts": 3,
            "error_retry_delay": 2.0,  # 秒
            "timeout": 120.0,  # 秒
            "auto_recovery": True,
            "save_state_interval": 300.0,  # 秒
            "state_dir": os.path.join(os.path.dirname(current_dir), "state")
        }
        
        # 更新配置
        if config:
            self.config.update(config)
        
        # 设置日志记录器
        self.logger = self._setup_logger()
        
        # 注册的代理
        self.agents = {}
        
        # 代理通信总线
        self.message_bus = {}
        
        # 全局知识库
        self.knowledge_base = {}
        
        # 任务历史记录
        self.task_history = []
        
        # 线程安全锁
        self.lock = threading.RLock()
        
        # 代理状态保存计时器
        self._start_state_save_timer()
        
        self.logger.info("Agent Manager initialized successfully")
    
    def _setup_logger(self) -> logging.Logger:
        """
        设置日志记录器。
        
        Returns:
            配置好的日志记录器
        """
        logger = logging.getLogger("AgentManager")
        
        # 设置日志级别
        log_level = getattr(logging, self.config["log_level"].upper(), logging.INFO)
        logger.setLevel(log_level)
        
        # 检查是否已有处理程序
        if logger.handlers:
            return logger
        
        # 创建控制台处理程序
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # 创建格式化程序
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # 添加处理程序到日志记录器
        logger.addHandler(console_handler)
        
        # 创建日志目录（如果不存在）
        log_dir = os.path.join(os.path.dirname(current_dir), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建文件处理程序
        file_handler = logging.FileHandler(os.path.join(log_dir, "agent_manager.log"))
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        
        # 添加文件处理程序到日志记录器
        logger.addHandler(file_handler)
        
        return logger
    
    def _start_state_save_timer(self) -> None:
        """启动定期保存代理状态的计时器"""
        save_interval = self.config["save_state_interval"]
        
        def save_state_worker():
            while True:
                time.sleep(save_interval)
                try:
                    self.save_all_agent_states()
                except Exception as e:
                    self.logger.error(f"Error saving agent states: {str(e)}")
        
        # 创建并启动后台线程
        thread = threading.Thread(target=save_state_worker, daemon=True)
        thread.start()
    
    def register_agent(self, agent: BaseAgent) -> None:
        """
        注册单个代理。
        
        Args:
            agent: 要注册的代理实例
        """
        with self.lock:
            self.agents[agent.name] = agent
            agent.set_agent_manager(self)
            self.logger.info(f"Registered agent: {agent.name}")
    
    def register_agents(self, agents: Dict[str, BaseAgent]) -> None:
        """
        批量注册多个代理。
        
        Args:
            agents: 代理名称到代理实例的映射字典
        """
        for name, agent in agents.items():
            agent.name = name  # 确保代理名称与键匹配
            self.register_agent(agent)
    
    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """
        获取指定名称的代理。
        
        Args:
            agent_name: 代理名称
            
        Returns:
            代理实例或None（如果未找到）
        """
        return self.agents.get(agent_name)
    
    def get_all_agents(self) -> Dict[str, BaseAgent]:
        """
        获取所有注册的代理。
        
        Returns:
            代理名称到代理实例的映射字典
        """
        return self.agents.copy()
    
    def execute_task(self, agent_name: str, task_type: str, **task_args) -> Any:
        """
        执行特定代理的任务。
        
        Args:
            agent_name: 代理名称
            task_type: 任务类型
            **task_args: 任务参数
            
        Returns:
            任务执行结果
            
        Raises:
            ValueError: 如果代理不存在
            Exception: 任务执行中的其他错误
        """
        start_time = time.time()
        task_id = str(uuid.uuid4())
        
        # 获取代理
        agent = self.get_agent(agent_name)
        if not agent:
            error_msg = f"Agent not found: {agent_name}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # 记录任务开始
        self.logger.info(f"Executing task: {task_type} on agent: {agent_name} (ID: {task_id})")
        
        # 添加到任务历史
        task_record = {
            "id": task_id,
            "agent": agent_name,
            "task_type": task_type,
            "args": task_args,
            "start_time": start_time,
            "end_time": None,
            "status": "running",
            "result": None,
            "error": None
        }
        
        with self.lock:
            self.task_history.append(task_record)
        
        retry_attempts = self.config["error_retry_attempts"]
        retry_delay = self.config["error_retry_delay"]
        
        # 尝试执行任务，支持重试
        for attempt in range(1, retry_attempts + 1):
            try:
                # 执行任务
                result = agent.execute_task(task_type, **task_args)
                
                # 更新任务记录
                end_time = time.time()
                with self.lock:
                    for task in self.task_history:
                        if task["id"] == task_id:
                            task["end_time"] = end_time
                            task["status"] = "completed"
                            task["result"] = "success"
                
                # 记录任务完成
                execution_time = end_time - start_time
                self.logger.info(f"Task completed: {task_type} (ID: {task_id}) in {execution_time:.2f}s")
                
                return result
                
            except Exception as e:
                error_msg = str(e)
                stack_trace = traceback.format_exc()
                
                # 记录错误
                self.logger.error(f"Error executing task {task_type} on agent {agent_name} "
                                 f"(Attempt {attempt}/{retry_attempts}): {error_msg}")
                self.logger.debug(f"Stack trace: {stack_trace}")
                
                # 如果有更多重试机会，则等待后重试
                if attempt < retry_attempts:
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    # 更新任务记录
                    end_time = time.time()
                    with self.lock:
                        for task in self.task_history:
                            if task["id"] == task_id:
                                task["end_time"] = end_time
                                task["status"] = "error"
                                task["error"] = {
                                    "message": error_msg,
                                    "traceback": stack_trace
                                }
                    
                    # 记录最终失败
                    self.logger.error(f"Task failed after {retry_attempts} attempts: {task_type} (ID: {task_id})")
                    
                    # 重新抛出异常
                    raise
    
    def execute_task_async(self, agent_name: str, task_type: str, 
                         callback: Optional[Callable] = None, **task_args) -> str:
        """
        异步执行特定代理的任务。
        
        Args:
            agent_name: 代理名称
            task_type: 任务类型
            callback: 任务完成时调用的回调函数
            **task_args: 任务参数
            
        Returns:
            任务ID
        """
        task_id = str(uuid.uuid4())
        
        def thread_func():
            try:
                result = self.execute_task(agent_name, task_type, **task_args)
                
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
    
    def broadcast_message(self, message_type: str, content: Any, 
                        sender: Optional[str] = None) -> Dict[str, Any]:
        """
        向所有代理广播消息。
        
        Args:
            message_type: 消息类型
            content: 消息内容
            sender: 发送者名称（可选）
            
        Returns:
            包含各代理处理结果的字典
        """
        self.logger.info(f"Broadcasting message of type {message_type} from {sender or 'AgentManager'}")
        
        results = {}
        
        # 向每个代理发送消息
        for name, agent in self.agents.items():
            # 跳过发送者
            if name == sender:
                continue
                
            try:
                # 发送消息并接收响应
                response = agent.receive_message(message_type, content, sender or "AgentManager")
                results[name] = response
            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"Error sending message to agent {name}: {error_msg}")
                results[name] = {"error": error_msg}
        
        return results
    
    def send_message(self, from_agent: str, to_agent: str, 
                   message_type: str, content: Any) -> Any:
        """
        从一个代理向另一个代理发送消息。
        
        Args:
            from_agent: 发送者代理名称
            to_agent: 接收者代理名称
            message_type: 消息类型
            content: 消息内容
            
        Returns:
            接收者的响应
            
        Raises:
            ValueError: 如果代理不存在
        """
        # 获取接收者代理
        agent = self.get_agent(to_agent)
        if not agent:
            error_msg = f"Target agent not found: {to_agent}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # 记录消息
        self.logger.info(f"Sending message of type {message_type} from {from_agent} to {to_agent}")
        
        try:
            # 发送消息并获取响应
            response = agent.receive_message(message_type, content, from_agent)
            return response
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error sending message from {from_agent} to {to_agent}: {error_msg}")
            raise
    
    def update_knowledge_base(self, data: Dict[str, Any]) -> None:
        """
        更新全局知识库。
        
        Args:
            data: 要添加到知识库的数据字典
        """
        with self.lock:
            self.knowledge_base.update(data)
            self.logger.debug(f"Knowledge base updated with {len(data)} items")
    
    def get_knowledge_base(self) -> Dict[str, Any]:
        """
        获取全局知识库。
        
        Returns:
            知识库字典
        """
        return self.knowledge_base.copy()
    
    def get_task_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取任务执行历史。
        
        Args:
            limit: 返回的最大记录数（如果为None，则返回所有记录）
            
        Returns:
            任务历史记录列表
        """
        with self.lock:
            if limit:
                return self.task_history[-limit:]
            return self.task_history.copy()
    
    def save_all_agent_states(self) -> Dict[str, str]:
        """
        保存所有代理的状态。
        
        Returns:
            代理名称到状态文件路径的映射字典
        """
        state_dir = self.config["state_dir"]
        os.makedirs(state_dir, exist_ok=True)
        
        state_files = {}
        
        for name, agent in self.agents.items():
            try:
                file_path = os.path.join(state_dir, f"{name.lower()}_state.json")
                agent.save_state(file_path)
                state_files[name] = file_path
            except Exception as e:
                self.logger.error(f"Error saving state for agent {name}: {str(e)}")
        
        # 保存管理器自身状态
        try:
            manager_state = {
                "knowledge_base": self.knowledge_base,
                "task_history": self.task_history[-100:],  # 只保存最近100个任务
                "timestamp": time.time()
            }
            
            manager_state_path = os.path.join(state_dir, "agent_manager_state.json")
            with open(manager_state_path, 'w') as f:
                json.dump(manager_state, f, indent=2)
                
            state_files["manager"] = manager_state_path
            
        except Exception as e:
            self.logger.error(f"Error saving manager state: {str(e)}")
        
        self.logger.info(f"Saved states for {len(state_files)} agents")
        return state_files
    
    def load_all_agent_states(self) -> Dict[str, bool]:
        """
        加载所有代理的状态。
        
        Returns:
            代理名称到加载成功标志的映射字典
        """
        state_dir = self.config["state_dir"]
        
        if not os.path.exists(state_dir):
            self.logger.warning(f"State directory {state_dir} does not exist")
            return {}
        
        results = {}
        
        for name, agent in self.agents.items():
            try:
                file_path = os.path.join(state_dir, f"{name.lower()}_state.json")
                success = agent.load_state(file_path)
                results[name] = success
            except Exception as e:
                self.logger.error(f"Error loading state for agent {name}: {str(e)}")
                results[name] = False
        
        # 加载管理器自身状态
        try:
            manager_state_path = os.path.join(state_dir, "agent_manager_state.json")
            
            if os.path.exists(manager_state_path):
                with open(manager_state_path, 'r') as f:
                    manager_state = json.load(f)
                
                with self.lock:
                    if "knowledge_base" in manager_state:
                        self.knowledge_base = manager_state["knowledge_base"]
                    
                    if "task_history" in manager_state:
                        self.task_history = manager_state["task_history"]
                
                results["manager"] = True
            else:
                results["manager"] = False
                
        except Exception as e:
            self.logger.error(f"Error loading manager state: {str(e)}")
            results["manager"] = False
        
        self.logger.info(f"Loaded states for {sum(1 for v in results.values() if v)} agents")
        return results
    
    def check_agent_health(self) -> Dict[str, str]:
        """
        检查所有代理的健康状态。
        
        Returns:
            代理名称到健康状态描述的映射字典
        """
        health_status = {}
        
        for name, agent in self.agents.items():
            try:
                # 获取代理状态
                status = agent.get_status()
                
                # 检查上次活动时间
                last_activity = status.get("last_activity", 0)
                current_time = time.time()
                time_since_activity = current_time - last_activity
                
                # 根据活动时间和状态判断健康状态
                if time_since_activity > 3600:  # 超过1小时无活动
                    health_status[name] = "inactive"
                elif status.get("status") == "error":
                    health_status[name] = "error"
                elif status.get("busy"):
                    health_status[name] = "busy"
                else:
                    health_status[name] = "healthy"
                    
            except Exception as e:
                self.logger.error(f"Error checking health of agent {name}: {str(e)}")
                health_status[name] = "unknown"
        
        return health_status
    
    def restart_agent(self, agent_name: str) -> bool:
        """
        尝试重启特定代理。
        
        Args:
            agent_name: 要重启的代理名称
            
        Returns:
            重启是否成功
        """
        agent = self.get_agent(agent_name)
        if not agent:
            self.logger.error(f"Cannot restart: Agent not found: {agent_name}")
            return False
        
        try:
            # 保存当前状态
            self.logger.info(f"Saving state before restarting agent {agent_name}")
            state_file = agent.save_state()
            
            # 获取代理类
            agent_class = agent.__class__
            
            # 创建新实例
            self.logger.info(f"Creating new instance of agent {agent_name}")
            new_agent = agent_class(self)
            new_agent.name = agent_name
            
            # 加载保存的状态
            self.logger.info(f"Loading state into new agent instance")
            new_agent.load_state(state_file)
            
            # 更新代理注册
            with self.lock:
                self.agents[agent_name] = new_agent
            
            self.logger.info(f"Successfully restarted agent {agent_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error restarting agent {agent_name}: {str(e)}")
            return False
    
    def execute_workflow(self, workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行预定义工作流。
        
        Args:
            workflow_config: 工作流配置字典
            
        Returns:
            工作流结果字典
        """
        workflow_name = workflow_config.get("name", "unnamed_workflow")
        steps = workflow_config.get("steps", [])
        
        self.logger.info(f"Starting workflow: {workflow_name} with {len(steps)} steps")
        
        workflow_results = {
            "name": workflow_name,
            "start_time": time.time(),
            "end_time": None,
            "status": "running",
            "step_results": {}
        }
        
        for i, step in enumerate(steps):
            step_name = step.get("name", f"step_{i+1}")
            agent_name = step.get("agent")
            task_type = step.get("task")
            task_args = step.get("args", {})
            
            self.logger.info(f"Executing workflow step {i+1}/{len(steps)}: {step_name}")
            
            try:
                # 执行步骤
                result = self.execute_task(agent_name, task_type, **task_args)
                
                # 保存步骤结果
                workflow_results["step_results"][step_name] = {
                    "status": "completed",
                    "result": result
                }
                
            except Exception as e:
                error_msg = str(e)
                stack_trace = traceback.format_exc()
                
                self.logger.error(f"Error in workflow step {step_name}: {error_msg}")
                
                # 保存错误信息
                workflow_results["step_results"][step_name] = {
                    "status": "error",
                    "error": {
                        "message": error_msg,
                        "traceback": stack_trace
                    }
                }
                
                # 检查是否继续工作流
                if step.get("critical", False):
                    self.logger.error(f"Critical step {step_name} failed, aborting workflow")
                    workflow_results["status"] = "failed"
                    break
        
        # 完成工作流
        workflow_results["end_time"] = time.time()
        if workflow_results["status"] != "failed":
            workflow_results["status"] = "completed"
            
        execution_time = workflow_results["end_time"] - workflow_results["start_time"]
        self.logger.info(f"Workflow {workflow_name} {workflow_results['status']} in {execution_time:.2f}s")
        
        return workflow_results
    
    def shutdown(self) -> None:
        """
        关闭代理管理器，保存状态并清理资源。
        """
        self.logger.info("Shutting down Agent Manager")
        
        # 保存所有代理状态
        try:
            self.save_all_agent_states()
        except Exception as e:
            self.logger.error(f"Error saving agent states during shutdown: {str(e)}")
        
        # 通知所有代理关闭
        for name, agent in self.agents.items():
            try:
                agent.receive_message("shutdown", {}, "AgentManager")
            except Exception as e:
                self.logger.error(f"Error shutting down agent {name}: {str(e)}")
        
        self.logger.info("Agent Manager shutdown complete")