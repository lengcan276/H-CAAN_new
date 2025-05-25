"""
入口路由，管理整体任务流
协调任务链的执行和状态管理
"""
from typing import Dict, Any, List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime

from .task_chain import TaskChain
from ..agents.multi_agent_manager import MultiAgentManager

logger = logging.getLogger(__name__)

class WorkflowRouter:
    """工作流路由器"""
    
    def __init__(self):
        self.task_chain = TaskChain()
        self.manager = MultiAgentManager()
        self.active_workflows = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def route_request(self, request: Dict) -> Any:
        """
        路由请求到相应的处理流程
        
        Args:
            request: 包含请求类型和参数的字典
            
        Returns:
            处理结果
        """
        request_type = request.get('type')
        params = request.get('params', {})
        
        logger.info(f"路由请求: {request_type}")
        
        if request_type == 'execute_chain':
            return self.execute_chain(params.get('chain_name'), params.get('inputs'))
            
        elif request_type == 'execute_custom':
            return self.execute_custom_workflow(params.get('tasks'), params.get('inputs'))
            
        elif request_type == 'execute_parallel':
            return self.execute_parallel_tasks(params.get('tasks'), params.get('inputs'))
            
        elif request_type == 'get_status':
            return self.get_workflow_status(params.get('workflow_id'))
            
        elif request_type == 'cancel_workflow':
            return self.cancel_workflow(params.get('workflow_id'))
            
        else:
            raise ValueError(f"未知请求类型: {request_type}")
            
    def execute_chain(self, chain_name: str, inputs: Dict) -> Dict:
        """
        执行预定义的任务链
        
        Args:
            chain_name: 任务链名称
            inputs: 输入参数
            
        Returns:
            执行结果
        """
        # 获取任务链
        chain = self.task_chain.get_chain(chain_name)
        
        # 创建工作流ID
        workflow_id = f"{chain_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 初始化工作流状态
        self.active_workflows[workflow_id] = {
            'status': 'running',
            'chain_name': chain_name,
            'start_time': datetime.now(),
            'current_task': None,
            'results': {},
            'errors': []
        }
        
        try:
            # 执行任务链
            results = self._execute_task_sequence(chain, inputs, workflow_id)
            
            # 更新状态
            self.active_workflows[workflow_id]['status'] = 'completed'
            self.active_workflows[workflow_id]['end_time'] = datetime.now()
            self.active_workflows[workflow_id]['results'] = results
            
            return {
                'workflow_id': workflow_id,
                'status': 'success',
                'results': results
            }
            
        except Exception as e:
            logger.error(f"工作流 {workflow_id} 执行失败: {str(e)}")
            self.active_workflows[workflow_id]['status'] = 'failed'
            self.active_workflows[workflow_id]['errors'].append(str(e))
            
            return {
                'workflow_id': workflow_id,
                'status': 'error',
                'error': str(e)
            }
            
    def execute_custom_workflow(self, tasks: List[str], inputs: Dict) -> Dict:
        """执行自定义工作流"""
        # 创建自定义任务链
        custom_chain = self.task_chain.create_custom_chain(tasks)
        
        # 使用execute_chain执行
        return self.execute_chain('custom', inputs)
        
    def execute_parallel_tasks(self, task_groups: List[List[str]], inputs: Dict) -> Dict:
        """
        并行执行任务组
        
        Args:
            task_groups: 任务组列表，每组内的任务并行执行
            inputs: 输入参数
            
        Returns:
            执行结果
        """
        workflow_id = f"parallel_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results = {}
        
        for group_idx, task_group in enumerate(task_groups):
            logger.info(f"执行并行任务组 {group_idx}: {task_group}")
            
            # 使用线程池并行执行组内任务
            futures = []
            for task_name in task_group:
                future = self.executor.submit(
                    self._execute_single_task,
                    task_name,
                    inputs,
                    results
                )
                futures.append((task_name, future))
                
            # 等待组内任务完成
            for task_name, future in futures:
                try:
                    task_result = future.result(timeout=300)  # 5分钟超时
                    results[task_name] = task_result
                except Exception as e:
                    logger.error(f"任务 {task_name} 执行失败: {str(e)}")
                    results[task_name] = {'error': str(e)}
                    
            # 更新输入，包含当前组的结果
            inputs.update(results)
            
        return {
            'workflow_id': workflow_id,
            'status': 'success',
            'results': results
        }
        
    def _execute_task_sequence(self, chain: List[Dict], inputs: Dict, 
                              workflow_id: str) -> Dict:
        """顺序执行任务链"""
        results = {}
        context = inputs.copy()
        
        for task in chain:
            # 更新当前任务
            self.active_workflows[workflow_id]['current_task'] = task['name']
            
            # 准备任务参数
            task_params = {}
            for required_input in task['required_inputs']:
                if required_input in context:
                    task_params[required_input] = context[required_input]
                elif required_input in results:
                    task_params[required_input] = results[required_input]
                else:
                    raise ValueError(f"任务 {task['name']} 缺少输入: {required_input}")
                    
            # 执行任务
            logger.info(f"执行任务: {task['name']}")
            result = self.manager.dispatch_task(task['name'], **task_params)
            
            # 保存结果
            if len(task['outputs']) == 1:
                results[task['outputs'][0]] = result
                context[task['outputs'][0]] = result
            else:
                # 多输出情况
                for idx, output_name in enumerate(task['outputs']):
                    results[output_name] = result[idx] if isinstance(result, (list, tuple)) else result
                    context[output_name] = results[output_name]
                    
        return results
        
    def _execute_single_task(self, task_name: str, inputs: Dict, 
                            shared_results: Dict) -> Any:
        """执行单个任务"""
        # 根据任务名找到对应的任务定义
        task_def = None
        for chain in self.task_chain.chains.values():
            for task in chain:
                if task['name'] == task_name:
                    task_def = task
                    break
                    
        if not task_def:
            raise ValueError(f"未找到任务定义: {task_name}")
            
        # 准备参数
        task_params = {}
        for required_input in task_def['required_inputs']:
            if required_input in inputs:
                task_params[required_input] = inputs[required_input]
            elif required_input in shared_results:
                task_params[required_input] = shared_results[required_input]
                
        # 执行任务
        return self.manager.dispatch_task(task_name, **task_params)
        
    def get_workflow_status(self, workflow_id: str) -> Dict:
        """获取工作流状态"""
        if workflow_id not in self.active_workflows:
            return {'status': 'not_found'}
            
        workflow = self.active_workflows[workflow_id]
        
        # 计算执行时间
        if 'end_time' in workflow:
            duration = (workflow['end_time'] - workflow['start_time']).total_seconds()
        else:
            duration = (datetime.now() - workflow['start_time']).total_seconds()
            
        return {
            'workflow_id': workflow_id,
            'status': workflow['status'],
            'chain_name': workflow.get('chain_name'),
            'current_task': workflow.get('current_task'),
            'duration': duration,
            'errors': workflow.get('errors', [])
        }
        
    def cancel_workflow(self, workflow_id: str) -> Dict:
        """取消工作流"""
        if workflow_id not in self.active_workflows:
            return {'status': 'not_found'}
            
        # 标记为已取消
        self.active_workflows[workflow_id]['status'] = 'cancelled'
        self.active_workflows[workflow_id]['end_time'] = datetime.now()
        
        return {'status': 'cancelled', 'workflow_id': workflow_id}
        
    async def execute_chain_async(self, chain_name: str, inputs: Dict) -> Dict:
        """异步执行任务链"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.execute_chain,
            chain_name,
            inputs
        )
        
    def cleanup_completed_workflows(self, retention_hours: int = 24):
        """清理已完成的工作流"""
        current_time = datetime.now()
        to_remove = []
        
        for workflow_id, workflow in self.active_workflows.items():
            if workflow['status'] in ['completed', 'failed', 'cancelled']:
                if 'end_time' in workflow:
                    elapsed = (current_time - workflow['end_time']).total_seconds() / 3600
                    if elapsed > retention_hours:
                        to_remove.append(workflow_id)
                        
        for workflow_id in to_remove:
            del self.active_workflows[workflow_id]
            
        logger.info(f"清理了 {len(to_remove)} 个过期工作流")