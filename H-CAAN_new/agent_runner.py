"""
LangChain智能体统一调度与管理
"""
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.schema import AgentAction, AgentFinish
from typing import List, Tuple, Any, Union
import re
import logging

from agents.multi_agent_manager import MultiAgentManager
from workflows.router import WorkflowRouter

logger = logging.getLogger(__name__)

class HCAANAgentRunner:
    """H-CAAN智能体运行器"""
    
    def __init__(self):
        self.manager = MultiAgentManager()
        self.router = WorkflowRouter()
        self.tools = self._create_tools()
        
    def _create_tools(self) -> List[Tool]:
        """创建LangChain工具"""
        tools = [
            Tool(
                name="LoadData",
                func=lambda x: self.manager.dispatch_task('load_data', data_path=x),
                description="加载分子数据文件"
            ),
            Tool(
                name="PreprocessData",
                func=lambda x: self.manager.dispatch_task('preprocess_data', raw_data=x),
                description="预处理原始数据"
            ),
            Tool(
                name="FuseFeatures",
                func=lambda x: self.manager.dispatch_task('fuse_features', processed_data=x),
                description="融合多模态特征"
            ),
            Tool(
                name="TrainModel",
                func=lambda x: self.manager.dispatch_task('train_model', **x),
                description="训练预测模型"
            ),
            Tool(
                name="GenerateReport",
                func=lambda x: self.manager.dispatch_task('explain', **x),
                description="生成解释报告"
            ),
            Tool(
                name="GeneratePaper",
                func=lambda x: self.manager.dispatch_task('generate_paper', **x),
                description="自动生成论文"
            )
        ]
        return tools
        
    def run_workflow(self, workflow_name: str, inputs: dict) -> Any:
        """运行工作流"""
        logger.info(f"运行工作流: {workflow_name}")
        
        try:
            result = self.router.route_request({
                'type': 'execute_chain',
                'params': {
                    'chain_name': workflow_name,
                    'inputs': inputs
                }
            })
            
            return result
            
        except Exception as e:
            logger.error(f"工作流执行失败: {str(e)}")
            return {'status': 'error', 'message': str(e)}
            
    def parse_command(self, command: str) -> Tuple[str, dict]:
        """解析用户命令"""
        # 简单的命令解析
        parts = command.strip().split()
        
        if not parts:
            return None, {}
            
        action = parts[0].lower()
        params = {}
        
        # 解析参数
        for i in range(1, len(parts)):
            if '=' in parts[i]:
                key, value = parts[i].split('=', 1)
                params[key] = value
                
        return action, params
        
    def execute_command(self, command: str) -> Any:
        """执行命令"""
        action, params = self.parse_command(command)
        
        if action == 'train':
            return self.run_workflow('full_pipeline', params)
        elif action == 'predict':
            return self.run_workflow('prediction_only', params)
        elif action == 'analyze':
            return self.run_workflow('analysis_only', params)
        else:
            return {'status': 'error', 'message': f'未知命令: {action}'}

if __name__ == "__main__":
    # 测试运行
    runner = HCAANAgentRunner()
    
    # 示例命令
    commands = [
        "train data_path=data/raw/example.csv",
        "predict model_path=data/models/model.pkl data_path=data/raw/test.csv",
        "analyze model_path=data/models/model.pkl"
    ]
    
    for cmd in commands:
        print(f"\n执行命令: {cmd}")
        result = runner.execute_command(cmd)
        print(f"结果: {result}")