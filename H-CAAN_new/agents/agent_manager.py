import os
import logging
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

from agents.data_agent import DataAgent
from agents.model_agent import ModelAgent
from agents.evaluation_agent import EvaluationAgent
from agents.writing_agent import WritingAgent

class AgentManager:
    """
    Manager for the agent-based drug property prediction research pipeline.
    
    This class orchestrates the collaboration between specialized agents for
    data processing, model training, evaluation, and paper writing.
    """
    
    def __init__(self, openai_api_key=None, verbose=True):
        """
        Initialize the agent manager.
        
        Args:
            openai_api_key (str, optional): OpenAI API key. If None, use environment variable.
            verbose (bool): Whether to output detailed logs
        """
        # Set up logging
        self.verbose = verbose
        self.logger = self._setup_logging()
        
        # Get API key from environment variable if not provided
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            self.logger.warning("OpenAI API key not provided. Some agent functionalities may be limited.")
        
        # Initialize shared knowledge base
        self.knowledge_base = {}
        
        # Initialize agents
        self.agents = self._setup_agents()
        
        # Initialize LLM for coordination
        self.llm = self._setup_llm()
        
        # Initialize shared memory
        self.memory = ConversationBufferMemory()
        
        # Initialize conversation chain
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=self.verbose
        )
        
        self.logger.info("Agent Manager initialized successfully")
    
    def _setup_logging(self):
        """Set up logging configuration"""
        logger = logging.getLogger("AgentManager")
        
        if self.verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)
        
        return logger
    
    def _setup_llm(self):
        """Set up the language model for coordination"""
        if self.openai_api_key:
            try:
                llm = ChatOpenAI(
                    model_name="gpt-4",
                    temperature=0.2,
                    openai_api_key=self.openai_api_key
                )
                return llm
            except Exception as e:
                self.logger.error(f"Error initializing ChatOpenAI: {str(e)}")
                self.logger.warning("Falling back to simulated LLM.")
        
        # Fallback to simulated LLM
        llm = self._simulated_llm()
        return llm
    
    def _simulated_llm(self):
        """Create a simulated LLM for demonstration purposes"""
        from langchain.llms.fake import FakeListLLM
        
        responses = [
            "Based on the data analysis, I recommend focusing on molecular weight and LogP as key features.",
            "The H-CAAN model architecture should prioritize the cross-modal attention components.",
            "Ablation studies show that the chemical-aware attention mechanism contributes significantly to performance.",
            "The paper should emphasize the novel hierarchical fusion approach and its benefits.",
            "I'll coordinate the agents to complete the research pipeline efficiently."
        ]
        
        return FakeListLLM(responses=responses)
    
    def _setup_agents(self):
        """Initialize all specialized agents"""
        agents = {}
        
        # Data agent for dataset processing
        agents["data"] = DataAgent(
            knowledge_base=self.knowledge_base,
            openai_api_key=self.openai_api_key,
            verbose=self.verbose
        )
        
        # Model agent for architecture design and training
        agents["model"] = ModelAgent(
            knowledge_base=self.knowledge_base,
            openai_api_key=self.openai_api_key,
            verbose=self.verbose
        )
        
        # Evaluation agent for results analysis
        agents["evaluation"] = EvaluationAgent(
            knowledge_base=self.knowledge_base,
            openai_api_key=self.openai_api_key,
            verbose=self.verbose
        )
        
        # Writing agent for paper generation
        agents["writing"] = WritingAgent(
            knowledge_base=self.knowledge_base,
            openai_api_key=self.openai_api_key,
            verbose=self.verbose
        )
        
        return agents
    
    def get_agent(self, agent_name):
        """
        Get a specific agent by name.
        
        Args:
            agent_name (str): Name of the agent to retrieve
            
        Returns:
            Agent: The requested agent
        """
        if agent_name not in self.agents:
            self.logger.error(f"Agent '{agent_name}' not found")
            return None
        
        return self.agents[agent_name]
    
    def execute_pipeline(self, dataset_path, model_config, training_config):
        """
        Execute the full research pipeline.
        
        Args:
            dataset_path (str): Path to the dataset
            model_config (dict): Model configuration
            training_config (dict): Training configuration
            
        Returns:
            dict: Results of the pipeline execution
        """
        self.logger.info("Starting full research pipeline execution")
        
        # Step 1: Data processing
        self.logger.info("Step 1: Data processing")
        data_agent = self.agents["data"]
        data_results = data_agent.process_dataset(dataset_path)
        
        # Update knowledge base
        self.knowledge_base.update(data_results)
        
        # Step 2: Model configuration and training
        self.logger.info("Step 2: Model configuration and training")
        model_agent = self.agents["model"]
        
        # Configure model
        model_summary = model_agent.configure_model(model_config)
        
        # Train model
        training_results = model_agent.train_model(training_config)
        
        # Update knowledge base
        self.knowledge_base.update({
            "model_summary": model_summary,
            "training_results": training_results
        })
        
        # Step 3: Evaluation
        self.logger.info("Step 3: Evaluation and analysis")
        evaluation_agent = self.agents["evaluation"]
        evaluation_results = evaluation_agent.evaluate_results(
            training_results, 
            model_config
        )
        
        # Update knowledge base
        self.knowledge_base.update({
            "evaluation_results": evaluation_results
        })
        
        # Step 4: Paper generation
        self.logger.info("Step 4: Paper generation")
        writing_agent = self.agents["writing"]
        paper = writing_agent.generate_paper()
        
        # Compile final results
        pipeline_results = {
            "data_results": data_results,
            "model_summary": model_summary,
            "training_results": training_results,
            "evaluation_results": evaluation_results,
            "paper": paper
        }
        
        self.logger.info("Research pipeline execution completed successfully")
        
        return pipeline_results
    
    def coordinate_agents(self, query):
        """
        Coordinate agents based on a high-level query.
        
        Args:
            query (str): High-level research query
            
        Returns:
            str: Response from the coordination
        """
        # Create tools for each agent
        tools = []
        
        # Data agent tools
        tools.append(
            Tool(
                name="DataProcessing",
                func=self.agents["data"].process_dataset,
                description="Analyzes and processes molecular datasets. Use this for data preparation tasks."
            )
        )
        
        # Model agent tools
        tools.append(
            Tool(
                name="ModelConfiguration",
                func=self.agents["model"].configure_model,
                description="Configures the H-CAAN model architecture. Use this for model design tasks."
            )
        )
        
        tools.append(
            Tool(
                name="ModelTraining",
                func=self.agents["model"].train_model,
                description="Trains the H-CAAN model. Use this for model training and optimization tasks."
            )
        )
        
        # Evaluation agent tools
        tools.append(
            Tool(
                name="ResultsEvaluation",
                func=self.agents["evaluation"].evaluate_results,
                description="Analyzes and evaluates model results. Use this for performance assessment tasks."
            )
        )
        
        # Writing agent tools
        tools.append(
            Tool(
                name="PaperGeneration",
                func=self.agents["writing"].generate_paper,
                description="Generates research paper drafts. Use this for paper writing tasks."
            )
        )
        
        # Use conversation chain to respond
        response = self.conversation.predict(input=query)
        
        return response
    
    def get_knowledge_base(self):
        """
        Get the current state of the knowledge base.
        
        Returns:
            dict: Current knowledge base
        """
        return self.knowledge_base
    
    def update_knowledge_base(self, key, value):
        """
        Update a specific entry in the knowledge base.
        
        Args:
            key (str): Key to update
            value: Value to set
        """
        self.knowledge_base[key] = value
        self.logger.debug(f"Knowledge base updated: {key}")