import os
import logging
import json
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
import re
from pathlib import Path
import time
from urllib.parse import quote_plus

from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import PubmedQueryRun

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)

# Create formatters
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)

# Add handlers to the logger
logger.addHandler(c_handler)

class ResearchAgent:
    """
    Agent for literature search and analysis for H-CAAN project
    
    This agent is responsible for:
    1. Searching relevant literature for multimodal learning in drug discovery
    2. Summarizing and analyzing papers related to molecular property prediction
    3. Identifying state-of-the-art methods and performance benchmarks
    4. Supporting literature review section of the paper
    """
    
    def __init__(self, api_keys: Dict = None, model: str = "gpt-4"):
        """
        Initialize the research agent with necessary components
        
        Args:
            api_keys: Dictionary containing API keys (openai, google_search, etc.)
            model: The LLM model to use for the agent
        """
        self.name = "ResearchAgent"
        self.api_keys = api_keys or {}
        self.model_name = model
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.results_dir = Path("results/research")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize resources
        self._init_llm()
        self._init_tools()
        self._init_agent()
        
        # Initialize research databases
        self.papers_db = self._load_papers_db()
        self.sota_results = self._load_sota_results()
        
        logger.info(f"Initialized {self.name} with {self.model_name}")
    
    def _init_llm(self):
        """Initialize the LLM for the agent"""
        try:
            if 'OPENAI_API_KEY' in os.environ or 'openai' in self.api_keys:
                api_key = os.environ.get('OPENAI_API_KEY', self.api_keys.get('openai', ''))
                self.llm = ChatOpenAI(
                    openai_api_key=api_key,
                    model_name=self.model_name,
                    temperature=0.1
                )
                logger.info(f"Initialized {self.model_name} LLM")
            else:
                raise ValueError("OpenAI API key not found")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            # Fallback to a local model or stub for development
            self.llm = None
            logger.warning("Using fallback LLM (stub for development)")
    
    def _init_tools(self):
        """Initialize tools for the agent"""
        tools = []
        
        # Google Search tool
        try:
            if 'google_search' in self.api_keys:
                search = GoogleSearchAPIWrapper(
                    google_api_key=self.api_keys['google_search'],
                    google_cse_id=self.api_keys['google_cse_id']
                )
                tools.append(
                    Tool(
                        name="Google Search",
                        description="Search for information on Google about molecular property prediction, multimodal learning, drug discovery, and related topics",
                        func=search.run
                    )
                )
        except Exception as e:
            logger.error(f"Error initializing Google Search tool: {e}")
            # Add mock search for development
            tools.append(
                Tool(
                    name="Google Search",
                    description="Search for information on Google (mock for development)",
                    func=lambda query: f"Mock search results for: {query}"
                )
            )
        
        # Arxiv tool
        try:
            arxiv = ArxivAPIWrapper()
            tools.append(
                Tool(
                    name="Arxiv Search",
                    description="Search for papers on Arxiv related to molecular property prediction, multimodal learning, transformers, graph neural networks",
                    func=arxiv.run
                )
            )
        except Exception as e:
            logger.error(f"Error initializing Arxiv tool: {e}")
            tools.append(
                Tool(
                    name="Arxiv Search",
                    description="Search for papers on Arxiv (mock for development)",
                    func=lambda query: f"Mock Arxiv results for: {query}"
                )
            )
        
        # PubMed tool
        try:
            pubmed = PubmedQueryRun()
            tools.append(
                Tool(
                    name="PubMed Search",
                    description="Search for papers on PubMed related to molecular property prediction, drug discovery, QSAR",
                    func=pubmed.run
                )
            )
        except Exception as e:
            logger.error(f"Error initializing PubMed tool: {e}")
            tools.append(
                Tool(
                    name="PubMed Search",
                    description="Search for papers on PubMed (mock for development)",
                    func=lambda query: f"Mock PubMed results for: {query}"
                )
            )
        
        # Custom tool for scraping paper details
        tools.append(
            Tool(
                name="Paper Details",
                description="Get details about a specific paper given its DOI, Arxiv ID, or URL",
                func=self._get_paper_details
            )
        )
        
        # Custom tool for analyzing SOTA results
        tools.append(
            Tool(
                name="SOTA Analysis",
                description="Analyze state-of-the-art results for a specific dataset or task",
                func=self._analyze_sota
            )
        )
        
        self.tools = tools
        logger.info(f"Initialized {len(tools)} tools for research agent")
    
    def _init_agent(self):
        """Initialize the agent with tools and prompt"""
        if not self.llm:
            self.agent = None
            return
        
        prefix = """You are a research assistant for H-CAAN (Hierarchical Cross-modal Adaptive Attention Network), 
        a novel multimodal deep learning model for molecular property prediction.
        
        Your task is to find and analyze relevant literature for this project. Focus on:
        
        1. Recent papers on multimodal learning for drug discovery and molecular property prediction
        2. State-of-the-art methods for integrating SMILES, molecular fingerprints, molecular graphs, and language model embeddings
        3. Transformer-based approaches for molecular representations
        4. Attention mechanisms for multimodal fusion
        5. Benchmarks and datasets for molecular property prediction
        
        You have access to the following tools:"""
        
        format_instructions = """To use a tool, please use the following format:
        
        ```
        Thought: I need to find information about X
        Action: tool_name
        Action Input: the input to the tool
        ```
        
        After using a tool, you'll get an observation. Please continue with your thought process:
        
        ```
        Observation: result of the tool
        Thought: I now know X, and need to find out about Y
        Action: another_tool
        Action Input: the input for this tool
        ```
        
        When you have gathered enough information to provide a comprehensive answer, respond with:
        
        ```
        Thought: I now have all the information I need to answer.
        Final Answer: Your detailed, well-structured answer with specific papers, methods, and findings.
        ```
        
        Make sure to cite papers properly with authors, year, and title.
        """
        
        suffix = """Begin!
        
        Previous conversation history:
        {chat_history}
        
        Question: {input}
        {agent_scratchpad}"""
        
        prompt = ZeroShotAgent.create_prompt(
            self.tools,
            prefix=prefix,
            format_instructions=format_instructions,
            suffix=suffix,
            input_variables=["input", "chat_history", "agent_scratchpad"]
        )
        
        agent = ZeroShotAgent(
            llm_chain=LLMChain(llm=self.llm, prompt=prompt),
            tools=self.tools, 
            verbose=True
        )
        
        self.agent = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True
        )
    
    def _load_papers_db(self) -> pd.DataFrame:
        """Load or initialize the papers database"""
        db_path = self.results_dir / "papers_database.csv"
        
        if db_path.exists():
            try:
                return pd.read_csv(db_path)
            except Exception as e:
                logger.error(f"Error loading papers database: {e}")
        
        # Create a new database
        papers_db = pd.DataFrame({
            'title': [],
            'authors': [],
            'year': [],
            'journal': [],
            'doi': [],
            'url': [],
            'abstract': [],
            'keywords': [],
            'methods': [],
            'datasets': [],
            'results': [],
            'relevance': []
        })
        
        # Save empty database
        papers_db.to_csv(db_path, index=False)
        return papers_db
    
    def _load_sota_results(self) -> Dict:
        """Load or initialize SOTA results"""
        sota_path = self.results_dir / "sota_results.json"
        
        if sota_path.exists():
            try:
                with open(sota_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading SOTA results: {e}")
        
        # Initialize new SOTA results
        sota_results = {
            'datasets': {
                'Delaney': {
                    'best_model': 'Chemprop',
                    'best_rmse': 0.58,
                    'best_r2': 0.91,
                    'models': {
                        'Chemprop': {'rmse': 0.58, 'r2': 0.91, 'method': 'GNN', 'paper': 'Yang et al., 2019'},
                        'AttentiveFP': {'rmse': 0.62, 'r2': 0.89, 'method': 'GNN+Attention', 'paper': 'Xiong et al., 2019'},
                        'MPNN': {'rmse': 0.65, 'r2': 0.88, 'method': 'MPNN', 'paper': 'Gilmer et al., 2017'}
                    }
                },
                'Lipophilicity': {
                    'best_model': 'MolBERT',
                    'best_rmse': 0.56,
                    'best_r2': 0.92,
                    'models': {
                        'MolBERT': {'rmse': 0.56, 'r2': 0.92, 'method': 'Transformer', 'paper': 'Fabian et al., 2020'},
                        'Chemprop': {'rmse': 0.61, 'r2': 0.90, 'method': 'GNN', 'paper': 'Yang et al., 2019'},
                        'MPNN': {'rmse': 0.69, 'r2': 0.87, 'method': 'MPNN', 'paper': 'Gilmer et al., 2017'}
                    }
                },
                'BACE': {
                    'best_model': 'MMFDL',
                    'best_auc': 0.91,
                    'best_accuracy': 0.87,
                    'models': {
                        'MMFDL': {'auc': 0.91, 'accuracy': 0.87, 'method': 'Multimodal', 'paper': 'Lu et al., 2024'},
                        'MolBERT': {'auc': 0.88, 'accuracy': 0.85, 'method': 'Transformer', 'paper': 'Fabian et al., 2020'},
                        'AttentiveFP': {'auc': 0.86, 'accuracy': 0.84, 'method': 'GNN+Attention', 'paper': 'Xiong et al., 2019'}
                    }
                }
            },
            'methods': {
                'GNN': {
                    'description': 'Graph Neural Networks for molecular property prediction',
                    'key_papers': [
                        'Yang et al., 2019 - Analyzing learned molecular representations for property prediction',
                        'Kearnes et al., 2016 - Molecular graph convolutions: moving beyond fingerprints'
                    ],
                    'strengths': ['Captures molecular structure', 'Handles variable-size molecules'],
                    'weaknesses': ['May miss global patterns', 'Limited by graph representation']
                },
                'Transformer': {
                    'description': 'Transformer models applied to SMILES or other molecular representations',
                    'key_papers': [
                        'Fabian et al., 2020 - MolBERT: Molecular representation learning with language models',
                        'Chithrananda et al., 2020 - ChemBERTa: Large-scale self-supervised pretraining for molecular property prediction'
                    ],
                    'strengths': ['Captures long-range dependencies', 'Learns from large unlabeled data'],
                    'weaknesses': ['May lose structural information', 'Requires large datasets']
                },
                'Multimodal': {
                    'description': 'Models that integrate multiple molecular representations',
                    'key_papers': [
                        'Lu et al., 2024 - Multimodal fused deep learning for drug property prediction: Integrating chemical language and molecular graph',
                        'Abdel-Aty and Gould, 2022 - Large-Scale Distributed Training of Transformers for Chemical Fingerprinting'
                    ],
                    'strengths': ['Leverages complementary information', 'More robust predictions'],
                    'weaknesses': ['Complex architecture', 'Harder to train', 'May overfit on small datasets']
                }
            }
        }
        
        # Save initialized SOTA results
        with open(sota_path, 'w') as f:
            json.dump(sota_results, f, indent=2)
        
        return sota_results
    
    def _get_paper_details(self, query: str) -> str:
        """
        Get details about a paper from its DOI, Arxiv ID, or URL
        
        Args:
            query: DOI, Arxiv ID, or URL of the paper
            
        Returns:
            Structured information about the paper
        """
        # Extract DOI if URL is provided
        doi_match = re.search(r'10\.\d{4,}\/[^\/\s]+', query)
        if doi_match:
            doi = doi_match.group(0)
        else:
            doi = query
        
        # For demonstration purposes, return mock data
        # In a real implementation, this would call APIs to get paper details
        
        # Check if we already have this paper in our database
        if not self.papers_db.empty:
            paper_match = self.papers_db[self.papers_db['doi'].str.contains(doi, na=False)]
            if not paper_match.empty:
                paper = paper_match.iloc[0]
                return f"""
                Title: {paper['title']}
                Authors: {paper['authors']}
                Year: {paper['year']}
                Journal: {paper['journal']}
                DOI: {paper['doi']}
                URL: {paper['url']}
                Abstract: {paper['abstract']}
                Keywords: {paper['keywords']}
                Methods: {paper['methods']}
                Datasets: {paper['datasets']}
                Results: {paper['results']}
                Relevance: {paper['relevance']}
                """
        
        # Mock data for demonstration
        if "multimodal" in query.lower():
            return """
            Title: Multimodal fused deep learning for drug property prediction: Integrating chemical language and molecular graph
            Authors: Lu X, Xie L, Xu L, Mao R, Xu X, Chang S
            Year: 2024
            Journal: Computational and Structural Biotechnology Journal
            DOI: 10.1016/j.csbj.2024.04.030
            URL: https://www.sciencedirect.com/science/article/pii/S2001037024000308
            Abstract: Accurately predicting molecular properties is a challenging but essential task in drug discovery. Recently, many mono-modal deep learning methods have been successfully applied to molecular property prediction. However, mono-modal learning is inherently limited as it relies solely on a single modality of molecular representation, which restricts a comprehensive understanding of drug molecules. To overcome the limitations, we propose a multimodal fused deep learning (MMFDL) model to leverage information from different molecular representations. Specifically, we construct a triple-modal learning model by employing Transformer-Encoder, Bidirectional Gated Recurrent Unit (BiGRU), and graph convolutional network (GCN) to process three modalities of information from chemical language and molecular graph: SMILES-encoded vectors, ECFP fingerprints, and molecular graphs, respectively.
            Keywords: Multimodal learning, Deep learning, Drug discovery, Transformer, Graph
            Methods: Triple-modal integration, Transformer-Encoder for SMILES, BiGRU for ECFP, GCN for molecular graphs
            Datasets: Delaney, Llinas2020, Lipophilicity, SAMPL, BACE, pKa
            Results: MMFDL outperformed mono-modal models across all datasets. For Delaney solubility dataset, achieved RMSE of 0.620 and Pearson coefficient of 0.96.
            Relevance: High - Directly relevant to H-CAAN as it employs multimodal learning for molecular property prediction
            """
        elif "transformer" in query.lower() or "bert" in query.lower():
            return """
            Title: Large-Scale Distributed Training of Transformers for Chemical Fingerprinting
            Authors: Abdel-Aty H, Gould IR
            Year: 2022
            Journal: Journal of Chemical Information and Modeling
            DOI: 10.1021/acs.jcim.2c00715
            URL: https://pubs.acs.org/doi/full/10.1021/acs.jcim.2c00715
            Abstract: Transformer models have become a popular choice for various machine learning tasks due to their often outstanding performance. Recently, transformers have been used in chemistry for classifying reactions, reaction prediction, physiochemical property prediction, and more. These models require huge amounts of data and localized compute to train effectively. In this work, we demonstrate that these models can successfully be trained for chemical problems in a distributed manner across many computers—a more common scenario for chemistry institutions. We introduce MFBERT: Molecular Fingerprints through Bidirectional Encoder Representations from Transformers. We use distributed computing to pre-train a transformer model on one of the largest aggregate datasets in chemical literature and achieve state-of-the-art scores on a virtual screening benchmark for molecular fingerprints.
            Keywords: Transformer, BERT, molecular fingerprints, chemical fingerprinting, distributed training
            Methods: RoBERTa architecture, distributed training, virtual screening
            Datasets: GDB-13, Zinc 15, PubChem, ChEMBL, USPTO
            Results: MFBERT outperforms current state-of-the-art methods with an improvement of 15% in AUCROC and 70% for BEDROC20
            Relevance: High - MFBERT is directly applicable as one of the modalities in H-CAAN
            """
        else:
            return f"No detailed information found for query: {query}. Try a different DOI or search term."
    
    def _analyze_sota(self, query: str) -> str:
        """
        Analyze state-of-the-art results for a dataset or method
        
        Args:
            query: The dataset or method to analyze
            
        Returns:
            Analysis of SOTA results
        """
        # Check if query is about a dataset
        for dataset, data in self.sota_results['datasets'].items():
            if dataset.lower() in query.lower():
                models_data = data['models']
                models_list = []
                
                for model, metrics in models_data.items():
                    metrics_str = ", ".join([f"{k}: {v}" for k, v in metrics.items() if k not in ['method', 'paper']])
                    models_list.append(f"- {model} ({metrics['method']}): {metrics_str} [Source: {metrics['paper']}]")
                
                models_summary = "\n".join(models_list)
                
                return f"""
                State-of-the-Art Analysis for {dataset} Dataset:
                
                Best performing model: {data['best_model']}
                Key performance metrics:
                {', '.join([f"{k.replace('best_', '')}: {v}" for k, v in data.items() if k.startswith('best_')])}
                
                All models performance:
                {models_summary}
                
                Trends and observations:
                - The best results are achieved by {data['best_model']} using {models_data[data['best_model']]['method']} approach
                - Performance differences between methods suggest that {dataset} benefits from {models_data[data['best_model']]['method']} architectures
                """
        
        # Check if query is about a method
        for method, data in self.sota_results['methods'].items():
            if method.lower() in query.lower():
                key_papers = "\n".join([f"- {paper}" for paper in data['key_papers']])
                strengths = "\n".join([f"- {strength}" for strength in data['strengths']])
                weaknesses = "\n".join([f"- {weakness}" for weakness in data['weaknesses']])
                
                return f"""
                Analysis of {method} Methods:
                
                Description: {data['description']}
                
                Key papers:
                {key_papers}
                
                Strengths:
                {strengths}
                
                Weaknesses:
                {weaknesses}
                
                Application in molecular property prediction:
                This approach has been successfully applied to various molecular property prediction tasks and shows particular promise for H-CAAN integration.
                """
        
        # If no match is found
        return f"No SOTA analysis available for '{query}'. Try a specific dataset (e.g., 'Delaney', 'Lipophilicity', 'BACE') or method (e.g., 'GNN', 'Transformer', 'Multimodal')."
    
    def search_literature(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search for relevant literature based on a query
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            List of paper details
        """
        if not self.agent:
            # Mock implementation for development
            logger.warning("Using mock literature search implementation")
            
            if "multimodal" in query.lower():
                papers = [
                    {
                        'title': 'Multimodal fused deep learning for drug property prediction',
                        'authors': 'Lu X, et al.',
                        'year': 2024,
                        'journal': 'Computational and Structural Biotechnology Journal',
                        'relevance': 'High'
                    },
                    {
                        'title': 'Multi-modality attribute learning-based method for drug-protein interaction prediction',
                        'authors': 'Dong W, et al.',
                        'year': 2023,
                        'journal': 'Briefings in Bioinformatics',
                        'relevance': 'Medium'
                    }
                ]
            else:
                papers = [
                    {
                        'title': 'Large-Scale Distributed Training of Transformers for Chemical Fingerprinting',
                        'authors': 'Abdel-Aty H, Gould IR',
                        'year': 2022,
                        'journal': 'Journal of Chemical Information and Modeling',
                        'relevance': 'High'
                    },
                    {
                        'title': 'Analyzing learned molecular representations for property prediction',
                        'authors': 'Yang K, et al.',
                        'year': 2019,
                        'journal': 'Journal of Chemical Information and Modeling',
                        'relevance': 'Medium'
                    }
                ]
            
            return papers[:max_results]
        
        # Format query for the agent
        agent_query = f"""
        I need to search for literature relevant to molecular property prediction.
        
        Please search for papers related to: {query}
        
        Focus on recent papers (last 5 years) and high-impact journals.
        Provide details for each paper: title, authors, year, journal, relevance to multimodal learning.
        Limit to the {max_results} most relevant papers.
        """
        
        # Execute agent
        response = self.agent.run(agent_query)
        
        # Parse response to extract papers
        # This would need a more sophisticated parser in a real implementation
        papers = []
        lines = response.split('\n')
        current_paper = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_paper and 'title' in current_paper:
                    papers.append(current_paper)
                    current_paper = {}
                continue
            
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if key == 'title':
                    if current_paper and 'title' in current_paper:
                        papers.append(current_paper)
                        current_paper = {}
                    current_paper['title'] = value
                elif key in ['authors', 'year', 'journal', 'relevance']:
                    current_paper[key] = value
        
        # Add last paper if not added
        if current_paper and 'title' in current_paper:
            papers.append(current_paper)
        
        return papers[:max_results]
    
    def analyze_papers(self, papers: List[Dict]) -> Dict:
        """
        Analyze a list of papers to extract key insights
        
        Args:
            papers: List of paper details
            
        Returns:
            Analysis of the papers
        """
        if not self.agent:
            # Mock implementation for development
            logger.warning("Using mock paper analysis implementation")
            
            return {
                'key_methods': [
                    'Transformer-based encoders for SMILES',
                    'Graph Neural Networks for molecular graphs',
                    'Cross-modal attention for multimodal fusion'
                ],
                'key_datasets': [
                    'Delaney solubility dataset',
                    'Lipophilicity dataset',
                    'BACE classification dataset'
                ],
                'key_results': [
                    'Multimodal approaches consistently outperform mono-modal approaches',
                    'Attention mechanisms improve information integration',
                    'Pre-trained language models provide valuable molecular representations'
                ],
                'trends': [
                    'Increasing focus on multimodal learning',
                    'Rise of transformer architectures for molecular representation',
                    'Integration of chemical knowledge into deep learning models'
                ],
                'research_gaps': [
                    'Limited exploration of hierarchical fusion strategies',
                    'Lack of adaptive weighting mechanisms for different molecular representations',
                    'Need for more interpretable models'
                ]
            }
        
        # Format papers for the agent
        papers_text = "\n\n".join([
            f"Title: {paper.get('title', 'Unknown')}\n"
            f"Authors: {paper.get('authors', 'Unknown')}\n"
            f"Year: {paper.get('year', 'Unknown')}\n"
            f"Journal: {paper.get('journal', 'Unknown')}\n"
            f"Abstract: {paper.get('abstract', 'Not available')}"
            for paper in papers
        ])
        
        agent_query = f"""
        Please analyze the following papers and extract key insights relevant to multimodal learning for molecular property prediction:
        
        {papers_text}
        
        Provide the following analysis:
        1. Key methods mentioned in these papers
        2. Key datasets used
        3. Important results and findings
        4. Research trends in this area
        5. Research gaps that could be addressed
        
        Focus on aspects that are relevant to H-CAAN (Hierarchical Cross-modal Adaptive Attention Network).
        """
        
        # Execute agent
        response = self.agent.run(agent_query)
        
        # Parse response to extract analysis
        # A more sophisticated parser would be needed in a real implementation
        analysis = {
            'key_methods': [],
            'key_datasets': [],
            'key_results': [],
            'trends': [],
            'research_gaps': []
        }
        
        current_section = None
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('1. Key methods') or 'Key methods' in line:
                current_section = 'key_methods'
                continue
            elif line.startswith('2. Key datasets') or 'Key datasets' in line:
                current_section = 'key_datasets'
                continue
            elif line.startswith('3. Important results') or 'Key results' in line:
                current_section = 'key_results'
                continue
            elif line.startswith('4. Research trends') or 'Trends' in line:
                current_section = 'trends'
                continue
            elif line.startswith('5. Research gaps') or 'Research gaps' in line:
                current_section = 'research_gaps'
                continue
            
            if current_section and line.startswith('-'):
                item = line[1:].strip()
                if item:
                    analysis[current_section].append(item)
        
        return analysis
    
    def generate_literature_review(self, topic: str) -> str:
        """
        Generate a literature review section on a specific topic
        
        Args:
            topic: The topic for the literature review
            
        Returns:
            Formatted literature review section
        """
        if not self.agent:
            # Mock implementation for development
            logger.warning("Using mock literature review generation")
            
            if topic.lower() == "multimodal learning for molecular property prediction":
                return """
                ## Related Work
                
                ### Multimodal Learning for Molecular Property Prediction
                
                Recent advancements in deep learning have led to significant improvements in molecular property prediction. Traditional approaches typically rely on a single molecular representation, such as SMILES strings, molecular fingerprints, or molecular graphs (Yang et al., 2019). While these mono-modal approaches have achieved considerable success, they are inherently limited by the information captured in each representation.
                
                Multimodal learning approaches aim to overcome these limitations by integrating multiple molecular representations. Lu et al. (2024) proposed MMFDL, a triple-modal fusion model that combines SMILES-encoded vectors processed by Transformer-Encoder, ECFP fingerprints processed by BiGRU, and molecular graphs processed by GCN. Their results demonstrated that multimodal approaches consistently outperform mono-modal methods across various datasets, including Delaney, Lipophilicity, and BACE.
                
                Another notable work is by Dong et al. (2023), who developed MMA-DPI, a multimodal attribute learning framework for drug-protein interaction prediction. Their model leverages both molecular transformer and graph convolutional networks to capture complementary information from different representations.
                
                ### Attention Mechanisms in Molecular Modeling
                
                Attention mechanisms have emerged as a powerful tool for molecular modeling. Xiong et al. (2019) introduced AttentiveFP, which uses graph attention networks to focus on relevant substructures for property prediction. Maziarka et al. (2020) proposed Molecule Attention Transformer (MAT), which incorporates 3D structural information into the self-attention mechanism.
                
                In the context of multimodal learning, cross-modal attention has been explored for integrating different molecular representations. Yang et al. (2022) developed Modality-DTA, which employs a multimodal fusion strategy with cross-attention for drug-target affinity prediction.
                
                ### Pre-trained Molecular Language Models
                
                Pre-trained molecular language models have shown great promise in capturing chemical knowledge from large unlabeled datasets. Abdel-Aty and Gould (2022) introduced MFBERT, a molecular fingerprinting method based on bidirectional encoder representations from transformers. Trained on a massive chemical dataset, MFBERT achieved state-of-the-art performance on molecular fingerprinting tasks.
                
                Other notable pre-trained models include MolBERT (Fabian et al., 2020) and ChemBERTa (Chithrananda et al., 2020), which adapt BERT-like architectures for molecular representation learning. These models have demonstrated strong performance across various molecular property prediction tasks.
                
                ### Research Gaps
                
                Despite these advancements, several research gaps remain. First, most existing multimodal approaches use simple fusion strategies, such as concatenation or weighted summation, which may not fully capture the complex relationships between different modalities. Second, the importance of each modality often varies across different molecules and prediction tasks, suggesting the need for adaptive weighting mechanisms. Third, the hierarchical nature of molecular information, from atomic-level features to global molecular properties, is not explicitly modeled in current approaches.
                
                Our proposed H-CAAN addresses these gaps by introducing a hierarchical fusion architecture with gated cross-modal attention units and task-specific weight generation, enabling more effective integration of information across modalities.
                """
            elif topic.lower() == "attention mechanisms":
                return """
                ## Related Work
                
                ### Attention Mechanisms in Deep Learning
                
                Attention mechanisms have revolutionized deep learning across various domains. Originally introduced by Bahdanau et al. (2015) for neural machine translation, attention allows models to focus on relevant parts of the input when making predictions. Vaswani et al. (2017) proposed the Transformer architecture, which relies entirely on self-attention mechanisms and has become the foundation for state-of-the-art models in natural language processing.
                
                In the context of molecular modeling, attention mechanisms have been adapted to capture important structural and chemical information. Xiong et al. (2019) introduced AttentiveFP, which uses graph attention networks to identify important atoms and bonds for molecular property prediction. Their model achieved state-of-the-art performance on several benchmark datasets, demonstrating the effectiveness of attention for molecular representation learning.
                
                ### Cross-Modal Attention for Multimodal Learning
                
                Cross-modal attention extends the attention mechanism to multimodal settings, allowing the model to attend to relevant information across different modalities. Lu et al. (2016) introduced co-attention for visual question answering, where the model jointly reasons about image and text features. Building on this work, Yu et al. (2019) proposed a deep modular co-attention network that performs multi-step reasoning with co-attention.
                
                In the chemistry domain, Yang et al. (2022) developed Modality-DTA, which employs cross-modal attention between drug and protein representations for drug-target affinity prediction. Their approach demonstrated superior performance compared to single-modal methods, highlighting the benefits of cross-modal attention for integrating heterogeneous chemical information.
                
                ### Gated Attention Mechanisms
                
                Gated attention mechanisms introduce additional control over information flow in attention networks. Li et al. (2018) proposed a gated attention network for knowledge graph embedding, where a gating mechanism determines the importance of different relation patterns. Similarly, Zheng et al. (2020) introduced a gated attention mechanism for graph neural networks, allowing the model to adaptively focus on important neighbor information.
                
                In multimodal learning, gating mechanisms have been used to control information exchange between modalities. Liu et al. (2018) proposed a learn-to-combine modalities approach, where a gating network determines the contribution of each modality to the final prediction. This approach allows the model to dynamically adjust the importance of different modalities based on the input.
                
                ### Chemical-Aware Attention
                
                Chemical-aware attention mechanisms incorporate domain-specific knowledge into the attention computation. Maziarka et al. (2020) proposed Molecule Attention Transformer (MAT), which incorporates 3D structural information and inter-atomic distances into the self-attention mechanism. This approach allows the model to capture spatial relationships between atoms that are not explicitly represented in the molecular graph.
                
                Similarly, Wang et al. (2021) introduced a chemistry-informed self-attention mechanism that incorporates chemical rules and heuristics into the attention computation. By integrating domain knowledge, their model achieved improved performance on reaction prediction tasks compared to standard attention mechanisms.
                
                ### Research Gaps
                
                Despite these advancements, several challenges remain in the application of attention mechanisms for molecular property prediction. First, most existing approaches focus on single-modal attention, with limited exploration of cross-modal attention for integrating different molecular representations. Second, the hierarchical nature of molecular information, from atomic-level features to global molecular properties, is not explicitly modeled in current attention mechanisms. Third, adaptive weighting of attention across different modalities and molecular complexity levels remains underexplored.
                
                Our proposed H-CAAN addresses these gaps by introducing hierarchical cross-modal adaptive attention mechanisms that dynamically integrate information across different modalities and chemical space representations.
                """
            else:
                return f"No pre-generated literature review available for topic: {topic}"
        
        # Format query for the agent
        agent_query = f"""
        Please generate a comprehensive literature review section on the topic: {topic}
        
        The literature review should:
        1. Cover recent advances in the field (last 5 years)
        2. Highlight key papers and their contributions
        3. Identify research trends and gaps
        4. Relate to H-CAAN (Hierarchical Cross-modal Adaptive Attention Network) when appropriate
        
        Format the literature review as a well-structured academic text with proper citations.
        Organize it with clear subsections and paragraphs.
        Make it around 800-1000 words.
        """
        
        # Execute agent
        response = self.agent.run(agent_query)
        
        return response
    
    def identify_sota_benchmarks(self, dataset_name: str) -> Dict:
        """
        Identify state-of-the-art benchmarks for a specific dataset
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with SOTA benchmarks
        """
        # Check if we have SOTA results for this dataset
        for dataset, data in self.sota_results['datasets'].items():
            if dataset.lower() == dataset_name.lower():
                return {
                    'dataset': dataset,
                    'best_model': data['best_model'],
                    'metrics': {k.replace('best_', ''): v for k, v in data.items() if k.startswith('best_')},
                    'all_models': data['models']
                }
        
        if not self.agent:
            # Mock implementation for development
            logger.warning("Using mock SOTA benchmark identification")
            
            if "delaney" in dataset_name.lower():
                return {
                    'dataset': 'Delaney',
                    'best_model': 'Chemprop',
                    'metrics': {'rmse': 0.58, 'r2': 0.91},
                    'all_models': {
                        'Chemprop': {'rmse': 0.58, 'r2': 0.91, 'method': 'GNN', 'paper': 'Yang et al., 2019'},
                        'AttentiveFP': {'rmse': 0.62, 'r2': 0.89, 'method': 'GNN+Attention', 'paper': 'Xiong et al., 2019'},
                        'MPNN': {'rmse': 0.65, 'r2': 0.88, 'method': 'MPNN', 'paper': 'Gilmer et al., 2017'}
                    }
                }
            else:
                return {
                    'dataset': dataset_name,
                    'best_model': 'Unknown',
                    'metrics': {},
                    'all_models': {}
                }
        
        # Format query for the agent
        agent_query = f"""
        I need to identify state-of-the-art benchmark results for the {dataset_name} dataset in molecular property prediction.
        
        Please search for:
        1. The best-performing models on this dataset
        2. Their key performance metrics (RMSE, R², etc.)
        3. The methods they use
        4. The papers that report these results
        
        Provide a comprehensive summary of the SOTA benchmark results.
        """
        
        # Execute agent
        response = self.agent.run(agent_query)
        
        # Parse response to extract SOTA benchmarks
        # This would need a more sophisticated parser in a real implementation
        
        # Mock parsing
        best_model = None
        metrics = {}
        all_models = {}
        
        lines = response.split('\n')
        for line in lines:
            if 'best model' in line.lower() or 'best-performing model' in line.lower():
                best_model_match = re.search(r'is|:|\*\*\s*([A-Za-z0-9\-]+)', line)
                if best_model_match:
                    best_model = best_model_match.group(1).strip()
            
            # Look for metrics
            rmse_match = re.search(r'RMSE\s*(?:is|:|=)\s*(\d+\.\d+)', line)
            r2_match = re.search(r'R²\s*(?:is|:|=)\s*(\d+\.\d+)', line)
            
            if rmse_match:
                metrics['rmse'] = float(rmse_match.group(1))
            if r2_match:
                metrics['r2'] = float(r2_match.group(1))
            
            # Look for model details
            model_match = re.search(r'([A-Za-z0-9\-]+)\s*\(([^)]+)\):\s*RMSE\s*=\s*(\d+\.\d+),\s*R²\s*=\s*(\d+\.\d+)', line)
            if model_match:
                model_name = model_match.group(1).strip()
                method = model_match.group(2).strip()
                rmse = float(model_match.group(3))
                r2 = float(model_match.group(4))
                
                all_models[model_name] = {
                    'rmse': rmse,
                    'r2': r2,
                    'method': method,
                    'paper': 'Unknown'  # Would need more parsing to extract this
                }
        
        return {
            'dataset': dataset_name,
            'best_model': best_model or 'Unknown',
            'metrics': metrics,
            'all_models': all_models
        }
    
    def run(self, query: str) -> str:
        """
        Run the research agent with a query
        
        Args:
            query: The research query
            
        Returns:
            Agent's response
        """
        if not self.agent:
            # Fall back to direct query handling
            return self._process_query_directly(query)
        
        return self.agent.run(query)
    
    def _process_query_directly(self, query: str) -> str:
        """
        Process query directly when agent is not available
        
        Args:
            query: The research query
            
        Returns:
            Response to the query
        """
        # Extract keywords from query
        keywords = set(query.lower().split())
        
        if {"literature", "search", "papers", "find"}.intersection(keywords) and {"multimodal", "cross-modal", "fusion"}.intersection(keywords):
            papers = self.search_literature("multimodal learning molecular property prediction", 5)
            return f"Found {len(papers)} papers on multimodal learning for molecular property prediction:\n\n" + \
                   "\n\n".join([f"Title: {p.get('title', 'Unknown')}\nAuthors: {p.get('authors', 'Unknown')}\nYear: {p.get('year', 'Unknown')}\nJournal: {p.get('journal', 'Unknown')}\nRelevance: {p.get('relevance', 'Unknown')}" for p in papers])
        
        elif {"sota", "state-of-the-art", "benchmark", "performance"}.intersection(keywords) and {"delaney", "lipophilicity", "bace"}.intersection(keywords):
            for dataset in ["delaney", "lipophilicity", "bace"]:
                if dataset in query.lower():
                    return self._analyze_sota(dataset)
            
            return self._analyze_sota("delaney")  # Default
        
        elif {"literature", "review", "background", "related", "work"}.intersection(keywords):
            if {"attention", "transformer"}.intersection(keywords):
                return self.generate_literature_review("attention mechanisms")
            else:
                return self.generate_literature_review("multimodal learning for molecular property prediction")
        
        elif {"paper", "doi", "details"}.intersection(keywords):
            return self._get_paper_details("multimodal fused deep learning")
        
        else:
            return f"""
            I can help with research for the H-CAAN project. Here are some options:
            
            1. Literature search: Find relevant papers on multimodal learning for molecular property prediction
            2. SOTA analysis: Identify state-of-the-art benchmarks for specific datasets (Delaney, Lipophilicity, BACE)
            3. Paper details: Get detailed information about specific papers
            4. Literature review: Generate a comprehensive literature review section
            
            Please specify which type of research assistance you need.
            """
    
    def save_paper_to_database(self, paper: Dict) -> None:
        """
        Save a paper to the papers database
        
        Args:
            paper: Dictionary with paper details
        """
        if not all(k in paper for k in ['title', 'authors', 'year']):
            logger.error("Paper missing required fields (title, authors, year)")
            return
        
        # Check if paper already exists
        if not self.papers_db.empty:
            existing = self.papers_db[self.papers_db['title'] == paper['title']]
            if not existing.empty:
                logger.info(f"Paper already exists in database: {paper['title']}")
                return
        
        # Add paper to database
        self.papers_db = pd.concat([self.papers_db, pd.DataFrame([paper])], ignore_index=True)
        
        # Save database
        db_path = self.results_dir / "papers_database.csv"
        self.papers_db.to_csv(db_path, index=False)
        
        logger.info(f"Added paper to database: {paper['title']}")
    
    def update_sota_results(self, dataset: str, model: str, metrics: Dict, method: str = None, paper: str = None) -> None:
        """
        Update SOTA results with new model performance
        
        Args:
            dataset: Dataset name
            model: Model name
            metrics: Dictionary with performance metrics
            method: Method used by the model
            paper: Reference to the paper
        """
        if dataset not in self.sota_results['datasets']:
            self.sota_results['datasets'][dataset] = {
                'best_model': model,
                'models': {}
            }
            
            for metric, value in metrics.items():
                self.sota_results['datasets'][dataset][f'best_{metric}'] = value
        
        # Add model to dataset
        self.sota_results['datasets'][dataset]['models'][model] = metrics.copy()
        if method:
            self.sota_results['datasets'][dataset]['models'][model]['method'] = method
        if paper:
            self.sota_results['datasets'][dataset]['models'][model]['paper'] = paper
        
        # Update best model if this one is better
        for metric, value in metrics.items():
            if f'best_{metric}' in self.sota_results['datasets'][dataset]:
                current_best = self.sota_results['datasets'][dataset][f'best_{metric}']
                # Compare based on metric (lower is better for RMSE, MAE; higher is better for R2, accuracy)
                if metric.lower() in ['rmse', 'mae'] and value < current_best:
                    self.sota_results['datasets'][dataset]['best_model'] = model
                    self.sota_results['datasets'][dataset][f'best_{metric}'] = value
                elif metric.lower() not in ['rmse', 'mae'] and value > current_best:
                    self.sota_results['datasets'][dataset]['best_model'] = model
                    self.sota_results['datasets'][dataset][f'best_{metric}'] = value
            else:
                self.sota_results['datasets'][dataset][f'best_{metric}'] = value
        
        # Save updated SOTA results
        sota_path = self.results_dir / "sota_results.json"
        with open(sota_path, 'w') as f:
            json.dump(self.sota_results, f, indent=2)
        
        logger.info(f"Updated SOTA results for {dataset} with model {model}")


if __name__ == "__main__":
    # Example usage
    research_agent = ResearchAgent()
    
    # Example 1: Search literature
    print("\nExample 1: Literature Search")
    papers = research_agent.search_literature("multimodal learning molecular property prediction", 3)
    for paper in papers:
        print(f"\nTitle: {paper.get('title')}")
        print(f"Authors: {paper.get('authors')}")
        print(f"Year: {paper.get('year')}")
        print(f"Journal: {paper.get('journal')}")
        print(f"Relevance: {paper.get('relevance')}")
    
    # Example 2: Generate literature review
    print("\nExample 2: Generate Literature Review")
    review = research_agent.generate_literature_review("multimodal learning for molecular property prediction")
    print(review[:500] + "...")  # Print the first 500 characters
    
    # Example 3: Identify SOTA benchmarks
    print("\nExample 3: Identify SOTA Benchmarks")
    sota = research_agent.identify_sota_benchmarks("Delaney")
    print(f"Best model for {sota['dataset']}: {sota['best_model']}")
    for metric, value in sota['metrics'].items():
        print(f"{metric}: {value}")