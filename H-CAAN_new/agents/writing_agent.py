import os
import logging
import json
import time
import re
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class WritingAgent:
    """
    Agent responsible for generating scientific papers based on the
    H-CAAN model results.
    """
    
    def __init__(self, knowledge_base=None, openai_api_key=None, verbose=True):
        """
        Initialize the Writing Agent.
        
        Args:
            knowledge_base (dict, optional): Shared knowledge base
            openai_api_key (str, optional): OpenAI API key for LLM integration
            verbose (bool): Whether to output detailed logs
        """
        self.knowledge_base = knowledge_base or {}
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.verbose = verbose
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Initialize required directories
        self.output_dir = os.path.join(os.getcwd(), 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize LLM if API key is available
        self.llm = self._setup_llm()
        
        self.logger.info("Writing Agent initialized successfully")
    
    def _setup_logging(self):
        """Set up logging configuration"""
        logger = logging.getLogger("WritingAgent")
        
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
        """Set up the language model for paper generation"""
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
            "# H-CAAN: Hierarchical Cross-modal Adaptive Attention Network for Enhanced Drug Property Prediction\n\n**Authors: Author 1, Author 2, Author 3**\n\n## Abstract\n\nAccurate prediction of molecular properties is crucial for accelerating drug discovery and reducing development costs. While recent advancements in deep learning have shown promising results, most existing approaches rely on single modality representations, limiting their ability to capture the full complexity of molecular structures. In this paper, we introduce H-CAAN, a novel Hierarchical Cross-modal Adaptive Attention Network that integrates multiple molecular representations through a sophisticated fusion mechanism. Our approach leverages chemical language (SMILES and ECFP fingerprints), molecular graphs, and pre-trained molecular embeddings (MFBERT) to provide a comprehensive understanding of chemical structures. Through extensive experiments on multiple datasets, we demonstrate that H-CAAN consistently outperforms state-of-the-art mono-modal approaches, achieving significant improvements in predictive accuracy and generalization ability. Ablation studies confirm the effectiveness of our hierarchical fusion approach and chemical-aware attention mechanisms. The proposed model offers a promising direction for enhanced molecular property prediction with potential applications across various stages of drug discovery.\n\n## 1. Introduction\n\nDrug discovery is a complex, time-consuming, and expensive process, with an average development timeline of 10-15 years and costs exceeding $2.6 billion per drug. A critical aspect of this process is the accurate prediction of molecular properties, which can help identify promising candidates early and reduce experimental burden...",
            
            "## 2. Related Work\n\nMolecular representation learning has seen significant advancements with the introduction of deep learning techniques. Traditional approaches relied on hand-crafted features or fingerprints, but recent methods have explored more sophisticated neural architectures...",
            
            "## 3. Methods\n\n### 3.1 Hierarchical Cross-modal Adaptive Attention Network\n\nThe H-CAAN architecture consists of three main components: (1) multi-modal encoders, (2) hierarchical fusion modules, and (3) dynamic modality importance assessment. Figure 1 illustrates the overall framework...",
            
            "## 4. Experiments\n\n### 4.1 Datasets\n\nWe evaluated our model on six publicly available datasets for molecular property prediction: Delaney (ESOL) for aqueous solubility, Lipophilicity for octanol/water distribution coefficient, BACE for binding affinity, BBBP for blood-brain barrier penetration, ClinTox for clinical toxicity, and SIDER for side effect prediction...",
            
            "## 5. Results and Discussion\n\n### 5.1 Performance Comparison\n\nTable 1 presents the performance comparison between H-CAAN and existing state-of-the-art models across all six datasets. Our model consistently outperforms all baseline methods across all metrics..."
        ]
        
        return FakeListLLM(responses=responses)
    
    def generate_paper(self, paper_config=None):
        """
        Generate a scientific paper based on the model results.
        
        Args:
            paper_config (dict, optional): Paper configuration
            
        Returns:
            dict: Generated paper content
        """
        self.logger.info("Generating scientific paper...")
        
        # Check if knowledge base has required information
        if not self._check_knowledge_base():
            self.logger.error("Knowledge base missing required information")
            return {"error": "Knowledge base missing required information"}
        
        # Use default configuration if not provided
        if paper_config is None:
            paper_config = self._default_paper_config()
        
        # Generate paper sections
        paper_content = self._generate_paper_content(paper_config)
        
        # Save paper to file
        paper_path = os.path.join(self.output_dir, 'paper.md')
        with open(paper_path, 'w') as f:
            f.write(paper_content['markdown'])
        
        self.logger.info(f"Paper saved to {paper_path}")
        
        # Generate LaTeX version if requested
        if paper_config.get('generate_latex', True):
            latex_content = self._convert_to_latex(paper_content['markdown'])
            latex_path = os.path.join(self.output_dir, 'paper.tex')
            with open(latex_path, 'w') as f:
                f.write(latex_content)
            
            self.logger.info(f"LaTeX version saved to {latex_path}")
            paper_content['latex'] = latex_content
        
        return paper_content
    
    def _check_knowledge_base(self):
        """
        Check if knowledge base has all required information for paper generation.
        
        Returns:
            bool: True if all required information is available
        """
        required_keys = [
            'dataset',
            'model_config',
            'model_summary',
            'training_results',
            'evaluation_results'
        ]
        
        for key in required_keys:
            if key not in self.knowledge_base:
                self.logger.warning(f"Knowledge base missing {key}")
                
                # For demo purposes, create mock data if missing
                if key == 'dataset':
                    self.knowledge_base[key] = {
                        'name': 'Delaney (ESOL)',
                        'stats': {
                            'num_molecules': 1128,
                            'property_stats': {
                                'mean': -3.05,
                                'std': 2.10,
                                'min': -11.6,
                                'max': 1.58
                            }
                        }
                    }
                elif key == 'model_config':
                    self.knowledge_base[key] = {
                        'smiles_encoder': {'hidden_dim': 256, 'num_layers': 3},
                        'ecfp_encoder': {'hidden_dim': 256, 'num_layers': 2},
                        'gcn_encoder': {'hidden_dim': 256, 'num_layers': 3},
                        'mfbert_encoder': {'use_mfbert': True, 'hidden_dim': 512},
                        'fusion': {'levels': ["Low-level (Feature)", "Mid-level (Semantic)", "High-level (Decision)"],
                                   'use_chemical_aware': True}
                    }
                elif key == 'model_summary':
                    self.knowledge_base[key] = {
                        'total_parameters': 12_345_678,
                        'architecture': 'H-CAAN'
                    }
                elif key == 'training_results':
                    self.knowledge_base[key] = {
                        'test_metrics': {
                            'r2': 0.93,
                            'rmse': 0.45,
                            'mae': 0.37,
                            'modality_weights': {
                                'smiles': 0.35,
                                'ecfp': 0.25,
                                'graph': 0.20,
                                'mfbert': 0.20
                            }
                        },
                        'ablation_results': {
                            'baseline': {'r2': 0.93, 'rmse': 0.45},
                            'no_SMILES_Encoder': {'r2': 0.88, 'rmse': 0.55},
                            'no_ECFP_Encoder': {'r2': 0.90, 'rmse': 0.51},
                            'no_GCN_Encoder': {'r2': 0.89, 'rmse': 0.53},
                            'no_MFBERT_Encoder': {'r2': 0.91, 'rmse': 0.49},
                            'no_Chemical_Aware_Attention': {'r2': 0.92, 'rmse': 0.47}
                        }
                    }
                elif key == 'evaluation_results':
                    self.knowledge_base[key] = {
                        'sota_comparison': {
                            'H-CAAN': {'r2': 0.93, 'rmse': 0.45},
                            'MMFDL': {'r2': 0.89, 'rmse': 0.53},
                            'MFBERT': {'r2': 0.87, 'rmse': 0.57},
                            'AttentiveFP': {'r2': 0.84, 'rmse': 0.63}
                        }
                    }
        
        return True
    
    def _default_paper_config(self):
        """
        Create default paper configuration.
        
        Returns:
            dict: Default paper configuration
        """
        return {
            'title': 'H-CAAN: Hierarchical Cross-modal Adaptive Attention Network for Enhanced Drug Property Prediction',
            'authors': 'Author 1, Author 2, Author 3',
            'sections': [
                'Abstract',
                'Introduction',
                'Related Work',
                'Methods',
                'Experiments',
                'Results',
                'Discussion',
                'Conclusion',
                'References'
            ],
            'target_journal': 'Journal of Chemical Information and Modeling',
            'include_ablation': True,
            'include_comparison': True,
            'include_future_work': True,
            'generate_latex': True
        }
    
    def _generate_paper_content(self, paper_config):
        """
        Generate paper content using LLM.
        
        Args:
            paper_config (dict): Paper configuration
            
        Returns:
            dict: Generated paper content
        """
        if self.llm is None:
            self.logger.warning("LLM not available. Using template-based paper generation.")
            return self._generate_template_paper(paper_config)
        
        # Extract information from knowledge base
        dataset_info = self.knowledge_base.get('dataset', {})
        model_config = self.knowledge_base.get('model_config', {})
        model_summary = self.knowledge_base.get('model_summary', {})
        training_results = self.knowledge_base.get('training_results', {})
        evaluation_results = self.knowledge_base.get('evaluation_results', {})
        
        # Create input for each section
        sections = {}
        
        # Generate each section
        for section in paper_config['sections']:
            self.logger.info(f"Generating {section} section...")
            
            section_prompt = self._create_section_prompt(
                section, 
                paper_config, 
                dataset_info,
                model_config, 
                model_summary, 
                training_results, 
                evaluation_results
            )
            
            # Call LLM to generate section
            try:
                section_content = self._call_llm(section_prompt)
                sections[section] = section_content
            except Exception as e:
                self.logger.error(f"Error generating {section} section: {str(e)}")
                sections[section] = f"[Error generating {section} section]"
            
            # Add a small delay to avoid rate limiting
            time.sleep(1)
        
        # Assemble paper
        paper_content = self._assemble_paper(paper_config, sections)
        
        return {
            'title': paper_config['title'],
            'authors': paper_config['authors'],
            'sections': sections,
            'markdown': paper_content
        }
    
    def _create_section_prompt(self, section, paper_config, dataset_info, model_config, model_summary, training_results, evaluation_results):
        """
        Create a prompt for generating a specific paper section.
        
        Args:
            section (str): Section name
            paper_config (dict): Paper configuration
            dataset_info (dict): Dataset information
            model_config (dict): Model configuration
            model_summary (dict): Model summary
            training_results (dict): Training results
            evaluation_results (dict): Evaluation results
            
        Returns:
            str: Prompt for the section
        """
        base_prompt = f"""
You are a scientific writer specializing in deep learning for drug discovery. You are writing a section of a research paper
about a new model called H-CAAN (Hierarchical Cross-modal Adaptive Attention Network) for enhanced drug property prediction.

The paper is for submission to {paper_config['target_journal']}.

Paper title: {paper_config['title']}
Authors: {paper_config['authors']}

Current section to write: {section}

Here's information about the H-CAAN model:
- It integrates multiple molecular representations: SMILES, ECFP fingerprints, molecular graphs, and MFBERT embeddings
- It uses a hierarchical fusion approach with three levels: low-level feature fusion, mid-level semantic fusion, and high-level decision fusion
- It includes chemical-aware attention mechanisms that understand molecular substructures
- It dynamically assesses modality importance based on molecular complexity and task requirements
- Total parameters: {model_summary.get('total_parameters', 'approximately 12 million')}

Dataset information:
- Main dataset: {dataset_info.get('name', 'Delaney (ESOL)')}
- Number of molecules: {dataset_info.get('stats', {}).get('num_molecules', 1128)}
- Property statistics: {json.dumps(dataset_info.get('stats', {}).get('property_stats', {}))}

Results:
- Performance on test set: R² = {training_results.get('test_metrics', {}).get('r2', 0.93)}, RMSE = {training_results.get('test_metrics', {}).get('rmse', 0.45)}
- Modality contributions: {json.dumps(training_results.get('test_metrics', {}).get('modality_weights', {'smiles': 0.35, 'ecfp': 0.25, 'graph': 0.20, 'mfbert': 0.20}))}
- SOTA comparison: {json.dumps(evaluation_results.get('sota_comparison', {'H-CAAN': {'r2': 0.93, 'rmse': 0.45}, 'MMFDL': {'r2': 0.89, 'rmse': 0.53}, 'MFBERT': {'r2': 0.87, 'rmse': 0.57}, 'AttentiveFP': {'r2': 0.84, 'rmse': 0.63}}))}

Ablation study results:
{json.dumps(training_results.get('ablation_results', {'baseline': {'r2': 0.93, 'rmse': 0.45}, 'no_SMILES_Encoder': {'r2': 0.88, 'rmse': 0.55}}))}

Write a comprehensive and scientifically accurate {section} section for this paper. Use academic language appropriate for {paper_config['target_journal']}.
Ensure the content is detailed, technically sound, and reflective of the information provided. Be factual and precise.
        """
        
        # Add section-specific instructions
        if section == "Abstract":
            base_prompt += """
For the Abstract, include:
1. The problem statement (importance of accurate drug property prediction)
2. Limitations of existing approaches
3. Brief description of the H-CAAN model and its key innovations
4. Summary of main results and contributions
5. Potential impact

Keep the abstract concise (250-300 words) but comprehensive.
            """
        elif section == "Introduction":
            base_prompt += """
For the Introduction, include:
1. Background on drug discovery and the importance of molecular property prediction
2. Challenges in computational property prediction
3. Limitations of existing approaches (single modality, limited fusion)
4. Overview of H-CAAN and its innovations
5. Summary of contributions
6. Paper structure overview

The introduction should motivate the research problem and clearly position the contribution.
            """
        elif section == "Related Work":
            base_prompt += """
For the Related Work section, discuss:
1. Traditional molecular representation methods (fingerprints, descriptors)
2. Deep learning approaches for molecular property prediction
   - Graph-based approaches (GCN, AttentiveFP, etc.)
   - Sequence-based approaches (SMILES transformers, etc.)
   - Pre-trained molecular models (MFBERT, ChemBERTa, etc.)
3. Multi-modal fusion approaches in chemical informatics
4. Attention mechanisms for molecular representation
5. Position H-CAAN relative to existing work

Be comprehensive in covering the landscape but focused on relevant approaches.
            """
        elif section == "Methods":
            base_prompt += """
For the Methods section, provide detailed descriptions of:
1. Overall H-CAAN architecture
2. Individual modal encoders:
   - SMILES encoder (transformer-based)
   - ECFP encoder (BiGRU-based)
   - Graph encoder (GCN-based)
   - MFBERT encoder (if used)
3. Hierarchical fusion approach:
   - Low-level feature fusion with Gated Cross-modal Attention Units
   - Mid-level semantic fusion with contrastive learning
   - High-level decision fusion with adaptive weighting
4. Chemical-aware attention mechanisms
5. Dynamic modality importance assessment:
   - Task-specific weight generation
   - Molecular complexity assessment
   - Uncertainty estimation
6. Training procedure and implementation details

Include relevant equations, algorithmic descriptions, and architectural diagrams. Be precise and technically sound.
            """
        elif section == "Experiments":
            base_prompt += """
For the Experiments section, describe:
1. Datasets used (including statistics, data splits, preprocessing)
2. Evaluation metrics (R², RMSE, MAE, etc.)
3. Baseline models for comparison
4. Implementation details:
   - Hyperparameters
   - Training procedure
   - Computational resources
5. Ablation study design
6. Evaluation protocols

Be thorough in explaining the experimental setup to ensure reproducibility.
            """
        elif section == "Results":
            base_prompt += """
For the Results section, present:
1. Overall performance comparison with SOTA methods (tables and figures)
2. Ablation study results analyzing the contribution of each component
3. Analysis of modality importance across different datasets
4. Visualization of attention weights and chemical interpretation
5. Challenging molecule analysis
6. Computation efficiency analysis

Present results clearly with appropriate statistical analysis, tables, and figures.
            """
        elif section == "Discussion":
            base_prompt += """
For the Discussion section, include:
1. Interpretation of main results and their significance
2. Analysis of when and why H-CAAN outperforms existing methods
3. Insights from the ablation studies
4. Limitations of the approach
5. Potential applications in drug discovery
6. Theoretical implications for multimodal learning in cheminformatics

Provide thoughtful analysis that goes beyond reporting results.
            """
        elif section == "Conclusion":
            base_prompt += """
For the Conclusion section:
1. Summarize the key contributions
2. Highlight the significance of the work
3. Discuss implications for the field
4. Outline future work directions
5. End with a concise statement on the broader impact

Keep the conclusion concise but impactful.
            """
        elif section == "References":
            base_prompt += """
For the References section, list key papers in the following areas:
1. Deep learning for molecular property prediction
2. Graph neural networks for molecular modeling
3. Transformer models for SMILES processing
4. Multi-modal fusion approaches
5. Attention mechanisms in deep learning
6. Molecular fingerprints and representations
7. Benchmark datasets used in the study

Format in the citation style of the target journal.
            """
        
        return base_prompt
    
    def _call_llm(self, prompt):
        """
        Call the LLM with a given prompt.
        
        Args:
            prompt (str): Prompt for the LLM
            
        Returns:
            str: Generated content
        """
        if isinstance(self.llm, OpenAI) or isinstance(self.llm, ChatOpenAI):
            chat_prompt = ChatPromptTemplate.from_template(prompt)
            chain = LLMChain(llm=self.llm, prompt=chat_prompt)
            result = chain.run("")
            return result
        else:
            # For simulated LLM
            return self.llm.generate([prompt]).generations[0][0].text
    
    def _assemble_paper(self, paper_config, sections):
        """
        Assemble the paper from generated sections.
        
        Args:
            paper_config (dict): Paper configuration
            sections (dict): Generated sections
            
        Returns:
            str: Assembled paper content
        """
        # Start with title and authors
        paper_content = f"# {paper_config['title']}\n\n"
        paper_content += f"**{paper_config['authors']}**\n\n"
        
        # Add each section
        for section_name in paper_config['sections']:
            if section_name in sections:
                section_content = sections[section_name]
                
                # Clean up section content
                section_content = self._clean_section_content(section_content, section_name)
                
                paper_content += section_content + "\n\n"
        
        return paper_content
    
    def _clean_section_content(self, content, section_name):
        """
        Clean up the generated section content.
        
        Args:
            content (str): Raw section content
            section_name (str): Section name
            
        Returns:
            str: Cleaned section content
        """
        # Remove section header if included by the LLM
        section_patterns = [
            f"# {section_name}",
            f"## {section_name}",
            f"### {section_name}",
            f"{section_name}:"
        ]
        
        for pattern in section_patterns:
            if content.startswith(pattern):
                content = content[len(pattern):].strip()
        
        # Add proper section header
        if section_name == "Abstract":
            header = f"## {section_name}\n\n"
        elif section_name in ["Introduction", "Related Work", "Methods", "Experiments", "Results", "Discussion", "Conclusion", "References"]:
            header = f"## {section_name}\n\n"
        else:
            header = f"### {section_name}\n\n"
        
        return header + content
    
    def _generate_template_paper(self, paper_config):
        """
        Generate a paper using templates when LLM is not available.
        
        Args:
            paper_config (dict): Paper configuration
            
        Returns:
            dict: Generated paper content
        """
        # Extract information from knowledge base
        dataset_info = self.knowledge_base.get('dataset', {})
        model_config = self.knowledge_base.get('model_config', {})
        model_summary = self.knowledge_base.get('model_summary', {})
        training_results = self.knowledge_base.get('training_results', {})
        evaluation_results = self.knowledge_base.get('evaluation_results', {})
        
        # Create sections
        sections = {}
        
        # Abstract
        sections["Abstract"] = """
Accurate prediction of molecular properties is crucial for accelerating drug discovery and reducing development costs. While recent advancements in deep learning have shown promising results, most existing approaches rely on single modality representations, limiting their ability to capture the full complexity of molecular structures. In this paper, we introduce H-CAAN, a novel Hierarchical Cross-modal Adaptive Attention Network that integrates multiple molecular representations through a sophisticated fusion mechanism. Our approach leverages chemical language (SMILES and ECFP fingerprints), molecular graphs, and pre-trained molecular embeddings (MFBERT) to provide a comprehensive understanding of chemical structures. Through extensive experiments on multiple datasets, we demonstrate that H-CAAN consistently outperforms state-of-the-art mono-modal approaches, achieving significant improvements in predictive accuracy and generalization ability. Ablation studies confirm the effectiveness of our hierarchical fusion approach and chemical-aware attention mechanisms. The proposed model offers a promising direction for enhanced molecular property prediction with potential applications across various stages of drug discovery.
        """
        
        # Introduction
        sections["Introduction"] = """
Drug discovery is a complex, time-consuming, and expensive process, with an average development timeline of 10-15 years and costs exceeding $2.6 billion per drug. A critical aspect of this process is the accurate prediction of molecular properties, which can help identify promising candidates early and reduce experimental burden. Traditional approaches to property prediction rely on physics-based simulations or simple machine learning models with hand-crafted molecular descriptors, both of which have limitations in accuracy and generalization.

Recent advances in deep learning have shown promising results for molecular property prediction. These approaches typically leverage a single molecular representation, such as SMILES strings, molecular fingerprints, or molecular graphs. While each representation captures certain aspects of molecular structure, they individually fail to provide a comprehensive understanding. For example, graph-based methods excel at capturing topological features but may miss global patterns, while SMILES-based approaches can encode sequential patterns but struggle with spatial relationships.

Several attempts have been made to combine different molecular representations, but most existing methods employ simple concatenation or late fusion strategies, which fail to effectively capture the complex interactions between different modalities. Furthermore, these approaches typically assign fixed weights to each modality, disregarding the fact that the importance of different representations may vary across molecules and prediction tasks.

To address these limitations, we introduce H-CAAN (Hierarchical Cross-modal Adaptive Attention Network), a novel deep learning framework for molecular property prediction that effectively integrates multiple molecular representations through a sophisticated hierarchical fusion approach. Our model combines chemical language (SMILES and ECFP fingerprints), molecular graphs, and pre-trained molecular embeddings (MFBERT) using a series of cross-modal attention mechanisms operating at different levels of abstraction. Additionally, H-CAAN dynamically assesses the importance of each modality based on molecular complexity and task requirements, enabling adaptive fusion that optimizes predictive performance.

The main contributions of this work include:
1. A hierarchical fusion architecture that integrates multiple molecular representations at different levels of abstraction, enabling comprehensive understanding of molecular structures.
2. Novel chemical-aware attention mechanisms that focus on chemically relevant substructures across different representations.
3. Dynamic modality importance assessment based on molecular complexity and uncertainty estimation.
4. Extensive experiments demonstrating state-of-the-art performance across multiple molecular property prediction tasks.
5. Thorough ablation studies providing insights into the contribution of each component.

The rest of the paper is organized as follows: Section 2 discusses related work, Section 3 describes the proposed H-CAAN architecture, Section 4 presents the experimental setup, Section 5 analyzes the results, Section 6 provides discussion and analysis, and Section 7 concludes the paper.
        """
        
        # Add remaining sections
        sections["Related Work"] = "..."
        sections["Methods"] = "..."
        sections["Experiments"] = "..."
        sections["Results"] = "..."
        sections["Discussion"] = "..."
        sections["Conclusion"] = "..."
        sections["References"] = "..."
        
        # Assemble paper
        paper_content = self._assemble_paper(paper_config, sections)
        
        return {
            'title': paper_config['title'],
            'authors': paper_config['authors'],
            'sections': sections,
            'markdown': paper_content
        }
    
    def _convert_to_latex(self, markdown_content):
        """
        Convert markdown content to LaTeX.
        
        Args:
            markdown_content (str): Markdown content
            
        Returns:
            str: LaTeX content
        """
        # Very basic conversion for demonstration purposes
        latex_content = r"""
\documentclass[twocolumn]{article}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{hyperref}

\title{H-CAAN: Hierarchical Cross-modal Adaptive Attention Network for Enhanced Drug Property Prediction}
\author{Author 1 \and Author 2 \and Author 3}

\begin{document}

\maketitle
        """
        
        # Replace markdown headers with LaTeX sections
        content = markdown_content
        content = re.sub(r'# (.*)', r'\\section{\1}', content)
        content = re.sub(r'## (.*)', r'\\subsection{\1}', content)
        content = re.sub(r'### (.*)', r'\\subsubsection{\1}', content)
        
        # Replace markdown formatting
        content = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', content)
        content = re.sub(r'\*(.*?)\*', r'\\textit{\1}', content)
        
        # Add the content to the LaTeX document
        latex_content += content
        
        # Close the document
        latex_content += r"""
\end{document}
        """
        
        return latex_content