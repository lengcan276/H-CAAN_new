import os
import sys
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def launch_streamlit(host="localhost", port=8501):
    """
    Launch the Streamlit application using the current Python interpreter.
    """
    import subprocess
    import sys
    
    # 使用当前Python解释器运行Streamlit
    python_executable = sys.executable
    
    # 直接运行Python模块，而不是依赖PATH中的streamlit命令
    streamlit_args = [
        python_executable, "-m", "streamlit", "run", "streamlit/app.py",
        "--server.address", host,
        "--server.port", str(port),
        "--server.headless", "true",
        "--server.enableCORS", "false" 
    ]
    
    print(f"Launching Streamlit on {host}:{port} using Python at {python_executable}")
    subprocess.run(streamlit_args)
    
def prepare_environment():
    """
    Set up the environment for the H-CAAN framework.
    """
    # Create required directories
    directories = [
        'data',
        'models',
        'outputs',
        'results',
        'config'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Check for required packages
    try:
        import streamlit
        import torch
        import rdkit
        import pandas
        import matplotlib
        import sklearn
        import langchain
        import transformers
        import torch_geometric
        print("All required packages are installed.")
    except ImportError as e:
        print(f"Missing required package: {str(e)}")
        print("Please install all required packages.")
        sys.exit(1)
    
    # Create default config if not exists
    config_path = os.path.join('config', 'default_config.json')
    if not os.path.exists(config_path):
        import json
        default_config = {
            "model": {
                "smiles_encoder": {
                    "hidden_dim": 256,
                    "num_layers": 3,
                    "num_heads": 8
                },
                "ecfp_encoder": {
                    "hidden_dim": 256,
                    "num_layers": 2
                },
                "gcn_encoder": {
                    "hidden_dim": 256,
                    "num_layers": 3
                },
                "mfbert_encoder": {
                    "use_mfbert": True,
                    "hidden_dim": 512
                },
                "fusion": {
                    "levels": ["Low-level (Feature)", "Mid-level (Semantic)", "High-level (Decision)"],
                    "use_chemical_aware": True,
                    "use_adaptive_gating": True,
                    "use_multi_scale": True
                }
            },
            "training": {
                "batch_size": 64,
                "epochs": 100,
                "learning_rate": 0.001,
                "optimizer": "AdamW",
                "loss_function": "MSE"
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        print(f"Created default configuration at {config_path}")

def train_model(args):
    """
    Train the H-CAAN model with the given arguments.
    
    Args:
        args: Command line arguments
    """
    from agents.agent_manager import AgentManager
    
    # Initialize agent manager
    agent_manager = AgentManager(openai_api_key=args.openai_api_key, verbose=args.verbose)
    
    # Process dataset
    data_agent = agent_manager.get_agent("data")
    data_results = data_agent.process_dataset(args.dataset)
    
    # Configure model
    model_agent = agent_manager.get_agent("model")
    model_config = {
        "smiles_encoder": {
            "hidden_dim": args.smiles_hidden_dim,
            "num_layers": args.smiles_layers,
            "num_heads": args.smiles_heads
        },
        "ecfp_encoder": {
            "hidden_dim": args.ecfp_hidden_dim,
            "num_layers": args.ecfp_layers
        },
        "gcn_encoder": {
            "hidden_dim": args.gcn_hidden_dim,
            "num_layers": args.gcn_layers
        },
        "mfbert_encoder": {
            "use_mfbert": not args.no_mfbert,
            "hidden_dim": args.mfbert_hidden_dim
        },
        "fusion": {
            "levels": [level for level in ["Low-level (Feature)", "Mid-level (Semantic)", "High-level (Decision)"] 
                      if level.lower().split()[0] in args.fusion_levels],
            "use_chemical_aware": args.use_chemical_aware,
            "use_adaptive_gating": args.use_adaptive_gating,
            "use_multi_scale": args.use_multi_scale
        },
        "modal_importance": {
            "use_task_specific": args.use_task_specific,
            "use_complexity_aware": args.use_complexity_aware,
            "use_uncertainty": args.use_uncertainty
        },
        "general": {
            "dropout": args.dropout,
            "output_dim": args.output_dim
        }
    }
    
    model_summary = model_agent.configure_model(model_config)
    
    # Train model
    training_config = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "loss_function": args.loss_function,
        "early_stopping": {
            "use": not args.no_early_stopping,
            "patience": args.patience
        },
        "regularization": {
            "weight_decay": args.weight_decay
        },
        "lr_scheduler": args.scheduler,
        "augmentation": args.augmentation,
        "ablation_study": {
            "run": args.ablation,
            "components": args.ablation_components
        }
    }
    
    training_results = model_agent.train_model(training_config)
    
    # Evaluate results
    evaluation_agent = agent_manager.get_agent("evaluation")
    evaluation_results = evaluation_agent.evaluate_results(training_results, model_config)
    
    # Generate paper if requested
    if args.generate_paper:
        writing_agent = agent_manager.get_agent("writing")
        paper = writing_agent.generate_paper()
        
        print(f"Paper generated and saved to {os.path.join('outputs', 'paper.md')}")
    
    print("Training and evaluation completed successfully!")
    print(f"Results saved to {os.path.join('outputs', 'results.json')}")

def main():
    """
    Main entry point for the H-CAAN framework.
    """
    parser = argparse.ArgumentParser(description="H-CAAN: Hierarchical Cross-modal Adaptive Attention Network")
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="ui", choices=["ui", "prepare", "train", "eval"],
                        help="Operation mode (ui: launch UI, prepare: prepare environment, train: train model, eval: evaluate model)")
    parser.add_argument("--host", type=str, default="localhost",
                        help="Host address to bind to (for UI mode)")
    parser.add_argument("--port", type=int, default=8501,
                        help="Port to run Streamlit on (for UI mode)")
    parser.add_argument("--streamlit", action="store_true", help="Legacy option: Launch Streamlit interface")
    parser.add_argument("--prepare", action="store_true", help="Legacy option: Prepare environment")
    parser.add_argument("--train", action="store_true", help="Legacy option: Train model from command line")
    
    # Dataset
    parser.add_argument("--dataset", type=str, default="Delaney (ESOL)", 
                        help="Dataset name or path")
    
    # Model configuration
    parser.add_argument("--smiles-hidden-dim", type=int, default=256,
                      help="Hidden dimension for SMILES encoder")
    parser.add_argument("--smiles-layers", type=int, default=3,
                      help="Number of layers for SMILES encoder")
    parser.add_argument("--smiles-heads", type=int, default=8,
                      help="Number of attention heads for SMILES encoder")
    
    parser.add_argument("--ecfp-hidden-dim", type=int, default=256,
                      help="Hidden dimension for ECFP encoder")
    parser.add_argument("--ecfp-layers", type=int, default=2,
                      help="Number of layers for ECFP encoder")
    
    parser.add_argument("--gcn-hidden-dim", type=int, default=256,
                      help="Hidden dimension for GCN encoder")
    parser.add_argument("--gcn-layers", type=int, default=3,
                      help="Number of layers for GCN encoder")
    
    parser.add_argument("--no-mfbert", action="store_true",
                      help="Disable MFBERT encoder")
    parser.add_argument("--mfbert-hidden-dim", type=int, default=512,
                      help="Hidden dimension for MFBERT encoder")
    
    parser.add_argument("--fusion-levels", type=str, nargs="+", 
                       default=["low", "mid", "high"],
                       help="Fusion levels to use (low, mid, high)")
    
    parser.add_argument("--use-chemical-aware", action="store_true", default=True,
                      help="Use chemical-aware attention")
    parser.add_argument("--use-adaptive-gating", action="store_true", default=True,
                      help="Use adaptive gating")
    parser.add_argument("--use-multi-scale", action="store_true", default=True,
                      help="Use multi-scale attention")
    
    parser.add_argument("--use-task-specific", action="store_true", default=True,
                      help="Use task-specific weights")
    parser.add_argument("--use-complexity-aware", action="store_true", default=True,
                      help="Use complexity-aware modality selection")
    parser.add_argument("--use-uncertainty", action="store_true", default=True,
                      help="Use uncertainty estimation")
    
    parser.add_argument("--dropout", type=float, default=0.1,
                      help="Dropout rate")
    parser.add_argument("--output-dim", type=int, default=128,
                      help="Output dimension for intermediate representations")
    
    # Training configuration
    parser.add_argument("--batch-size", type=int, default=64,
                      help="Batch size")
    parser.add_argument("--epochs", type=int, default=100,
                      help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                      help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="AdamW",
                      choices=["Adam", "AdamW", "SGD", "RMSprop"],
                      help="Optimizer")
    parser.add_argument("--loss-function", type=str, default="MSE",
                      choices=["MSE", "MAE", "Huber", "Custom Multi-objective"],
                      help="Loss function")
    parser.add_argument("--no-early-stopping", action="store_true",
                      help="Disable early stopping")
    parser.add_argument("--patience", type=int, default=20,
                      help="Patience for early stopping")
    parser.add_argument("--weight-decay", type=float, default=0.0001,
                      help="Weight decay for regularization")
    parser.add_argument("--scheduler", type=str, default="ReduceLROnPlateau",
                      choices=["ReduceLROnPlateau", "CosineAnnealing", "OneCycleLR"],
                      help="Learning rate scheduler")
    parser.add_argument("--augmentation", action="store_true",
                      help="Use data augmentation")
    
    # Ablation study
    parser.add_argument("--ablation", action="store_true",
                      help="Run ablation study")
    parser.add_argument("--ablation-components", type=str, nargs="+",
                      default=["SMILES Encoder", "ECFP Encoder", "GCN Encoder", "MFBERT Encoder", "Chemical-Aware Attention"],
                      help="Components to ablate")
    
    # Paper generation
    parser.add_argument("--generate-paper", action="store_true",
                      help="Generate research paper")
    
    # Other options
    parser.add_argument("--openai-api-key", type=str, default=None,
                       help="OpenAI API key for LLM integration")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # First, check for legacy options
    if args.prepare:
        prepare_environment()
        return
    elif args.streamlit:
        launch_streamlit(host=args.host, port=args.port)
        return
    elif args.train:
        train_model(args)
        return
        
    # Execute based on mode
    if args.mode == "prepare":
        prepare_environment()
    elif args.mode == "ui":
        launch_streamlit(host=args.host, port=args.port)
    elif args.mode == "train":
        train_model(args)
    elif args.mode == "eval":
        # Placeholder for evaluation mode
        print("Evaluation mode not yet implemented")
    else:
        # Default to launching Streamlit
        launch_streamlit(host=args.host, port=args.port)

if __name__ == "__main__":
    main()