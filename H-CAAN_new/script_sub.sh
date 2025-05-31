#ssh -L 8502:192.168.1.101:8502 lengcan@10.8.0.1 
#ssh -J lengcan@10.8.0.1 cleng@192.168.1.101 
python main.py --mode ui --port 8502 --host 0.0.0.0
export CUDA_VISIBLE_DEVICES=0,1
 python -m streamlit run streamlit/app.py --server.address 0.0.0.0 --server.port 8502 --server.headless true
 /vol1/cleng/miniconda3/envs/H-CAAN/bin/streamlit run streamlit_ui/Home.py --server.port 8502

 /vol1/cleng/miniconda3/envs/H-CAAN/bin/streamlit run streamlit_ui/Home.py --server.address 0.0.0.0 --server.port 8502 --server.enableCORS=false
