#ssh -L 8502:192.168.1.101:8502 lengcan@10.8.0.1 
#ssh -J lengcan@10.8.0.1 cleng@192.168.1.101 
python main.py --mode ui --port 8502 --host 0.0.0.0
 python -m streamlit run streamlit/app.py --server.address 0.0.0.0 --server.port 8502 --server.headless true
