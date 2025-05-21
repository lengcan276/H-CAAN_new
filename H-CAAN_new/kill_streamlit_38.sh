 for i in `lsof -i :8502 | grep "streamlit"|awk '{print $2}'`; do kill -9 $i; done

