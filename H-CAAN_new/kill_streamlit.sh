 for i in `lsof -i :8502 |grep  IPv4 |grep "python"|awk '{print $2}'`; do kill -9 $i; done

