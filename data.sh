export PYTHONPATH=$PYTHONPATH:~/workplace/python/CGEDN/
cd ./data/IMDB_small
nohup python process_train.py > output.log 2>&1 &