export PYTHONPATH=$PYTHONPATH:~/workplace/python/CGEDN/
cd ./experiments/ablation
nohup python main.py > output.log 2>&1 &