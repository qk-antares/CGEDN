export PYTHONPATH=$PYTHONPATH:~/workplace/python/CGEDN/
cd ./experiments/compare/algorithms/beamsearch
nohup python evaluate.py > output.log 2>&1 &