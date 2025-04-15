nohup python -u train_gnn.py --name grn_gin --epoch 200 --batch 256 --conv gin  --hidden 512 --gpu 0 --seed 42 --layer 8  > gin.log 2>&1  &

nohup python -u  train_gnn.py --name grn_diff --epoch 200 --batch 128 --conv diff --hidden 512 --gpu 1 --seed 42 >diff.log 2>&1 &

nohup python -u  train_gnn.py --name grn_gcn --epoch 200 --batch 256 --conv gcn --gpu 2 --hidden 512 --seed 42 --layer 9  >gcn.log 2>&1 &

# nohup python -u  train_gnn.py --name grn_gat --epoch 1000 --batch 32 --conv gat --gpu 2 >gat.log 2>&1 &

# nohup python -u  train_gnn.py --name grn_gcn --epoch 1000 --batch 32 --conv gcn --gpu 1 >gcn.log 2>&1 &
