#  Code For SARGE

This refers to an implementation of the paper titled *An Efficient Graph Autoencoder with Lightweight Desmoothing Decoder and Long-Range Modeling*.

## Require

- Python 3.9
- PyTorch 1.10.2
- dgl 0.7.2
- numpy  1.20.3

The  hyper-parameters for SARGE are as follows:
```python
python main.py --name_dataset cs --use_mlp --num_epochs 200 --lr2 5e-3 --wd2 1e-4 --lambd 1 --beta 1 --num_epochs 200 --hid_dim 1024 --out_dim 1024

python main.py --name_dataset computer --num_epochs 200 --lr2 1e-2 --wd2 1e-4 --lambd 1 --beta 1.0 --hid_dim 1024 --out_dim 1024

python main.py --name_dataset photo --num_epochs 100 --lr2 1e-2 --wd2 1e-4 --lambd 1 --beta 1 --hid_dim 1024 --out_dim 1024

python main.py --name_dataset physics --num_epochs 200 --hid_dim 1024 --out_dim 1024 --lr2 5e-3 --wd2 1e-4 --lambd 1 --beta 1

python main.py --name_dataset cora --num_epochs 50 --lr2 1e-2 --wd2 1e-4 --lambd 1 --beta 6 --t 0.12 --hid_dim 1024 --out_dim 1024

python main.py --name_dataset citeseer --num_layers 1 --num_epochs 5 --lr2 1e-2 --wd2 1e-2 --beta 5 --t 0.2 --hid_dim 1024 --out_dim 1024
```

The above hyperparameters do not undergo an extensive or complex grid search. With further tuning, you could achieve even better results.
