# NMNIST experiments in Sinabs

## Install requirements
```
pip install -r requirements.txt
```

## Convert a trained CNN to an SNN
Run the `test-converted-snn.ipynb` notebook

## Train CNN from scratch
```
python train.py --num_workers=4 --model=cnn --batch_size=64
```