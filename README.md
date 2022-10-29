# Probabilistic Deep Learning Model for Recovering Bone Microstructure from Low Resolution CT Images

## Usage

### Model Weights

| 85×85 -> 256×256 | Download trained checkpoint from: 
Place into model_checkpoints

### Data

Download sample train, validation, and testing data from: 
Place into data/datasets

### Training/Resume Training

```python
# Use sr.py to train the super resolution task from scratch
python sr.py -p train -c config/256_3_testing_config.json
```

### Test/Evaluation

```python
# Edit json to add pretrain model path and run the evaluation 
python test_pretrained.py
```
