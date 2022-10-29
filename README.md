# Probabilistic Deep Learning Model for Recovering Bone Microstructure from Low Resolution CT Images

## Usage

### Model Weights

| 85×85 -> 256×256 | Download trained checkpoint from: https://drive.google.com/drive/folders/1lwyhKn06zOzVBTpAyLLPImUx2TATADs8?usp=sharing
Place into ./model_checkpoints

### Data

Download sample train, validation, and testing data from: https://drive.google.com/drive/folders/1lwyhKn06zOzVBTpAyLLPImUx2TATADs8?usp=sharing
Place into ./data/datasets

### Training/Resume Training

```python
# Use sr.py to train the super resolution task from scratch
python sr.py
```

### Test/Evaluation

```python
# Edit json to add pretrain model path and run the evaluation 
python test_pretrained.py
```
