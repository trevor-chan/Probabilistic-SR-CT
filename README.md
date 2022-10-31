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

# Edit the config json file to resume training from a previous checkpoint
# Use sr.py -c config/<config_file>
python sr.py
```

### Test/Evaluation

```python
# run inference and calculate PSNR for a subset of the testing dataset
# results in ./inference_results
python test_pretrained.py
```

### Hardware requirements

Current setup is optimized for a Nvidia V100 32GB. For training, reduce the batch size in the json file in ./configs to reduce memory usage.

```
# Use
python sr.py -c config/small_model.json
```

Contact: tjchan@seas.upenn.edu and chamith@pennmedicine.upenn.edu
