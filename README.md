# Picture That: CNN for Aphantasia Detection

## Dataset Description
This project utilizes the dataset compiled by Bainbridge et al. (2023) in their paper [Quantifying Aphantasia through drawing: Those without visual imagery show deficits in object but not spatial memory](https://osf.io/cahyd/).

## Models & Their Respective Training Files
|   Model Name | Training Script | Sweep YAML |
|--------------|-----------------|------------|
|   CNN        | train.py        |  sweep.yaml|
|   Aug-CNN        | train-aug.py        |  sweep-aug.yaml|
|   FT-ResNet        | train-ft.py        |  sweep-ft.yaml|
|   Aug-FT-ResNet         | train-ft-aug.py        |  sweep-ft-aug.yaml|


## Usage
### Create environment 
``` 
conda create -n <env_name> python=3.9 
conda activate <env_name>
pip install -r requirements.txt
```

### Run experiments (for all four sweep files)
```
wandb sweeps/sweep*.yaml
wandb agent <insert generated sweep name>
```

