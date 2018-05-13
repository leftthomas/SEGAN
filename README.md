# SEGAN
A PyTorch implementation of SEGAN based on paper [SEGAN: Speech Enhancement Generative Adversarial Network](https://arxiv.org/abs/1703.09452).

## Requirements
* [Anaconda](https://www.anaconda.com/download/)
* PyTorch
```
conda install pytorch torchvision -c pytorch
```
* librosa
```
pip install librosa
```
* sox
```
brew install sox         ---- MacOS
sudo apt-get install sox ---- Ubuntu
```

## Datasets
The clear and noisy speech datasets are downloaded from [DataShare](https://datashare.is.ed.ac.uk/handle/10283/2791).
Download the `56kHZ` datasets and then extract them into `data` directory.

## Usage
### Data Preprocess
`python data_preprocess.py`

### Train
`python main.py`
