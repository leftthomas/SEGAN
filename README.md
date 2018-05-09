# Pytorch Implementation of SEGAN (Speech Enhancement GAN)
Implementation of [SEGAN](https://arxiv.org/abs/1703.09452) by Pascual et al. in 2017, using pytorch.

## Requirements
* [Anaconda](https://www.anaconda.com/download/)
* PyTorch
```
conda install pytorch torchvision -c pytorch
```
* PyTorchNet
```
pip install git+https://github.com/pytorch/tnt.git@master
```
* capsule-layer
```
pip install git+https://github.com/leftthomas/CapsuleLayer.git@master
```
* tqdm
```
conda install tqdm
```
* sox
```
brew install sox         ---- MacOS
sudo apt-get install sox ---- Ubuntu
```
* librosa
```
pip install librosa
```

## Data Preprocessing

Noisy speech dataset downloaded from [here](https://datashare.is.ed.ac.uk/handle/10283/2791)
Use `data_preprocess.py` file to preprocess downloaded data. 
Adjust the file paths at the beginning of the file to properly locate the data files, output folder, etc.

Data preprocessing consists of three main stages:
1. Downsampling - downsample original audio files (48k) to sampling rate of 16000.
2. Serialization - Splitting the audio files into 2^14-sample (about 1 second) snippets.
3. Verification - whether it contains proper number of samples.

## Training

`python model.py`

Again, fix and adjust datapaths in `model.py` according to your needs.
Especially, provide accurate path to where serialized data are stored.
