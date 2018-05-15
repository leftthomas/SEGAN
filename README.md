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
* tqdm
```
conda install tqdm
```

## Datasets
The clear and noisy speech datasets are downloaded from [DataShare](https://datashare.is.ed.ac.uk/handle/10283/2791).
Download the `56kHZ` train datasets and test datasets then extract them into `data` directory.

If you want using other datasets, you should change the path of data defined on `data_preprocess.py`.

## Usage
### Data Pre-process
```
python data_preprocess.py
```
The pre-processed datas are on `data/serialized_train_data` and `data/serialized_test_data`.

### Train Model and Test
```
python main.py ----batch_size 128 --num_epochs 300
optional arguments:
--batch_size             train batch size [default value is 50]
--num_epochs             train epochs number [default value is 86]
```
The test results are on `results`.

### Test Audio
```
python test_audio.py ----file_name p232_160.wav --epoch_name generator-80.pkl
optional arguments:
--file_name              audio file name
--epoch_name             generator epoch name
```
The generated enhanced audio is on the same directory of input audio.

## Results

<table border="0" cellspacing="10" cellpadding="10">
	<tbody>
		<tr>
			<td>Clear</td>
			<td>Noisy</td>
			<td>Enhanced</td>
		</tr>
		<tr valign="middle" align="left" style="border-bottom: 3px solid #f00;">
			<td>
				<audio style="width:200px" controls="controls">
					<source src="results/clear/p232_036.wav" type="audio/wav">
				</audio>
			</td>
			<td>
				<audio style="width:200px" controls="controls">
					<source src="results/noisy/p232_036.wav" type="audio/wav">
				</audio>
			</td>
			<td>
				<audio style="width:200px" controls="controls">
					<source src="results/enhanced/enhanced_p232_036.wav" type="audio/wav">
				</audio>
			</td>
		</tr>
				<tr valign="middle" align="left" style="border-bottom: 3px solid #f00;">
			<td>
				<audio style="width:200px" controls="controls">
					<source src="results/clear/p232_399.wav" type="audio/wav">
				</audio>
			</td>
			<td>
				<audio style="width:200px" controls="controls">
					<source src="results/noisy/p232_399.wav" type="audio/wav">
				</audio>
			</td>
			<td>
				<audio style="width:200px" controls="controls">
					<source src="results/enhanced/enhanced_p232_399.wav" type="audio/wav">
				</audio>
			</td>
		</tr>
		<tr valign="middle" align="left" style="border-bottom: 3px solid #f00;">
			<td>
				<audio style="width:200px" controls="controls">
					<source src="results/clear/p257_298.wav" type="audio/wav">
				</audio>
			</td>
			<td>
				<audio style="width:200px" controls="controls">
					<source src="results/noisy/p257_298.wav" type="audio/wav">
				</audio>
			</td>
			<td>
				<audio style="width:200px" controls="controls">
					<source src="results/enhanced/enhanced_p257_298.wav" type="audio/wav">
				</audio>
			</td>
		</tr>
				<tr valign="middle" align="left" style="border-bottom: 3px solid #f00;">
			<td>
				<audio style="width:200px" controls="controls">
					<source src="results/clear/p257_428.wav" type="audio/wav">
				</audio>
			</td>
			<td>
				<audio style="width:200px" controls="controls">
					<source src="results/noisy/p257_428.wav" type="audio/wav">
				</audio>
			</td>
			<td>
				<audio style="width:200px" controls="controls">
					<source src="results/enhanced/enhanced_p257_428.wav" type="audio/wav">
				</audio>
			</td>
		</tr>
	</tbody>
</table>

<iframe height= 500 width= 100% src="http://www.bilibili.com/video/av7019422/" frameborder=0 allowfullscreen></iframe>