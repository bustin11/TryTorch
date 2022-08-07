
# Introduction

The purpose of this assignment is predict the phonenome labels from spectrograms using a deep neural network

# Prequisites

## Libraries
- python3 with pytorch

## Computer
- a computer with a GPU, or AWS EC2 instance (trained on g4dn.x2large for a day)

## Dataset

For manual download:
- Go to [Link](https://www.kaggle.com/c/idl-fall2021-hw1p2/data). Select Download All  

For Kaggle API
- Make sure to create a token in your kaggle account. Create a directory `.kaggle` and place the api token in a file `.kaggle/kaggle.json`

In `./`:
```
chmod 600 /content/.kaggle/kaggle.json
kaggle config set -n path -v ./
kaggle competitions download -c idl-fall2021-hw1p2
```

After downloading, you must unzip the files
```
unzip competitions/idl-fall2021-hw1p2/train.npy.zip
unzip competitions/idl-fall2021-hw1p2/train_labels.npy.zip
unzip competitions/idl-fall2021-hw1p2/dev.npy.zip
unzip competitions/idl-fall2021-hw1p2/dev_labels.npy.zip
```

**IMPORTANT** Some way or another, make sure the datasets are on the same level as `hw.py`. You should have the following file names on the **same level**:
```
- hw.py
- train.npy
- train_labels.npy
- dev.npy
- dev_labels.npy
- test.npy
```


# Code
hw.py contains all the code necessary to train the model. Run:
```
python3 hw.py
```
When training, each epoch will take about half an hour. The model has about 78% accuracy on the test data at around 20 epochs

# Model Description

See code for more details:
I used 4 hidden layers, each consisting of a linear layer -> batchnorm -> sigmoid/ReLU -> dropout.
Context=20
Epochs=100
Batch Size=3194
optimizer=Adam with lr=.001







