# Computational Intelligence Lab 2023 -- Text Sentiment Classification

![Python package](https://github.com/oskopek/cil/workflows/Python%20package/badge.svg)

### Enrico Brusoni, Francesco Girlanda, Timon Kick, Matteo Mazzonelli
Department of Computer Science, ETH Zurich, Switzerland

## 1. Project definition

Perform text sentiment classification on Twitter data (tweets). The goal is to develop an automatic method to reveal the authors' opinion/sentiment.

## 2. Setup

Install the required packages by running the following command:
```
pip install -r requirements.txt
```

## 3. Preprocessing

Create a directory called twitter-datasets and download all data from https://www.kaggle.com/competitions/ethz-cil-text-classification-2023/data 

Run the following commands to preprocess training and test data:
```
cd ./preprocessed
python preprocess_train.py
python preprocess_train.py
```

## 4. Train models

### 4.1. Baselines models

Specify training dataset path in the `./src/config.yaml` file, then run

```
cd ./src
python baseline_script.py
```

### 4.2. Transformers models
