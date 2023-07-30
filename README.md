# Computational Intelligence Lab 2023 -- Text Sentiment Classification

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

Specify training dataset path in the [config.yaml](src/config.yaml) file, then run

```
cd ./src
python baseline_script.py
```

### 4.2. Train cross validation

Define all the parameters in the [config.yaml](src/config.yaml) file to perform a k-fold cross validation.
The first section includes general parameters:

* `seed`: Seed value used for reproducibility
* `train_path`: Path to the training data CSV file
* `test_path`: Path to the test data CSV file
* `n_folds`: Number of folds used in cross-validation
* `test_batch`: Batch size for test data during inference, helps if you train different model with different training batch size
* `column_text_name`: Name of the column containing the text data in the CSV files
* `column_label_name`: Name of the column containing the labels in the CSV files. Default value is 'label'
* `grid`: Boolean flag to indicate whether to perform grid search
* `ensemble`: Boolean flag to indicate whether to use ensemble techniques
* `debug`: Boolean flag to enable debug mode
* `use_subsampling`: Boolean flag to use subsampling for training data, useful for testing the code locally
* `wandb`: Boolean flag to enable Weights & Biases (wandb) logging

The other sections allows to further specify training details depending on the setting you choose in the first section.

After completing the setup run the following commands:

```
cd ./src
python train_script.py
```


### 4.3. Train for submission

Specify all the parameters in the [config.yaml](src/config.yaml) file following the same format used for training a cross-validation, then run 
```
cd ./src
python submission_script.py
```
The code generates a Prediction.csv file, which is ready for submission on Kaggle.

## 5. Results
Performances of every training and cross-validation are stored in [results.txt](results.txt).

The table below shows the most significant performances obtained:

| Model                              | Accuracy(\%)     | Variance(\%) |
|------------------------------------|------------------|--------------|
| BoW + Logistic Regression          | 80.23            | 0.208        |
| TFIDF + Logistic Regression        | 80.18            | 0.207        |
| BoW + LinearSVM                    | 73.22            | 0.382        |
| TFIDF + LinearSVM                  | 71.32            | 0.382        |
| BoW + XGBClassifier                | 75.71            | 0.239        |
| TFIDF + XGBClassifier              | 76.28            | 0.193        |
| BoW + GradientBoostingClassifier   | 69.98            | 0.149        |
| TFIDF + GradientBoostingClassifier | 70.03            | 0.232        |
| GloVe + LSTM                       | 86.57            | 0.07         |
| distilBERT                         | 89.61            | 0.239        |
| roBERTaTweet                       | 90.78            | 0.245        |
| BERTweet-base                      | 91.43            | 0.247        |
| Enhanced BERTweet ensemble         | $\mathbf{92.02}$ | 0.249        |

The strongest performance (**Enhanced BERTweet ensemble**) was obtained with an ensemble of three models:
* BERTweet-base
* BERTweet-large
* BERTweet-large using LoRA (Low-Rank Adaptation)

The ensemble has been cross-validated on a NVIDIA A100 80GB PCIe on Euler Cluster using [train.sh](src/train.sh)

The final submission has been generated on a NVIDIA A100 80GB PCIe on Euler Cluster using [submission.sh](src/submission.sh)