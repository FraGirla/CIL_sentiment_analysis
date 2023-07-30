# Baselines

BoW + LogisticRegression (solver:sage) &rarr; mean: 0.80234 | std: 0.00208

Bow + LinearSVC &rarr; mean: 0.73223 | std: 0.00330

Bow + XGBClassifier &rarr; mean: 0.75714 | std: 0.00238

Bow + GradientBoostingClassifier &rarr; mean: 0.69983 | std: 0.00147

TFIDF + LogisticRegression (solver:sage) &rarr; mean: 0.80181 | std: 0.00207

TFIDF + LinearSVC &rarr; mean: 0.71319 | std: 0.00381

TFIDF + XGBClassifier &rarr; mean: 0.76275 | std: 0.00193

TFIDF + GradientBoostingClassifier &rarr; mean: 0.70025 | std: 0.00231

-------------------------------------------------------------
# Transformers models:

First attempts on cardiffnlp/twitter-roberta-base-sentiment

training 0 layers &rarr; 0.7809

training 1 layers &rarr; 0.8261

training 2 layers no embedding &rarr; 0.8319

training 6 layers no embedding &rarr; ~0.84

training 12 layers no embedding &rarr; ~0.852

training 6 layers no embedding clean_tweet preprocessing &rarr; 0.8929

training 12 layers no embedding no preprocessing &rarr; ~0.83

training 12 layers with embedding no preprocessing &rarr; 0.8266

-------------------------------------------------------------

### List of hyperparameters:
* batch_size: 32
* learning_rate: 5e-5
* classification_dropout: 0.1
* num_epoch 1
* max_len 128/512 (BERTweet 128 / cardiffnlp/twitter-roberta-base-sentiment 512)

roBERTa 12 layers no embedding clean_tweet column &rarr; mean: 0.89231 | std: 0.00234 | Kaggle LB: 0.89300

roBERTa 12 layers with embedding clean_tweet column and custom nn &rarr; mean: 0.89226 | std: 0.00227

BERTweet-base 12 layers no embedding clean_tweet column &rarr; mean: 0.89558 | std: 0.00126 | Kaggle LB: 0.89680

roBERTa 12 layers with embedding partial_clean_tweet column &rarr; mean: 0.90243 | std: 0.00210

-------------------------------------------------------------

### List of hyperparameters:
* batch_size: 16
* learning_rate: 2e-5
* classification_dropout: 0.15
* max_len 128/512 (BERTweet 128 / cardiffnlp/twitter-roberta-base-sentiment 512)
* num_epochs 1

BERTweet 12 layers with embedding partial_clean_tweet 
mean: 0.9134797831262629
std: 0.0023152341891710666
ON KAGGLE PUBLIC: 0.90180

roBERTa 12 layers with embedding partial_clean_tweet
mean: 0.9056603245103336
std: 0.0024353505267768684

distilBERT 12 layers with embedding partial_clean_tweet
mean: 0.8930100234079588
std: 0.0017764420009107934

-------------------------------------------------------------

### List of hyperparameters:
* batch_size: 16
* learning_rate: 2e-5
* classification_dropout: 0.15
* max_len 128/512 (BERTweet 128 / cardiffnlp/twitter-roberta-base-sentiment 512)
* num_epochs 2

BERTweet 12 layers with embedding partial_clean_tweet
mean: 0.9128991857231459
std: 0.002484387193798799

roBERTa 12 layers with embedding partial_clean_tweet 
mean: 0.907185242982614
std: 0.0029945574861834656

distilBERT 12 layers with embedding partial_clean_tweet (BEST on distilBERT)
mean: 0.8961158793989956
std: 0.0023943923687078307

-------------------------------------------------------------

### List of hyperparameters:
* batch_size: 16
* learning_rate: 2e-5
* classification_dropout: 0.15
* max_len 128/512 (BERTweet 128 / cardiffnlp/twitter-roberta-base-sentiment 512)
* num_epochs 3

BERTweet 12 layers with embedding partial_clean_tweet
mean: 0.9036492407418523
std: 0.016181499731669262

roBERTa 12 layers with embedding partial_clean_tweet
mean: 0.9062641298041333
std: 0.0024089994517277726

distilBERT 12 layers with embedding partial_clean_tweet
mean: 0.894216833723466
std: 0.0023228035688484545

-------------------------------------------------------------

BATCH_SIZE 32
learning_rate 2e-5
dropout classification 0.15
MAX_LEN 128/512 (BERTweet 128 / roBERTa 128)

num_epochs 1

BERTweet 12 layers with embedding partial_clean_tweet 
mean: 0.9124886461396876
std: 0.00322504078137404

roBERTa 12 layers with embedding partial_clean_tweet
mean: 0.9057519556649261
std: 0.002426953272849001

distilBERT 12 layers with embedding partial_clean_tweet
mean: 0.8929720104835643
std: 0.00033529729421979856

ensemble (roBERTa, BERTweet, distilBERT - avg) 12 layers with embedding partial_clean_tweet
mean: 0.9105339815537283
std: 0.0025031634141788454

num_epochs 2

roBERTa 12 layers with embedding partial_clean_tweet 
mean: 0.9076530020206871
std: 0.0024907350080335387

BERTweet 12 layers with embedding partial_clean_tweet 
mean: 0.9139335374027169
std: 0.0025318895977325443

distilBERT 12 layers with embedding partial_clean_tweet
mean: 0.8949234739811537
std: 0.0026915223650484667

ensemble (roBERTa, BERTweet, distilBERT - avg) 12 layers with embedding partial_clean_tweet
mean: 0.9133001220414941
std: 0.0026821156489913343

num_epochs 3

roBERTa 12 layers with embedding partial_clean_tweet
mean: 0.907092811555929
std: 0.0023172508241897974

BERTweet 12 layers with embedding partial_clean_tweet 
mean: 0.9124878458675949
std: 0.0024632228108515787

distilBERT 12 layers with embedding partial_clean_tweet
mean: 0.8944917271872438
std: 0.002465297054377858


-------------------------------------------------------------

BATCH_SIZE 64
learning_rate 2e-5
dropout classification 0.15
MAX_LEN 128/512 (BERTweet 128 / roBERTa 128)

num_epochs 1

roBERTa 12 layers with embedding partial_clean_tweet
mean: 0.9062225156553227
std: 0.0025479026509919803

BERTweet 12 layers with embedding partial_clean_tweet 
mean: 0.9131772802752935
std: 0.0026107641626406694

distilBERT 12 layers with embedding partial_clean_tweet
mean: 0.8914899065682332
std: 0.0007752614985841124

ensemble (roBERTa, BERTweet - avg) 12 layers with embedding partial_clean_tweet
mean: 0.9125722745733549
std: 0.002385719360845971

num_epochs 2

roBERTa 12 layers with embedding partial_clean_tweet (BEST on roBERTa)
mean: 0.9078350639217334
std: 0.002451552736652662

BERTweet 12 layers with embedding partial_clean_tweet (BEST on BERTweet)
mean: 0.914256047055999
std: 0.0024719754659474837

distilBERT 12 layers with embedding partial_clean_tweet
mean: 0.8948650541184003
std: 0.0025017242255875516

TwHIN_BERT 12 layers with embedding partial_clean_tweet
mean: 0.9033351
std acc: 0.0024155

ensemble (roBERTa, BERTweet - avg) 12 layers with embedding partial_clean_tweet
mean: 0.9157261468899426
std: 0.002542241815489397

ensemble (BERTweet-base, BERTweet-large, BERTweet-large( with lora and 3 epochs) - avg) 12 layers with embedding partial_clean_tweet 
mean: 0.9195766560630615
std: 0.0025651254842828206

num_epochs 3

BERTweet 12 layers with embedding partial_clean_tweet 
mean: 0.9134633775483664
std: 0.0023901977261908225

distilBERT 12 layers with embedding partial_clean_tweet
mean: 0.8952579877158232
std: 0.0025448153625205407

-------------------------------------------------------------

BATCH_SIZE 128
learning_rate 2e-5
dropout classification 0.15
MAX_LEN 128/512 (BERTweet 128 / roBERTa 128)

num_epochs 1

ensemble (roBERTa, BERTweet - avg) 12 layers with embedding partial_clean_tweet
mean: 0.9120945121341254
std: 0.0025957656482455686

num_epochs 2

BERTweet large 12 layers with embedding partial_clean_tweet (AWP)
mean: 0.91764
std acc: 0.00274

ensemble (roBERTa, BERTweet - avg) 12 layers with embedding partial_clean_tweet (no AWP)
mean: 0.9146933957545565
std: 0.0026071055624712187

ensemble (roBERTa, BERTweet - avg) 12 layers with embedding partial_clean_tweet (BEST with AWP)
mean: 0.9155420843086649
std: 0.002558821241916289
ON KAGGLE PUBLIC: 0.91660

ensemble (bertweet-large, bertweet-base, roBERTa, TwHIN_BERT) (with AWP)
mean: 0.9170614008762978
std: 0.0026564300717972383

-------------------------------------------------------------

BATCH_SIZE 256
learning_rate 2e-5
dropout classification 0.15
MAX_LEN 128/512 (BERTweet 128 / roBERTa 128)

num_epochs 1

ensemble (roBERTa, BERTweet - avg) 12 layers with embedding partial_clean_tweet
mean: 0.912431426685073
std: 0.0017185745830838658

num_epochs 2

ensemble (roBERTa, BERTweet - avg) 12 layers with embedding partial_clean_tweet
mean: 0.9139471420282895
std: 0.0026826352435131316
