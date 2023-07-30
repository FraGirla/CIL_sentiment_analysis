# Baselines

BoW + LogisticRegression (solver:sage)
mean acc:  0.8023419962787347
std acc:  0.0020851275577559026

Bow + LinearSVC
mean acc:  0.7322389612468239
std acc:  0.003308177849289995

Bow + XGBClassifier
mean acc:  0.7571470299901967
std acc:  0.002386290879788144

Bow + GradientBoostingClassifier
mean acc:  0.699838745173359
std acc:  0.0014779866473545802

TFIDF + LogisticRegression (solver:sage)
mean acc:  0.8018114158813997
std acc:  0.002074899902368564

TFIDF + LinearSVC
mean acc:  0.7131920853090051
std acc:  0.003817513336296751

TFIDF + XGBClassifier
mean acc:  0.7627509353180081
std acc:  0.0019348882102178692

TFIDF + GradientBoostingClassifier
mean acc:  0.7002572874777424
std acc:  0.002315283288343248

-------------------------------------------------------------
TRANSFORMERS MODELS:
-------------------------------------------------------------

First attempts on cardiffnlp/twitter-roberta-base-sentiment
training 0 layers: 0.7809
training 1 layers: 0.8261
training 2 layers no embedding: 0.8319
training 6 layers no embedding ~0.84
training 12 layers no embedding ~0.852
training 6 layers no embedding clean_tweet 0.8929
training 12 layers no embedding no preprocessing ~0.83
training 12 layers with embedding no preprocessing 0.8266

-------------------------------------------------------------

Hyperparameters
BATCH_SIZE 32
learning_rate 5e-5
dropout classification 0.1
num_epoch 1
MAX_LEN 128/512 (BERT 128 / roBERTa 512)

roBERTa 12 layers no embedding clean_tweet
mean acc:  0.8923125862793351
std acc:  0.0023438754395944676
ON KAGGLE PUBLIC: 0.893

roBERTa 12 layers with embedding clean_tweet
mean acc:  0.89891753196 based on 4 fold (1 fold gived 0.5)
std acc:  0.0025041120717059

roBERTa 12 layers with embedding clean_tweet and custom nn
mean acc:  0.8922649700898304
std acc:  0.002273580409090531

BERT 12 layers no embedding clean_tweet 
mean acc: 0.89558 
std acc: 0.00126
ON KAGGLE PUBLIC: 0.8968

roBERTa 12 layers with embedding partial_clean_tweet
mean acc:  0.902438028929836
std acc:  0.0021023840899968478

-------------------------------------------------------------
BATCH_SIZE 16
learning_rate 2e-5
dropout classification 0.15
MAX_LEN 128/512 (BERT 128 / roBERTa 128)

num_epochs 1

BERT 12 layers with embedding partial_clean_tweet 
mean acc:  0.9134797831262629
std acc:  0.0023152341891710666
ON KAGGLE PUBLIC: 0.90180

roBERTa 12 layers with embedding partial_clean_tweet
mean acc:  0.9056603245103336
std acc:  0.0024353505267768684

distilBERT 12 layers with embedding partial_clean_tweet
mean acc:  0.8930100234079588
std acc:  0.0017764420009107934

num_epochs 2

BERT 12 layers with embedding partial_clean_tweet
mean acc:  0.9128991857231459
std acc:  0.002484387193798799

roBERTa 12 layers with embedding partial_clean_tweet 
mean acc:  0.907185242982614
std acc:  0.0029945574861834656

distilBERT 12 layers with embedding partial_clean_tweet (BEST on distilBERT)
mean acc:  0.8961158793989956
std acc:  0.0023943923687078307

num_epochs 3

BERT 12 layers with embedding partial_clean_tweet
mean acc:  0.9036492407418523
std acc:  0.016181499731669262

roBERTa 12 layers with embedding partial_clean_tweet
mean acc:  0.9062641298041333
std acc:  0.0024089994517277726

distilBERT 12 layers with embedding partial_clean_tweet
mean acc:  0.894216833723466
std acc:  0.0023228035688484545

-------------------------------------------------------------

BATCH_SIZE 32
learning_rate 2e-5
dropout classification 0.15
MAX_LEN 128/512 (BERT 128 / roBERTa 128)

num_epochs 1

BERT 12 layers with embedding partial_clean_tweet 
mean acc:  0.9124886461396876
std acc:  0.00322504078137404

roBERTa 12 layers with embedding partial_clean_tweet
mean acc:  0.9057519556649261
std acc:  0.002426953272849001

distilBERT 12 layers with embedding partial_clean_tweet
mean acc:  0.8929720104835643
std acc:  0.00033529729421979856

ensemble (roBERTa, BERTweet, distilBERT - avg) 12 layers with embedding partial_clean_tweet
mean acc:  0.9105339815537283
std acc:  0.0025031634141788454

num_epochs 2

roBERTa 12 layers with embedding partial_clean_tweet 
mean acc:  0.9076530020206871
std acc:  0.0024907350080335387

BERT 12 layers with embedding partial_clean_tweet 
mean acc:  0.9139335374027169
std acc:  0.0025318895977325443

distilBERT 12 layers with embedding partial_clean_tweet
mean acc:  0.8949234739811537
std acc:  0.0026915223650484667

ensemble (roBERTa, BERTweet, distilBERT - avg) 12 layers with embedding partial_clean_tweet
mean acc:  0.9133001220414941
std acc:  0.0026821156489913343

num_epochs 3

roBERTa 12 layers with embedding partial_clean_tweet
mean acc:  0.907092811555929
std acc:  0.0023172508241897974

BERT 12 layers with embedding partial_clean_tweet 
mean acc:  0.9124878458675949
std acc:  0.0024632228108515787

distilBERT 12 layers with embedding partial_clean_tweet
mean acc:  0.8944917271872438
std acc:  0.002465297054377858


-------------------------------------------------------------

BATCH_SIZE 64
learning_rate 2e-5
dropout classification 0.15
MAX_LEN 128/512 (BERT 128 / roBERTa 128)

num_epochs 1

roBERTa 12 layers with embedding partial_clean_tweet
mean acc:  0.9062225156553227
std acc:  0.0025479026509919803

BERT 12 layers with embedding partial_clean_tweet 
mean acc:  0.9131772802752935
std acc:  0.0026107641626406694

distilBERT 12 layers with embedding partial_clean_tweet
mean acc:  0.8914899065682332
std acc:  0.0007752614985841124

ensemble (roBERTa, BERTweet - avg) 12 layers with embedding partial_clean_tweet
mean acc:  0.9125722745733549
std acc:  0.002385719360845971

num_epochs 2

roBERTa 12 layers with embedding partial_clean_tweet (BEST on roBERTa)
mean acc:  0.9078350639217334
std acc:  0.002451552736652662

BERT 12 layers with embedding partial_clean_tweet (BEST on BERT)
mean acc:  0.914256047055999
std acc:  0.0024719754659474837

distilBERT 12 layers with embedding partial_clean_tweet
mean acc:  0.8948650541184003
std acc:  0.0025017242255875516

TwHIN_BERT 12 layers with embedding partial_clean_tweet
mean acc: 0.9033351
std acc: 0.0024155

ensemble (roBERTa, BERTweet - avg) 12 layers with embedding partial_clean_tweet
mean acc:  0.9157261468899426
std acc:  0.002542241815489397

ensemble (BERTweet-base, BERTweet-large, BERTweet-large( with lora and 3 epochs) - avg) 12 layers with embedding partial_clean_tweet 
mean acc:  0.9195766560630615
std acc:  0.0025651254842828206

num_epochs 3

BERT 12 layers with embedding partial_clean_tweet 
mean acc:  0.9134633775483664
std acc:  0.0023901977261908225

distilBERT 12 layers with embedding partial_clean_tweet
mean acc:  0.8952579877158232
std acc:  0.0025448153625205407

-------------------------------------------------------------

BATCH_SIZE 128
learning_rate 2e-5
dropout classification 0.15
MAX_LEN 128/512 (BERT 128 / roBERTa 128)

num_epochs 1

ensemble (roBERTa, BERTweet - avg) 12 layers with embedding partial_clean_tweet
mean acc:  0.9120945121341254
std acc:  0.0025957656482455686

num_epochs 2

BERTweet large 12 layers with embedding partial_clean_tweet (AWP)
mean acc: 0.91764
std acc: 0.00274

ensemble (roBERTa, BERTweet - avg) 12 layers with embedding partial_clean_tweet (no AWP)
mean acc:  0.9146933957545565
std acc:  0.0026071055624712187

ensemble (roBERTa, BERTweet - avg) 12 layers with embedding partial_clean_tweet (BEST with AWP)
mean acc:  0.9155420843086649
std acc:  0.002558821241916289
ON KAGGLE PUBLIC: 0.91660

ensemble (bertweet-large, bertweet-base, roBERTa, TwHIN_BERT) (with AWP)
mean acc:  0.9170614008762978
std acc:  0.0026564300717972383

-------------------------------------------------------------

BATCH_SIZE 256
learning_rate 2e-5
dropout classification 0.15
MAX_LEN 128/512 (BERT 128 / roBERTa 128)

num_epochs 1

ensemble (roBERTa, BERTweet - avg) 12 layers with embedding partial_clean_tweet
mean acc:  0.912431426685073
std acc:  0.0017185745830838658

num_epochs 2

ensemble (roBERTa, BERTweet - avg) 12 layers with embedding partial_clean_tweet
mean acc:  0.9139471420282895
std acc:  0.0026826352435131316
