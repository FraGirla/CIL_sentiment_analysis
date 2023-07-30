import re
from TweetNormalizer import normalizeTweet
import wordsegment as ws
ws.load()
import pandas as pd
import numpy as np


def segment_hashtags(tweet):
    hashtags = re.findall(r"#\w+", tweet)

    # Replace each hashtag with segmented words and special tokens
    for hashtag in hashtags:
        hashtag_word = hashtag[1:]  # Consume the #
        segmented_words = ws.segment(hashtag_word)
        processed_hashtag = "HASHTAG_START " + " ".join(segmented_words) + " HASHTAG_END"
        tweet = tweet.replace(hashtag, processed_hashtag)

    return tweet


if __name__ == '__main__':

    tweets = []
    ids = []

    def load(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.split(',')
                ids.append(int(line[0]))
                line = ','.join(line[1:])
                tweets.append(line.rstrip())

    load('../twitter-datasets/test_data.txt')
    print('Loaded all tweets')

    # Replace 3 or more repeated letters with 2, normalize the tweets
    processed_tweets = [normalizeTweet(re.sub(r'(\w)\1{2,}', r'\1\1', tweet)) 
                        for tweet in tweets]
    print('Preprocessed all tweets')

    processed_tweets = np.array(processed_tweets)

    test_df = pd.DataFrame({'text': processed_tweets},index=ids)

    #num_positive_tweets = train_df['label'].sum()
    #num_negative_tweets = len(train_df) - num_positive_tweets
    #print("Initial counts:")
    #print("Positive Tweets:", num_positive_tweets)
    #print("Negative Tweets:", num_negative_tweets)

    # Filter out rows with both label 0 and 1 for the same tweet
    #train_df = train_df.groupby('text').filter(lambda x: not set([0, 1]).issubset(x['label']))

    #num_positive_tweets = train_df['label'].sum()
    #num_negative_tweets = len(train_df) - num_positive_tweets
    #print("Counts after filtering:")
    #print("Positive Tweets:", num_positive_tweets)
    #print("Negative Tweets:", num_negative_tweets)

    # Drop duplicate tweets
    #train_df = train_df.drop_duplicates(subset='text')

    #num_positive_tweets = train_df['label'].sum()
    #num_negative_tweets = len(train_df) - num_positive_tweets
    #print("Counts after dropping duplicates:")
    #print("Positive Tweets:", num_positive_tweets)
    #print("Negative Tweets:", num_negative_tweets)
    print(test_df.head())
    test_df.to_csv('../preprocessed/test_full_new.csv', index=False)
