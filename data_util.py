import os
import json
import pickle
import numpy as np
from collections import Counter

def tweet_to_npy(min_frequency_token=500000, max_amount_of_lines=500000):

    tweets = []
    set_of_chars = Counter()
    for file in os.listdir('Data/AllTweets'):
        with open('Data/AllTweets/' + file, encoding='utf-8') as data_file:
            data = data_file.readlines()
            for tweet in data:
                try:
                    tweet = tweet.split('\t')[2][:140].lower()
                    tweets.append(tweet)
                    for char in tweet:
                        set_of_chars[char] += 1

                except IndexError:
                    pass

    number_of_token = 0
    removed = 0
    token_init = len(set_of_chars)
    for k in list(set_of_chars):
        number_of_token += set_of_chars[k]
        if set_of_chars[k] < min_frequency_token:
            removed += set_of_chars[k]
            del set_of_chars[k]
    print('% of raw token remaining :', (number_of_token - removed)/number_of_token*100.0)
    print('Initial amount of tokens :', token_init)
    print('Current amount of tokens :', len(set_of_chars))
    print('% of remaining tokens :', len(set_of_chars)/token_init)
    print('Max amount of lines :', len(tweets))

    set_of_chars = sorted(list(set(set_of_chars)))
    # +1 to reserve 0 as a no char.
    char_to_int = dict((c, i + 1) for i, c in enumerate(set_of_chars))
    int_to_char = dict((i + 1, c) for i, c in enumerate(set_of_chars))


    for index_file in range(len(tweets)//max_amount_of_lines):
        offset1, offset2 = index_file * max_amount_of_lines, index_file+1 * max_amount_of_lines


        datas = np.zeros(shape=(max_amount_of_lines, 140, len(set_of_chars) + 1), dtype=np.bool)

        for i, tweet in enumerate(tweets):
            if (i >= offset1) & (i < offset2):
                for c, char in enumerate(tweet):
                    try:
                        datas[i - offset1, c, char_to_int[char]] = 1
                    except KeyError:
                        pass
        shape1, shape2 = datas.shape[0], datas.shape[1]

        np.save('tensors/Tweet_' + str(index_file), datas)
        del datas
        latent = np.random.normal(size=(shape1, shape2, 10))
        np.save('tensors/TweetLatent_' + str(index_file), latent)
    pickle.dump(char_to_int, open("char_to_int.pkl", "wb"))
    pickle.dump(int_to_char, open("int_to_char.pkl", "wb"))


def predictions_to_string(predictions):
    int_to_char = pickle.load(open("int_to_char.pkl", "rb"))
    pred_string = ""
    for prediction in predictions:
        for i in range(prediction.shape[0]):
            pred = np.argmax(prediction[i,:])
            pred_string += int_to_char[pred]
        pred_string += "\n"
    return pred_string
