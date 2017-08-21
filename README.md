# GLO-Keras-tweet

Naive attempt at implementing https://arxiv.org/abs/1707.05776 on whole tweet generation

This attempts to train a densenet to make a whole tweet at once

Data was taken from https://archive.org/details/twitter_cikm_2010

The generation is made on chars, so we attempt to generate a sequence of 140 chars at once, this implementation does not seem to be giving any good result
