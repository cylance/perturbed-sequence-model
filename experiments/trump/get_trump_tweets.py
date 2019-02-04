"""
Source of file:
    https://www.kaggle.com/austinvernsonger/donaldtrumptweets/#data.csv
"""

import csv
import os

csv.field_size_limit(100000000)

THIS_DIR = os.path.dirname(__file__)


def get_rows(filepath):
    """
    Gets rows from a csv.  However, this will need processing

    :returns rows: List of strings
    """

    rows = []
    with open(filepath) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=" ", quotechar="|")
        for row in csvreader:
            rows.append(row)
    return list(reversed(rows))


def process_rows_of_trump_tweets(rows):
    """
    """
    text = []
    punctuation = [".", ",", "!", '"', "-", "?", ":"]
    # a "row" is a list of strings, but only some of them are words
    for row in rows[:-1]:  # first row is header etc.
        for (idx, entry) in enumerate(row):
            ## get text out of the row entries
            if idx == len(row) - 1:
                # this entry is a timestamp, e.g. '13:15:14,14906,3925,811560662853939200'
                continue
            elif idx == len(row) - 2:
                # this entry is a comma-delimited 2-ple of a word and a date, e.g. ['states!', '2016-12-21']
                entry = entry.split(",")[0]
                ##remove punctuation if necessary and grab the word
            word = entry
            while len(word) > 1:
                if word[-1] in punctuation:
                    word = word[:-1]
                elif word[0] in punctuation:
                    word = word[1:]
                else:
                    break
                    ## more cleaning
            if (
                (len(word) > 0)
                and (word not in punctuation)
                and (word[:4] != "http")
                and (word[0] is not "#")
            ):
                text.append(word.lower())
    return text


def get_sample_of_trump_tweets(filepath=os.path.join(THIS_DIR, "trump_tweet_data.csv")):
    rows = get_rows(filepath)
    sample = process_rows_of_trump_tweets(rows)
    return sample
