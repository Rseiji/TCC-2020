import pandas as pd 
import numpy as np
import os 
import csv 

from langdetect import detect 

#returns full dataset from the given path
def get_full_dataset(full_dataset_path: str):
    try:
        full_dataset = pd.read_csv(full_dataset_path, sep=",", index_col="id")
        return full_dataset

    except Exception as error:
        print(f'Error reading the file. Error: {error}')

#detects the language of each comment in the full dataset
#returns full dataset with the added column "language"
def detect_languages(full_dataset_path: str):
    full_dataset = get_full_dataset(full_dataset_path)

    #creates column language
    full_dataset["language"] = np.nan
    #gets a list of the comments from the full dataset
    comments = full_dataset["comment"].to_list()

    comment_languages = []
    #move = 0
    for comment in comments:
        #checks if move has a comment
        try:
            #detects comment language
            comment_languages.append(detect(comment))
        except:
            comment_languages.append(np.nan)

        #print(f'detecting language in move: {move}')
        #move += 1

    #saves languages detected to a log file
    language_log = open("language_log.txt", "w")
    for language in comment_languages:
        language_log.write(f'{language}\n')
    language_log.close()

    #saves detected languages in the full dataset
    full_dataset["language"] = comment_languages

    return full_dataset

#gets languages of the commentaries from a log file
#returns full dataset with the added column "language"
def get_languages_from_log(full_dataset_path: str):
    full_dataset = get_full_dataset(full_dataset_path)
    
    comment_languages = []
    log = open("language_log.txt")

    for i in range(len(full_dataset["move"])):
        comment_languages.append(log.readline().split("\n")[0])
    log.close()

    full_dataset["language"] = comment_languages

    return full_dataset


#filters matches with commentary in other languages out of the full dataset
#saves filtered full dataset to file
def language_filter():
    full_dataset_path = "full_dataset_segment_last_comment_label_100.csv"

    #check if log with the identified languages exists
    #otherwise, detects the languages in the dataset
    if os.path.isfile("language_log.txt"):
        full_dataset = get_languages_from_log(full_dataset_path)
    else:
        full_dataset = detect_languages(full_dataset_path)

    #gets list of the pgns in the dataset
    pgns = full_dataset["pgn_number"].unique()
    pgns_dropped = []

    #checks if over 50% of the commentary in one pgn is done in english
    #if false, that pgn is dropped from the dataset
    for pgn in pgns:
        #print(f'Checking pgn {pgn}')
        match = full_dataset[full_dataset["pgn_number"] == pgn]
        match.dropna(inplace=True)

        if match[match["language"] == 'en']["language"].count() < 0.5 * match["language"].count():
            full_dataset.drop(full_dataset[full_dataset["pgn_number"] == pgn].index, inplace=True)
            pgns_dropped.append(pgn)

    #saves the numbers of the dropped pgns to a log
    pgns_dropped_log = open("pgns_dropped_log.txt", "w")
    for pgn in pgns_dropped:
        pgns_dropped_log.write(f'{pgn}\n')
    pgns_dropped_log.close()

    #saves filtered dataset to a new file
    full_dataset.drop(columns='language', inplace=True)
    full_dataset.to_csv("filtered_full_dataset.csv", index=True, quoting=csv.QUOTE_NONNUMERIC)

    
if __name__ == "__main__":
    language_filter()
