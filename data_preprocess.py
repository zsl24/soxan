import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

import torchaudio
from sklearn.model_selection import train_test_split

import os
import sys

import json
data = []
jsonfile = open('sichuan_mandarin_classifer.json','r',encoding='utf-8')

for line in jsonfile.readlines():
    dic = json.loads(line)
    if dic['lang'] != 2 and 'mix' not in dic['text']:
        data.append({
            "path": dic['audio_filepath'],
            "label": dic['lang']
        })
df = pd.DataFrame(data)
print('number of sample is: ',len(df))

print("Labels: ", df["label"].unique())
print()
df.groupby("label").count()[["path"]]

save_path = "dataset"

train_df, test_df = train_test_split(df, test_size=0.1, random_state=101, stratify=df["label"])
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
train_df.to_csv(f"{save_path}/train.csv", sep="\t", encoding="utf-8", index=False)
test_df.to_csv(f"{save_path}/test.csv", sep="\t", encoding="utf-8", index=False)
print('number of sample in training set: ',train_df.shape)
print('number of sample in test set: ',test_df.shape)

