"""
This file constructs the text data that includes sponsor description and video level data.

Sponsors (from a separate file):
    - description (creator level, not video level)

Sponsored videos content:
    - video title
    - video description
    - video topics
    - video labels from vca
    - video transcript
"""

import pandas as pd
import os
import json
import numpy as np
import ast

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data")

# Load data (1: meta data, 2: vca data, 3: transcripts, 4: pooled data)
meta = pd.read_csv(os.path.join(PATH, "data_all_text.csv") , low_memory=False)

vca = pd.read_csv(os.path.join(PATH, "vca_types_17000_feb2021.csv"))

transcripts = pd.read_csv(os.path.join(PATH, "videos_transcripts.csv"))

pooled = pd.read_csv(os.path.join(PATH, "pooled_us_jul2024.csv"))

# modifying the dataframes
# 1. meta data (removing duplicates)
meta = meta.drop_duplicates(subset=['video_id','creator_id'], keep='first')
transcripts = transcripts.drop_duplicates(subset=['video_id','creator_id','transcript'], keep='first')
pooled['video_id'] = pooled['video_id'].astype(np.int64)

# Creating the merged data including:
# 1. video_id
# 2. new_id
# 3. creator_id
# 4. video_title
# 5. video_description (title and description is stored in text column in meta data)
# 6. video_topics
# 7. video_labels
# 8. video_transcript

# base data for sponsored videos
mask = pooled.sponsored==1
sponsored = pooled[mask][['video_id','new_id','creator_id','creator_name','sponsor_id','sponsor_name']].copy()

# adding video title, description, and topics
sponsored['title_description'] = sponsored.merge(meta, left_on='new_id', right_on='video_id', how='left').text.values
sponsored['topics'] = sponsored.merge(meta, left_on='new_id', right_on='video_id', how='left').topics.values

# adding video labels
sponsored['labels'] = sponsored.merge(vca, on='video_id', how='left').segment_labels.values
sponsored['labels2'] = sponsored.merge(vca, on='video_id', how='left').shot_labels.values

# adding video transcript
sponsored['transcript'] = sponsored.merge(transcripts, on=['video_id','creator_id'], how='left').transcript.values

print(sponsored.head())
print(sponsored.columns)
print(sponsored.notnull().sum())

def print_text(row):
    """
    Print the text data for a given row.
    """
    desc = row['title_description'].iloc[0]
    if row['topics'].isnull().all():
        topic = ''
    else:
        ' '.join(row['topics'].iloc[0].split(';'))
    if row['labels'].isnull().all():
        labels = ''
    else:
        labels = ' '.join(ast.literal_eval(row['labels'].iloc[0]))
    if row['labels2'].isnull().all():
        labels2 = ''
    else:
        labels2 = ' '.join(ast.literal_eval(row['labels2'].iloc[0]))
    if row['transcript'].isnull().all():
        transcript = ''
    else:
        transcript = row['transcript'].iloc[0]
    #print(f"Title: {row['title_description']} \n Topics: {row['topics']} \n Labels: {row['labels']} and {row['labels2']} \n Transcript: {row['transcript']} \n")
    print(f"Title: {desc}\nTopics: {topic}\nLabels: {labels} {labels2}\nTranscript: {transcript}")

# print_text(sponsored.loc[204112])

def create_text(row):
    """
    Create the text data for a given row.
    """
    desc = row['title_description'].lower()
    if type(row['topics']) is not str:
        topic = ''
    else:
        topic = ' '.join(row['topics'].lower().split(';'))
    if type(row['labels']) is not str:
        labels = ''
    else:
        labels = ' '.join(ast.literal_eval(row['labels'].lower()))
    if type(row['labels2']) is not str:
        labels2 = ''
    else:
        labels2 = ' '.join(ast.literal_eval(row['labels2'].lower()))
    if type(row['transcript']) is not str:
        transcript = ''
    else:
        transcript = row['transcript'].lower()
    return f"{desc} {topic} {labels} {labels2} {transcript}"
    
sponsored['text'] = sponsored.apply(create_text, axis=1)

# everything together
sponsored.to_csv(os.path.join(PATH,"sponsored_videos_data.csv"), index=False)
 