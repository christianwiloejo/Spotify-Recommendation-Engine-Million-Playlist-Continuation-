# Databricks notebook source
# MAGIC %md 
# MAGIC # Spotify Million Playlist Challenge
# MAGIC Given playlists and tracks data, can we suggest more tracks to be added to these playlists?

# COMMAND ----------

# MAGIC %md 
# MAGIC # Import Packages

# COMMAND ----------

import os
import numpy as np
from pyspark.sql.functions import *
from functools import reduce
from pyspark.sql import DataFrame
from pprint import pprint
import pandas as pd

# COMMAND ----------

# MAGIC %md 
# MAGIC # File Complete Checker (One Time Process)

# COMMAND ----------

# def completeUpload(file_path = '/dbfs/FileStore/spotify_million_playlist_raw_data/', count = 1000, checklist = pd.DataFrame({'file_range':np.arange(0,1000000,1000)})):
#   directory_files_df = pd.DataFrame({'files_in_dir': os.listdir(file_path)})
#   if directory_files_df.shape[0] == count: 
#     print('Complete JSON files uploaded to DBFS...')
#   else: 
#     print('The following files are missing')
#     directory_files_df['left_num'] = directory_files_df['files_in_dir'].str.split('mpd_slice_').str[1].str.split('_').str[0].astype('int')
#     checklist = pd.DataFrame({'file_range':np.arange(0,1000000,1000).tolist()})
#     print(checklist.loc[~checklist['file_range'].isin(directory_files_df['left_num'])])
# completeUpload()

# COMMAND ----------

# %fs rm -r 'dbfs:/FileStore/spotify_million_playlist_raw_data/'

# COMMAND ----------

# MAGIC %md 
# MAGIC # Reading, Transforming and Writing JSON Raw Data to Parquet (One Time Process)

# COMMAND ----------

# DfAppend = []

# path = '/dbfs/FileStore/spotify_million_playlist_raw_data/'
# count = 0 
# for json_file in os.listdir(path):
  
#   json_file_path = 'dbfs:/FileStore/spotify_million_playlist_raw_data/' + json_file
#   master_df = (spark.read
#       .option("inferSchema", True)
#       .option("multiline", True)
#       .json(json_file_path))
#   print(json_file_path)
#   playlists_df = master_df.select(explode("playlists").alias('playlists'))
  
#   playlists_tracks_df = playlists_df.select("playlists.pid", "playlists.name", explode("playlists.tracks").alias("tracks"))
#   playlists_tracks_df = playlists_tracks_df.withColumnRenamed("playlists.pid", "pid").withColumnRenamed("playlists.name", "playlist_name")
  
#   final_df = playlists_tracks_df.select("pid","playlist_name", "tracks.*")
  
  
#   DfAppend.append(final_df)
#   count += 1
  
#   print(f"Completed {count}/{len(os.listdir(path))}")

# #Concatenate all dataframes
# df = reduce(DataFrame.unionAll, DfAppend)

# #Write to parquet
# df.write.parquet("/dbfs/FileStore/spotify_million_playlist_raw_data/all_playlists_tracks_parsed.parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Read Parquet File

# COMMAND ----------

raw_df = spark.read.parquet("/dbfs/FileStore/spotify_million_playlist_raw_data/all_playlists_tracks_parsed.parquet")

raw_df.show()

# COMMAND ----------

raw_df.select('pid').distinct().count()

# COMMAND ----------

# MAGIC %md 
# MAGIC # 2. Prepare Data for PySpark ALS Model 
# MAGIC The goal: we need to end up with a dataframe that lists all playlists, tracks, and a column to indicate whether a track is in the playlist or not.  
# MAGIC The way: we need to cross join the distinct playlist series to the distinct tracks series.

# COMMAND ----------

# Check out sparsity value
playlists = raw_df.select('pid').distinct().count()
songs = raw_df.select('track_uri').distinct().count()

denom = playlists*songs
numera = raw_df.select('track_uri').count()

sparsity = 1-((numera*1.0)/denom)
print(f"Sparsity: {sparsity}")

# COMMAND ----------

raw_df = raw_df.withColumn("song_exist_in_playlist", lit(1))
raw_df.show()

# COMMAND ----------

# DELETE

# def add_zeros(raw_df):

  
#   # Joins playlists and tracks, fills blanks with 0
#   cross_join_df = playlists.crossJoin(songs).join(raw_df, ['pid', 'track_uri'], 'left').fillna(0)
  
#   return cross_join_df

# COMMAND ----------

# MAGIC %md 
# MAGIC # 3. Subset of the Data

# COMMAND ----------

subset_1000 = raw_df.filter(raw_df['pid'] < 1000).select('pid', 'track_uri', 'song_exist_in_playlist')


# Replace the track_uri with integer for ALS model
tracks_distinct = raw_df.select('track_uri').distinct()
tracks_distinct = tracks_distinct.coalesce(1)
tracks_distinct = tracks_distinct.withColumn("track_uri_int", monotonically_increasing_id()).persist()
subset_1000 = subset_1000.join(tracks_distinct, "track_uri", "left")
subset_1000.orderBy('track_uri_int', ascending = True).show()

# COMMAND ----------

subset_1000_clean = subset_1000.select('pid', 'track_uri_int', 'song_exist_in_playlist')
# Extract distinct playlists 
playlists_subset_1000 = subset_1000_clean.select('pid').distinct()
# playlists = playlists.coalesce(32)

# Extract distinct tracks
songs_subset_1000 = subset_1000_clean.select('track_uri_int').distinct()
# songs = songs.coalesce(32)

cross_join = playlists_subset_1000.crossJoin(songs_subset_1000).join(subset_1000_clean, ['pid', 'track_uri_int'], 'left').fillna(0)
cross_join.show()


# COMMAND ----------

cross_join.filter(cross_join['song_exist_in_playlist'] != 0).show()

# COMMAND ----------

display(cross_join.where("song_exist_in_playlist<>0").groupBy('track_uri_int').count())

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 3.1 ROEM Function for Evaluating Our Recommender System

# COMMAND ----------

# This function was initially created for the Million Songs Echo Nest Taste Profile dataset which had 3 columns: userId, songId,
# and num_plays. The column num_plays was used as implicit ratings with the ALS algorithm.

def ROEM(predictions):
  #Creates predictions table that can be queried
  predictions.createOrReplaceTempView("predictions") 
  
  #Sum of total number of plays of all songs
  denominator = predictions.groupBy().sum("song_exist_in_playlist").collect()[0][0]
  
  #Calculating rankings of songs predictions by user
  spark.sql("SELECT pid, song_exist_in_playlist, PERCENT_RANK() OVER (PARTITION BY pid ORDER BY prediction DESC) AS rank FROM predictions").createOrReplaceTempView("rankings")
  
  #Multiplies the rank of each song by the number of plays for each user
  #and adds the products together
  numerator = spark.sql('SELECT SUM(song_exist_in_playlist * rank) FROM rankings').collect()[0][0]
                         
  return numerator / denominator

# COMMAND ----------

(train, test) = cross_join.randomSplit([0.8, 0.2])

# Build ALS Model
from pyspark.ml.recommendation import ALS
als = ALS(
  userCol = "pid",
  itemCol = "track_uri_int",
  ratingCol = "song_exist_in_playlist", 
  rank = 25, 
  maxIter = 100, 
  regParam=.05, 
  alpha = 20,
  nonnegative=True, 
  coldStartStrategy="drop", 
  implicitPrefs=True)

# Fit model to train data
model = als.fit(train)

#Generate predictions on test data
prediction = model.transform(test)

#Tell Spark how to evaluate predictions
print(ROEM(prediction))

# COMMAND ----------

five_playlist_prediction = prediction.filter(prediction.pid < 5)
five_playlist_prediction.show()

# COMMAND ----------

display(prediction.filter().distinct().join(subset_1000, "track_uri_int", "left").join(raw_df.select('track_uri', 'track_name'), 'track_uri', "left"))

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. All Data

# COMMAND ----------



# COMMAND ----------

# Extract distinct playlists 
playlists = raw_df.select('pid').distinct()
playlists = playlists.coalesce(32)
# print(playlists.count())
# print(playlists.rdd.getNumPartitions())

# Extract distinct tracks
songs = raw_df.select('track_uri').distinct()
songs = songs.coalesce(32)

# print(songs.count())
# print(raw_df.count())
print(songs.rdd.getNumPartitions())

# COMMAND ----------

# Joins playlists and tracks, fills blanks with 0
cross_join_df = playlists.crossJoin(broadcast(songs)).join(broadcast(raw_df), ['pid', 'track_uri'], 'left').fillna(0)

# COMMAND ----------

sqlContext.getConf("spark.driver.maxResultSize")

# COMMAND ----------

cross_join_df.write.parquet("/dbfs/FileStore/spotify_million_playlist_raw_data/02 cross_joined_all_songs_all_playlists_joined_original_df.parquet")

# COMMAND ----------



# COMMAND ----------


