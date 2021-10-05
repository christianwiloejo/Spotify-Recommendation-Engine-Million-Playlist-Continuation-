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
from pyspark.ml.recommendation import ALS

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

# # Check out sparsity value
# playlists = raw_df.select('pid').distinct().count()
# songs = raw_df.select('track_uri').distinct().count()

# denom = playlists*songs
# numera = raw_df.select('track_uri').count()

# sparsity = 1-((numera*1.0)/denom)
# print(f"Sparsity: {sparsity}")

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
tracks_distinct = subset_1000.select('track_uri').distinct()
tracks_distinct = tracks_distinct.coalesce(1)
tracks_distinct = tracks_distinct.rdd.zipWithIndex().toDF()
tracks_distinct = tracks_distinct.withColumnRenamed('_1', 'track_uri')\
                    .withColumnRenamed('_2', 'track_uri_int')
tracks_distinct = tracks_distinct.select('track_uri.*', 'track_uri_int')

# Join the tracks_distinct with unique id per track to the main dataframe
subset_1000 = subset_1000.join(tracks_distinct, ['track_uri'], "left")
subset_1000.orderBy(['pid', 'track_uri_int'], ascending = True).show()

# COMMAND ----------

# Select the relevant columns
subset_1000_clean = subset_1000.select('pid', 'track_uri_int', 'song_exist_in_playlist')

# Extract distinct playlists 
playlists_subset_1000 = subset_1000_clean.select('pid').distinct()

# Extract distinct tracks
songs_subset_1000 = subset_1000_clean.select('track_uri_int').distinct()

cross_join = playlists_subset_1000.crossJoin(songs_subset_1000).join(subset_1000_clean, ['pid', 'track_uri_int'], 'left').fillna(0)
cross_join.orderBy('pid').show()


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

# Train, Test Split
(train, test) = cross_join.randomSplit([0.8, 0.2], seed = 123)

# Build ALS Model
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

recommend_10 = model.recommendForAllUsers(5)\
  .selectExpr("pid", "explode(recommendations) as recommendation_val_score")\
  .select("pid", 'recommendation_val_score.*')\
  .withColumnRenamed("pid", "recommend_pid")
recommend_10.show()

# COMMAND ----------

recommend_10_w_track_uri = recommend_10\
                              .join(tracks_distinct, on = 'track_uri_int', how = 'left')\
                              .join(cross_join, (recommend_10.recommend_pid == cross_join.pid) & (recommend_10.track_uri_int == cross_join.track_uri_int), how = 'left')

display(recommend_10_w_track_uri.orderBy('recommend_pid', ascending=True))

# COMMAND ----------


display(subset_1000.filter(subset_1000.pid == 5))
