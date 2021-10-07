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
from pyspark.sql.types import IntegerType

# COMMAND ----------

# MAGIC %md 
# MAGIC # Functions

# COMMAND ----------

# MAGIC %md 
# MAGIC ## ROEM
# MAGIC Since we are working with binary implicit ratings dataset, we can't use the regression evaluator provided by SparkML. Thus, we are creating ROEM function to evaluate whether a prediction is good or bad. 
# MAGIC Good prediction is indicated by a ROEM closer to 0 and abd prediction is indicated by ROEM closer to 0.5.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src ='files/media/ROEM.png'>

# COMMAND ----------

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

# MAGIC %md
# MAGIC # 1. Read Parquet File
# MAGIC As we can see, there are 1 million playlists, which contains 2,262,292 distinct tracks.

# COMMAND ----------

raw_df = spark.read.parquet("/dbfs/FileStore/spotify_million_playlist_raw_data/all_playlists_tracks_parsed.parquet")

raw_df.show(5)

# COMMAND ----------

raw_df.select('pid').distinct().count()

# COMMAND ----------

raw_df.select('track_uri').distinct().count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 Check Sparsity

# COMMAND ----------

# # Check out sparsity value
# playlists = raw_df.select('pid').distinct().count()
# songs = raw_df.select('track_uri').distinct().count()

# denom = playlists*songs
# numera = raw_df.select('track_uri').count()

# sparsity = 1-((numera*1.0)/denom)
# print(f"Sparsity: {sparsity}")

# COMMAND ----------

# MAGIC %md 
# MAGIC # 2. Prepare Data for PySpark ALS Model 
# MAGIC The goal: we need to end up with a dataframe that lists all playlists, tracks, and a column to indicate whether a track is in the playlist or not.  
# MAGIC The way: we need to cross join the distinct playlist series to the distinct tracks series.

# COMMAND ----------

raw_df = raw_df.withColumn("song_exist_in_playlist", lit(1))
raw_df.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC # 3. Subset of the Data
# MAGIC For time efficiency, in this notebook, we are going to work on subset of the complete dataset. Let's work on playlists with id **less than 1000**. 

# COMMAND ----------

# raw_df = raw_df.filter(raw_df['pid'] < 1000).select('pid', 'track_uri', 'song_exist_in_playlist')

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 4. 

# COMMAND ----------

# Replace the track_uri with integer for ALS model
tracks_distinct = raw_df.select('track_uri').distinct()
tracks_distinct = tracks_distinct.coalesce(1)
tracks_distinct = tracks_distinct.rdd.zipWithIndex().toDF()
tracks_distinct = tracks_distinct.withColumnRenamed('_1', 'track_uri')\
                    .withColumnRenamed('_2', 'track_uri_int')
tracks_distinct = tracks_distinct.select('track_uri.*', 'track_uri_int')

# Join the tracks_distinct with unique id per track to the main dataframe
df = raw_df.join(tracks_distinct, ['track_uri'], "left")
df.orderBy(['pid', 'track_uri_int'], ascending = True).show()

# COMMAND ----------

# Select the relevant columns
df_clean = df.select('pid', 'track_uri_int', 'song_exist_in_playlist')
df_clean = df_clean.selectExpr("cast(pid as int) pid", "cast(track_uri_int as int) track_uri_int", "cast(song_exist_in_playlist as int) song_exist_in_playlist")

# Extract distinct playlists 
playlists_distinct = df_clean.select('pid').distinct()
playlists_distinct = playlists_distinct.selectExpr('cast(pid as int) pid')
# playlists_distinct = playlists_distinct.withColumnRenamed("pid", "repartition_id")
playlists_distinct = playlists_distinct.repartition(400, playlists_distinct["pid"])

# Extract distinct tracks
tracks_distinct = df_clean.select('track_uri_int').distinct()
tracks_distinct = tracks_distinct.selectExpr('cast(track_uri_int as int) track_uri_int')
# tracks_distinct = tracks_distinct.withColumnRenamed("track_uri_int", "repartition_id")





# COMMAND ----------

spark.conf.set("spark.databricks.queryWatchdog.enabled", False)
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
# spark.conf.set("spark.databricks.queryWatchdog.outputRatioThreshold", '50000')

# COMMAND ----------

playlists_distinct.cache()
playlists_distinct.count()

cross_joined = broadcast(playlists_distinct).crossJoin(tracks_distinct.repartition(400, tracks_distinct["track_uri_int"]))#.join(df_clean, ['pid', 'track_uri_int'], 'left').fillna(0)
cross_joined.explain()
# cross_joined.orderBy('pid').show()

# COMMAND ----------

cross_joined.show()

# COMMAND ----------

# cross_joined.rdd.getNumPartitions()
# cross_joined = cross_joined.repartition(200)
cross_joined.write.parquet("/dbfs/FileStore/spotify_million_playlist_raw_data/01_playlist_tracks_cross_joined_step1of2.parquet")

# COMMAND ----------

cross_joined_w_exist = cross_joined.join(df_clean, ['pid', 'track_uri_int'], 'left').fillna(0)

# COMMAND ----------

# cross_joined_w_exist.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 3.1 ROEM Function for Evaluating Our Recommender System

# COMMAND ----------

# Train, Test Split
(train, test) = cross_joined_w_exist.randomSplit([0.8, 0.2], seed = 123)

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
                              .join(cross_joined, (recommend_10.recommend_pid == cross_joined.pid) & (recommend_10.track_uri_int == cross_joined.track_uri_int), how = 'left')

display(recommend_10_w_track_uri.orderBy('recommend_pid', ascending=True))

# COMMAND ----------


display(df.filter(df.pid == 5))
