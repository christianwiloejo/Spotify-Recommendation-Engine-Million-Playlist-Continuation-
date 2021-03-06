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
from numpy import *
from pyspark.sql.functions import *
from functools import reduce
from pyspark.sql import DataFrame
from pprint import pprint
import pandas as pd
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql.types import IntegerType
from pyspark.mllib.linalg import SparseVector, SparseMatrix

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
# MAGIC # 1. Read Parquet File and Reduce Tracks
# MAGIC As we can see, there are 1 million playlists, which contains 2,262,292 distinct tracks. When performing cartesian join for our ALS model, this will produce around 2 trillion rows. To save time and resources, we need to cut the tracks data down. 

# COMMAND ----------

raw_df = spark.read.parquet("/dbfs/FileStore/spotify_million_playlist_raw_data/all_playlists_tracks_parsed.parquet")
raw_df.show(5)

# COMMAND ----------

print(f"There is {raw_df.select('pid').distinct().count()} distinct playlists, and {raw_df.select('track_uri').distinct().count()} distinct tracks.")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 1.1 Limiting number of distinct songs for faster processing

# COMMAND ----------

count_pid_by_song = raw_df.groupBy('track_uri').count()
count_pid_by_song.orderBy("count", ascending=False)
track_uri_more_100_pids = count_pid_by_song.where("count > 5000").select('track_uri')
print(f"Now, there are {track_uri_more_100_pids.count()} distinct tracks. This is easier to process.")

# COMMAND ----------

raw_df = raw_df.join(track_uri_more_100_pids, on = 'track_uri')
raw_df.count()

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

# raw_df = raw_df.filter(raw_df['pid'] < 1000).select('pid', 'track_uri', 'song_exist_in_playlist')

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
playlists_distinct = playlists_distinct.repartition(400, playlists_distinct["pid"])

# Extract distinct tracks
tracks_distinct = df_clean.select('track_uri_int').distinct()
tracks_distinct = tracks_distinct.selectExpr('cast(track_uri_int as int) track_uri_int')

# COMMAND ----------

cross_joined = playlists_distinct.crossJoin(broadcast(tracks_distinct.repartition(400, tracks_distinct["track_uri_int"])))

df_clean = df_clean.repartition(400)
cross_joined_w_exist = cross_joined.join(broadcast(df_clean.repartition(400)), ['pid', 'track_uri_int'], 'left').fillna(0)

# COMMAND ----------

# (cross_joined_w_exist.write.format("parquet")
#       .mode("overwrite")
#       .save("/dbfs/FileStore/spotify_million_playlist_raw_data/playlist_tracks_cross_joined.parquet"))

# COMMAND ----------

# MAGIC %md 
# MAGIC # ALS MODEL

# COMMAND ----------

cross_joined_w_exist = spark.read.parquet("/dbfs/FileStore/spotify_million_playlist_raw_data/playlist_tracks_cross_joined.parquet")
cross_joined_w_exist = cross_joined_w_exist.repartition(100)
cross_joined_w_exist.rdd.getNumPartitions()


# COMMAND ----------

# Train, Test Split
(train, test) = cross_joined_w_exist.randomSplit([0.8, 0.2], seed = 123)

# COMMAND ----------

// %scala
// spark.conf.set("spark.sql.shuffle.partitions",100)
// spark.conf.set("spark.default.parallelism",100)
// spark.conf.set("spark.sql.autoBroadcastJoinThreshold",-1)

# COMMAND ----------

# Set checkpoint
# sc.setCheckpointDir('/checkpoint/')
# Build ALS Model
als = ALS(
  userCol = "pid",
  itemCol = "track_uri_int",
  ratingCol = "song_exist_in_playlist", 
  rank = 25, 
  maxIter = 10, 
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

model.write().overwrite().save('/model/spotify_03_main_analysis_model')

# COMMAND ----------

model = ALSModel.load('/model/spotify_03_main_analysis_model')

# COMMAND ----------

recommend_5 = model.recommendForAllUsers(5)\
  .selectExpr("pid", "explode(recommendations) as recommendation_val_score")\
  .select("pid", 'recommendation_val_score.*')\
  .withColumnRenamed("pid", "recommend_pid")
recommend_5.show(15)

# COMMAND ----------

df = df.withColumn('track_uri_int', df["track_uri_int"].cast("int"))
df = df.repartition("track_uri_int")#.cache()
# df.count()
recommend_5 = recommend_5.repartition("track_uri_int")

# COMMAND ----------

track_info = df.select('track_uri', 'track_uri_int', 'track_name', 'artist_name', 'album_name').distinct()

recommend_5_w_track_uri = (recommend_5
                              .join(track_info, on = 'track_uri_int', how = 'left')).cache()

# COMMAND ----------

recommend_5_w_track_uri.orderBy('recommend_pid').show(1000, truncate = False)

# COMMAND ----------

display(df.filter(df.pid == 209))

# COMMAND ----------


