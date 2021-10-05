# Databricks notebook source
# MAGIC %md 
# MAGIC # File Complete Checker (One Time Process)

# COMMAND ----------

def completeUpload(file_path = '/dbfs/FileStore/spotify_million_playlist_raw_data/', count = 1000, checklist = pd.DataFrame({'file_range':np.arange(0,1000000,1000)})):
  directory_files_df = pd.DataFrame({'files_in_dir': os.listdir(file_path)})
  if directory_files_df.shape[0] == count: 
    print('Complete JSON files uploaded to DBFS...')
  else: 
    print('The following files are missing')
    directory_files_df['left_num'] = directory_files_df['files_in_dir'].str.split('mpd_slice_').str[1].str.split('_').str[0].astype('int')
    checklist = pd.DataFrame({'file_range':np.arange(0,1000000,1000).tolist()})
    print(checklist.loc[~checklist['file_range'].isin(directory_files_df['left_num'])])
completeUpload()

# COMMAND ----------

# MAGIC %md 
# MAGIC # Reading, Transforming and Writing JSON Raw Data to Parquet (One Time Process)

# COMMAND ----------

DfAppend = []

path = '/dbfs/FileStore/spotify_million_playlist_raw_data/'
count = 0 
for json_file in os.listdir(path):
  
  json_file_path = 'dbfs:/FileStore/spotify_million_playlist_raw_data/' + json_file
  master_df = (spark.read
      .option("inferSchema", True)
      .option("multiline", True)
      .json(json_file_path))
  print(json_file_path)
  playlists_df = master_df.select(explode("playlists").alias('playlists'))
  
  playlists_tracks_df = playlists_df.select("playlists.pid", "playlists.name", explode("playlists.tracks").alias("tracks"))
  playlists_tracks_df = playlists_tracks_df.withColumnRenamed("playlists.pid", "pid").withColumnRenamed("playlists.name", "playlist_name")
  
  final_df = playlists_tracks_df.select("pid","playlist_name", "tracks.*")
  
  
  DfAppend.append(final_df)
  count += 1
  
  print(f"Completed {count}/{len(os.listdir(path))}")

#Concatenate all dataframes
df = reduce(DataFrame.unionAll, DfAppend)

#Write to parquet
df.write.parquet("/dbfs/FileStore/spotify_million_playlist_raw_data/all_playlists_tracks_parsed.parquet")
