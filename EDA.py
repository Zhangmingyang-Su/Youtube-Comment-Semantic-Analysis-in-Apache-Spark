import pyspark.sql.functions as F

# read data
df = spark.read.load("/FileStore/tables/animals_comments_csv-5aaff.gz", format='csv', header = True, inferSchema = True)
df.show(10)

# check datatype
df.dtypes

print("Number of rows:", df.count())

# Number of null value in each columns
print("Number of null values in creator_name:", df.filter(df["creator_name"].isNull()).count())
print('Number of null values in userid: ',df.filter(df['userid'].isNull()).count())
print('Number of null values in comment: ',df.filter(df['comment'].isNull()).count())

# Udf function to drop out rows with no comments, no userid and duplicates
def pre_process(df):
  df_drop = df.filter(df['comment'].isNotNull())
  df_drop = df_drop.filter(df_drop['userid'].isNotNull())
  df_drop = df_drop.dropDuplicates()
  
  print('After dropping, we have',str(df_drop.count()), 'rows in dataframe')
  return df_drop

df_drop = pre_process(df)

#Convert text in comments to lower case.
df_clean = df_drop.withColumn('comment', F.lower(F.col('comment')))

display(df_clean)
