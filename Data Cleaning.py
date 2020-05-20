from pyspark.ml.feature import RegexTokenizer
from pyspark.sql.functions import rand
from pyspark.ml.feature import Word2Vec
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType


## This is an unlabeled dataset and we want to train a clasifier to identify cat and dog owners. Thus first thing to do is to label each comment.
## 1.Label comment when he/she has dogs or cats.
## 2.label comment when he/she don't have a dog or cat.
## 3.Combine 1 and 2 as our training dataset, and rest of the dataset will be the data we predict.
## 4.The strategy to tell if a user own or not own is just using key words (like I have a dog) to tell. Otherwise we can't have better ways and don't have labels.

# find user with preference of dog and cat 
cond = (df_clean["comment"].like("%my dog%") | df_clean["comment"].like("%i have a dog%")\
        | df_clean["comment"].like("%my cat%") | df_clean["comment"].like("%i have a cat%") \
        | df_clean["comment"].like("%my dogs%") | df_clean["comment"].like("%my cats%")\
        | df_clean["comment"].like("%my cat%") | df_clean["comment"].like("%i have dogs%")\
        | df_clean["comment"].like("%i have cats%") | df_clean["comment"].like("%my puppy%")\
        | df_clean["comment"].like("%my kitten%") | df_clean["comment"].like("%i have a puppy%")\
        | df_clean["comment"].like("%i have puppies%"))

df_clean = df_clean.withColumn('dog_cat',  cond)

# find user do not have 
df_clean = df_clean.withColumn('no_pet', ~df_clean["comment"].like("%my%") & ~df_clean["comment"].like("%have%") & ~df_clean["comment"].like("%my dog%") \
                              & ~df_clean["comment"].like("%my cat%"))


df_clean.show(10)


regexTokenizer = RegexTokenizer(inputCol="comment", outputCol="word", pattern="\\W")
df_clean = regexTokenizer.transform(df_clean)
df_clean.show(10)

# for community edition, smalle dataset for testing
df_clean.orderBy(rand(seed=2020)).createOrReplaceTempView("table1")
df_clean = spark.sql("select * from table1 limit 2000000")

# use word2vec get text vector feature.
# Learn a mapping from words to Vectors. (choose higher vectorSize here)
word2Vec = Word2Vec(vectorSize=50, minCount=1, inputCol="word", outputCol="WordToVector")
model = word2Vec.fit(df_clean)

df_model = model.transform(df_clean)
df_model.show(10)

df_pets = df_model.filter(F.col('dog_cat') == True) 
df_no_pets = df_model.filter(F.col('no_pet') ==  True)
print("Number of confirmed user who own dogs or cats: ", df_pets.count())
print("Number of confirmed user who don't have pet's: ", df_no_pets.count())

df_pets.show(10)
df_no_pets.show(10)

# downsampling
df_no_pets.orderBy(rand()).createOrReplaceTempView("table")
Num_Pos_Label = df_model.filter(F.col('dog_cat') == True).count() 
Num_Neg_Label = df_model.filter(F.col('no_pet') ==  True).count()

#Q1 = spark.sql("SELECT col1 from table where col2>500 limit {}, 1".format(q25))
#pass variable to sql
df_no_pets_down = spark.sql("select * from table where limit {}".format(Num_Pos_Label*2))

print('Now after balancing the lables, we have ')   
print('Positive label: ', Num_Pos_Label)
print('Negtive label: ', df_no_pets_down.count())

def get_label(df_pets,df_no_pets_down):
  df_labeled = df_pets.select('dog_cat','WordToVector').union(df_no_pets_down.select('dog_cat','WordToVector'))
  return df_labeled

df_labeled = get_label(df_pets,df_no_pets_down)
df_labeled.show(10)

#convert Boolean value to 1 and 0's

def multiple(x):
  return int(x*1)
udf_boolToInt= udf(lambda z: multiple(z),IntegerType())
df_labeled = df_labeled.withColumn('label',udf_boolToInt('dog_cat'))
df_labeled.show(10)


