from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier

# train and test split
train, test = df_labeled.randomSplit([0.8, 0.2], seed=2020)


# Create Confusion Matrix
def get_evaluation_result(predictions):
  evaluator = BinaryClassificationEvaluator(
      labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
  AUC = evaluator.evaluate(predictions)

  TP = predictions[(predictions["label"] == 1) & (predictions["prediction"] == 1.0)].count()
  FP = predictions[(predictions["label"] == 0) & (predictions["prediction"] == 1.0)].count()
  TN = predictions[(predictions["label"] == 0) & (predictions["prediction"] == 0.0)].count()
  FN = predictions[(predictions["label"] == 1) & (predictions["prediction"] == 0.0)].count()

  accuracy = (TP + TN)*1.0 / (TP + FP + TN + FN)
  precision = TP*1.0 / (TP + FP)
  recall = TP*1.0 / (TP + FN)


  print ("True Positives:", TP)
  print ("False Positives:", FP)
  print ("True Negatives:", TN)
  print ("False Negatives:", FN)
  print ("Test Accuracy:", accuracy)
  print ("Test Precision:", precision)
  print ("Test Recall:", recall)
  print ("Test AUC of ROC:", AUC)

# Logistic Regression
lr = LogisticRegression(featuresCol = "WordToVector", labelCol = "label", maxIter=10, regParam=0.1, elasticNetParam=0.8)
lrModel = lr.fit(train)
# make the defalut prediction based on the hyperparameters without tuning.
prediction = lrModel.transform(test)
prediction.show(10)
print("Prediction result summary for Logistic Regression Model:  ")
get_evaluation_result(prediction)

# Random Forest
# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="label", featuresCol="WordToVector", numTrees=15)
# Train model.  This also runs the indexers.
rf_model = rf.fit(train)
# Make predictions.
predictions = rf_model.transform(test)
# Select example rows to display.
predictions.show(10)
print("Prediction result summary for Random Forest Model:  ")
get_evaluation_result(predictions)

# Gradient-Boosted Tree 
gbt = GBTClassifier(labelCol="label", featuresCol="WordToVector", maxIter=10)
gbtModel = gbt.fit(train)
predictions = gbtModel.transform(test)
predictions.show(10)
print("Prediction result summary for Gradient-Boosted Tree Model:  ")
get_evaluation_result(predictions)
