# Databricks notebook source
# MAGIC %md
# MAGIC ## Exploratory data analysis

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Display the contents of our object store (S3 bucket):

# COMMAND ----------

raw_df = spark.read.load("s3://databricks-e2demofieldengwest/b169b504-4c54-49f2-bc3a-adf4b128f36d/tables/d429e556-ad55-42ed-a622-956dc5cd959f")

# COMMAND ----------

display(raw_df)

# COMMAND ----------

display(raw_df.drop("ra", "dec").describe())

# COMMAND ----------

silver_df = raw_df.na.replace(0, None)
silver_df = raw_df.na.replace(-9999.0, None)

# COMMAND ----------

silver_df = silver_df.dropna(how='any')
silver_df.count()

# COMMAND ----------

display(silver_df.drop("ra", "dec").describe())

# COMMAND ----------

raw_df.count()

# COMMAND ----------

display(silver_df.drop("ra", "dec").summary())

# COMMAND ----------

from pyspark.sql.functions import round, col

filtered_df = raw_df.select(round("petromag_r", 1).alias("petromag_r"), "class").filter(raw_df["class"] != "QSO")

display(filtered_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature selection

# COMMAND ----------

# From the article: "Color parameters include all types of magnitudes: fiber, Petrosian, model and PSF"
# Generate new derived columns for subsequent inclusion in the "features" matrix

# Fiber
silver_df = silver_df.withColumn("fibermag_u-fibermag_g", silver_df["fibermag_u"] - silver_df["fibermag_g"])
silver_df = silver_df.withColumn("fibermag_g-fibermag_r", silver_df["fibermag_g"] - silver_df["fibermag_r"])
silver_df = silver_df.withColumn("fibermag_r-fibermag_i", silver_df["fibermag_r"] - silver_df["fibermag_i"])
silver_df = silver_df.withColumn("fibermag_i-fibermag_z", silver_df["fibermag_i"] - silver_df["fibermag_z"])

# Petrosian
silver_df = silver_df.withColumn("petromag_u-petromag_g", silver_df["petromag_u"] - silver_df["petromag_g"])
silver_df = silver_df.withColumn("petromag_g-petromag_r", silver_df["petromag_g"] - silver_df["petromag_r"])
silver_df = silver_df.withColumn("petromag_r-petromag_i", silver_df["petromag_r"] - silver_df["petromag_i"])
silver_df = silver_df.withColumn("petromag_i-petromag_z", silver_df["petromag_i"] - silver_df["petromag_z"])

# Model
silver_df = silver_df.withColumn("modelmag_u-modelmag_g", silver_df["modelmag_u"] - silver_df["modelmag_g"])
silver_df = silver_df.withColumn("modelmag_g-modelmag_r", silver_df["modelmag_g"] - silver_df["modelmag_r"])
silver_df = silver_df.withColumn("modelmag_r-modelmag_i", silver_df["modelmag_r"] - silver_df["modelmag_i"])
silver_df = silver_df.withColumn("modelmag_i-modelmag_z", silver_df["modelmag_i"] - silver_df["modelmag_z"])

# Point Spread Function (PSF)
silver_df = silver_df.withColumn("psfmag_u-psfmag_g", silver_df["psfmag_u"] - silver_df["psfmag_g"])
silver_df = silver_df.withColumn("psfmag_g-psfmag_r", silver_df["psfmag_g"] - silver_df["psfmag_r"])
silver_df = silver_df.withColumn("psfmag_r-psfmag_i", silver_df["psfmag_r"] - silver_df["psfmag_i"])
silver_df = silver_df.withColumn("psfmag_i-psfmag_z", silver_df["psfmag_i"] - silver_df["psfmag_z"])

# COMMAND ----------

# Create a list of features to be used for model training
# Include the "class" column as the predictor
feature_list = silver_df.drop("ra", "dec").columns

feature_list

# COMMAND ----------

base_df = silver_df.select(feature_list).filter('class != "QSO"')
base_df.cache().count()
display(base_df)

# COMMAND ----------

star_galaxy_silver_df.select("class").distinct().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Boosted decision tree

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

categorical_cols = [field for (field, dataType) in base_df.dtypes if dataType == "string"]
index_output_cols = [x + "Index" for x in categorical_cols]
string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")

# COMMAND ----------

ohe_df = string_indexer.fit(base_df).transform(base_df)
display(ohe_df)

# COMMAND ----------

from pyspark.ml. feature import VectorAssembler

# Filter for just numeric columns
numeric_cols = [field for (field, dataType) in ohe_df.dtypes if ((dataType == "double") & (field != "classIndex"))]
# Combine output of StringIndexer defined above and numeric columns
assembler_inputs = numeric_cols

vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

# COMMAND ----------

display(output)

# COMMAND ----------

train_df, test_df = output.randomSplit([.8, .2], seed=42)

# COMMAND ----------

display(train_df)

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier

classifier = DecisionTreeClassifier(seed = 42, labelCol="classIndex",
                                    featuresCol="features",
                                    predictionCol="prediction")

# COMMAND ----------

model = classifier.fit(train_df)


# COMMAND ----------

predictions = model.transform(train_df)


# COMMAND ----------

display(predictions.select("class","classIndex", "prediction"))

# COMMAND ----------

output = vec_assembler.transform(ohe_df)

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol="classIndex",
                                        predictionCol="prediction")

evaluator.setMetricName("accuracy").evaluate(predictions)
evaluator.setMetricName("f1").evaluate(predictions)

# COMMAND ----------

from pyspark.sql.functions import udf

def calc_impurity(df):
  for val in df:
    if (df.classIndex == df.predictions) & (df.classIndex == 0):
      num_galaxies += 1
    elif (df.classIndex == df.predictions) & (predictions.classIndex == 1):
      num_stars += 1
  return num_galaxies, num_stars
  

# COMMAND ----------

calc_impurity(predictions)

# COMMAND ----------


