from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from numpy import ones, power as pow

import logging

"""
The core partition of the project included two section
1. Read data preprocessed in ahead section.
2. Implement of distributed l1/2 algorithm.
"""
# set logger object to print logging
logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s -  %(asctime)s - %(filename)s -  %(name)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)

# create sparkSession for ready to read data
spark = SparkSession.builder \
    .appName("InnovationProject-Core") \
    .enableHiveSupport() \
    .getOrCreate()
    # .master("local[*]")

# read data
df = spark.read \
    .format("csv") \
    .options(**{"header": True, "inferSchema": True}) \
    .load("hdfs://hdp-01:9000/InnovationProject/Intermediate/part-00000-a4d9deb7-8b1e-4f9d-abe9-09b3cdeaf807-c000.csv", sep=',')

# get number of rows and columns
nrow = df.count()
ncol = len(df.columns) - 1

col_names = df.columns
feature_names = col_names[1:len(col_names)]

# Aggregate all features into one column named features
vectorAssembler = VectorAssembler() \
    .setInputCols(feature_names) \
    .setOutputCol("features")

# item_df_vectors represent the Training data, the first column of it named “Energy (J)”, Second column named features.
item_df_vectors = vectorAssembler \
    .transform(df) \
    .select("Energy (J)", "features")

# get rdd from above dataframe named item_df_vectors
rdd = item_df_vectors.rdd

# persist above rdd in memory
rdd.persist()

# w represent the optimal variable,
# we initialize it as a vector which all element is one.it cause essentially solve a lasso in the first iteration.
w = ones(ncol)

# Lambda is the hyper parameter of the regularization term.
Lambda = 1e4
# Iteration numbers
ITERATION = 20
# threshold for deciding whether cut off w to zero
cutoff_threshold = 1e-4
# following variable is to let w be not zero
eps = 1e-6

# Implement of l1/2
for i in range(ITERATION):
    for k in range(ncol):
        multiplier = float(w[k] ** (1 / 4))
        points = rdd.map(lambda row: (row[0] * multiplier, row[1] * multiplier))
        threshold = (2 / ncol) * points.map(
            lambda row: (row[0] - (row[1] @ w - row[1][k] * float(w[k]))) * row[1][k]) \
            .reduce(lambda a, b: a + b)
        if threshold < -Lambda:
            w[k] = ncol * (threshold + Lambda) / (
                    2 * pow(points.map(lambda row: row[1][k]).collect(), 2).sum()) + eps
        elif Lambda < threshold:
            w[k] = ncol * (threshold - Lambda) / (
                    2 * pow(points.map(lambda row: row[1][k]).collect(), 2).sum()) + eps
        else:
            w[k] = eps
    with open("/root/res.txt", mode="a") as f:
        f.writelines("ITERATION - {}, w - {}".format(i + 1, w), "\n")
    logger.info("ITERATION - {}, w - {}".format(i + 1, w))

for i in range(ncol):
    if abs(w[i]) < cutoff_threshold:
        w[i] = 0

with open("/root/res.txt", mode="a") as f:
    f.writelines("w - {}".format(w), "\n")
logger.info(w)
