# ###########################################################################
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2021
#  All rights reserved.
#
# 모델 시뮬레이션
# ###########################################################################

import time, os, random, json, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cmlbootstrap import CMLBootstrap
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType, IntegerType, StringType
import dbldatagen as dg
import dbldatagen.distributions as dist
from dbldatagen import FakerTextFactory, DataGenerator, fakerText
from faker.providers import bank, credit_card, currency
import cmlapi
from src.api import ApiUtility
from sklearn.metrics import classification_report
import cml.data_v1 as cmldata
import cml.metrics_v1 as metrics
import cml.models_v1 as models
import math


class BankDataGen:
    #금융데이터 생성

    def __init__(self, username, dbname, storage, connectionName):
        self.username = username
        self.storage = storage
        self.dbname = dbname
        self.connectionName = connectionName


    def dataGen(self, spark, shuffle_partitions_requested = 5, partitions_requested = 2, data_rows = 200):
        #금융 거래 생성, fraud = 1, 사기

        # setup use of Faker
        FakerTextUS = FakerTextFactory(locale=['en_US'], providers=[bank])

        # partition parameters etc.
        spark.conf.set("spark.sql.shuffle.partitions", shuffle_partitions_requested)

        fakerDataspec = (DataGenerator(spark, rows=data_rows, partitions=partitions_requested)
                    .withColumn("age", "float", minValue=10, maxValue=100, random=True)
                    .withColumn("credit_card_balance", "float", minValue=100, maxValue=30000, random=True)
                    .withColumn("bank_account_balance", "float", minValue=0.01, maxValue=100000, random=True)
                    .withColumn("mortgage_balance", "float", minValue=0.01, maxValue=1000000, random=True)
                    .withColumn("sec_bank_account_balance", "float", minValue=0.01, maxValue=100000, random=True)
                    .withColumn("savings_account_balance", "float", minValue=0.01, maxValue=500000, random=True)
                    .withColumn("sec_savings_account_balance", "float", minValue=0.01, maxValue=500000, random=True)
                    .withColumn("total_est_nworth", "float", minValue=10000, maxValue=500000, random=True)
                    .withColumn("primary_loan_balance", "float", minValue=0.01, maxValue=5000, random=True)
                    .withColumn("secondary_loan_balance", "float", minValue=0.01, maxValue=500000, random=True)
                    .withColumn("uni_loan_balance", "float", minValue=0.01, maxValue=10000, random=True)
                    .withColumn("longitude", "float", minValue=-180, maxValue=180, random=True)
                    .withColumn("latitude", "float", minValue=-90, maxValue=90, random=True)
                    .withColumn("transaction_amount", "float", minValue=0.01, maxValue=30000, random=True)
                    .withColumn("fraud", "integer", minValue=0, maxValue=1, random=True)
                    )
        df = fakerDataspec.build()

        return df


    def createSparkConnection(self):
        #세션 생성
        from pyspark import SparkContext
        SparkContext.setSystemProperty('spark.executor.cores', '2')
        SparkContext.setSystemProperty('spark.executor.memory', '4g')

        import cml.data_v1 as cmldata
        conn = cmldata.get_connection(self.connectionName)
        spark = conn.get_spark_session()

        return spark


    def saveFileToCloud(self, df):
        #클라우드 스토리지에 저장
        df.write.format("csv").mode('overwrite').save(self.storage + "/bank_fraud_demo/" + self.username)


    def createDatabase(self, spark):
        #DB생성
        spark.sql("CREATE DATABASE IF NOT EXISTS {}".format(self.dbname))

        print("SHOW DATABASES LIKE '{}'".format(self.dbname))
        spark.sql("SHOW DATABASES LIKE '{}'".format(self.dbname)).show()


    def createOrReplace(self, df):
        #테이블이 없으면 생성, 있으면 추가
        try:
            df.writeTo("{0}.BANK_TX_XGB_RESULT".format(self.dbname))\
              .using("iceberg").tableProperty("write.format.default", "parquet").append()

        except:
            df.writeTo("{0}.BANK_TX_XGB_RESULT".format(self.dbname))\
                .using("iceberg").tableProperty("write.format.default", "parquet").createOrReplace()


    def validateTable(self, spark):
        #제대로 되었는지 검증
        print("SHOW TABLES FROM '{}'".format(self.dbname))
        spark.sql("SHOW TABLES FROM {}".format(self.dbname)).show()

#---------------------------------------------------
# 배치 데이터 생성 - 실제 비즈니스에서 들어오는 데이터 만들기
#---------------------------------------------------

USERNAME = os.environ["PROJECT_OWNER"]
DBNAME = "MA_MLOps"
STORAGE = "s3a://go01-demo/user"
CONNECTION_NAME = "go01-aw-dl"

# Instantiate BankDataGen class
dg = BankDataGen(USERNAME, DBNAME, STORAGE, CONNECTION_NAME)

# Create CML Spark Connection
spark = dg.createSparkConnection()

# Create Banking Transactions DF - 실제 비즈니스에서 생성된 사기탐지 결과 데이터 포함하여 생성
df = dg.dataGen(spark).toPandas()
df_result = dg.dataGen(spark)

# You can access all models with API V2
client = cmlapi.default_client()

project_id = os.environ["CDSW_PROJECT_ID"]
client.list_models(project_id)

# You can use an APIV2-based utility to access the latest model's metadata. For more, explore the src folder
apiUtil = ApiUtility()

model_name = "XGB_Fraud_SH_7a0ab8f8"

Model_AccessKey = apiUtil.get_latest_deployment_details(model_name=model_name)["model_access_key"]
Deployment_CRN = apiUtil.get_latest_deployment_details(model_name=model_name)["latest_deployment_crn"]

# Get the various Model Endpoint details
HOST = os.getenv("CDSW_API_URL").split(":")[0] + "://" + os.getenv("CDSW_DOMAIN")
model_endpoint = (
    HOST.split("//")[0] + "//modelservice." + HOST.split("//")[1] + "/model"
)

# Create an array of model responses.
response_labels = []

# Run Similation to make 1000 calls to the model with increasing error
percent_counter = 0
percent_max = len(df)

record = '{"dataframe_split":{"columns":["age", "credit_card_balance", "bank_account_balance", "mortgage_balance", "sec_bank_account_balance", "savings_account_balance", "sec_savings_account_balance", "total_est_nworth", "primary_loan_balance", "secondary_loan_balance", "uni_loan_balance", "longitude", "latitude", "transaction_amount"],"data":[[35.5, 20000.5, 3900.5, 14000.5, 2944.5, 3400.5, 12000.5, 29000.5, 1300.5, 15000.5, 10000.5, 2000.5, 90.5, 120.5]]}}'

import random
import numpy as np

user_api_key = "cf832bad2a96927c4522308e11b7ca53bef8e681ba9cdc68c43aa67b659c5d17.60f018956a7fefa65f3a2c341e1c33b10f27ef069c200630dd3d4a2d89fd6044"
predicted_result = []
actual_result = []
uuid_result = []

#예측값, 실제값을 입력하고, 정확도 계산하기
for i in range(df.shape[0]):  
    record = '{"dataframe_split":{"columns":["age", "credit_card_balance", "bank_account_balance", "mortgage_balance", "sec_bank_account_balance", "savings_account_balance", "sec_savings_account_balance", "total_est_nworth", "primary_loan_balance", "secondary_loan_balance", "uni_loan_balance", "longitude", "latitude", "transaction_amount"],\
    "data":""}}'
    p_data = json.loads(record)
    ddf_data = df.iloc[i].drop(['fraud']) #실제값을 제외한 결과     
    p_data['dataframe_split']['data'] = [ddf_data.values.tolist()]
    
    # 모델 호출 - 예측값을 받음
    response = models.call_model(Model_AccessKey, p_data, api_key=user_api_key)
    #print("Response : ", response)
    
    predicted_result.append(response["response"]["prediction"])    
    uuid_result.append(response["response"]["uuid"])
    #실제값
    actual_result.append(df.iloc[i]['fraud'])
    
    #추론하기
    metrics.track_delayed_metrics({"actual_result":df.iloc[i]['fraud']}, response["response"]["uuid"])
    #print("actual_result : ", actual_result)

    if i % 100 == 0:
      start_time_ms = int(math.floor(time.time() * 1000)) 
      
    if i % 100 == 99:
      end_time_ms = int(math.floor(time.time() * 1000))        
      accuracy = classification_report(actual_result,predicted_result,output_dict=True)['accuracy']
      allresult=metrics.track_aggregate_metrics({"accuracy": accuracy}, start_time_ms , end_time_ms, model_deployment_crn=Deployment_CRN)
      print("adding accuracy measure of " ,accuracy)
 