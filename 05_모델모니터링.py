# ###########################################################################
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2024
#  All rights reserved.
#
#  모델 모니터링
# ###########################################################################
import time, os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from cmlbootstrap import CMLBootstrap
import seaborn as sns
import sqlite3
import cmlapi
import cml.metrics_v1 as metrics
import cml.models_v1 as models
from src.api import ApiUtility

# You can access all models with API V2
client = cmlapi.default_client()

project_id = os.environ["CDSW_PROJECT_ID"]
client.list_models(project_id)

# You can use an APIV2-based utility to access the latest model's metadata. For more, explore the src folder
apiUtil = ApiUtility()

model_name = "XGB_Fraud_SH_7a0ab8f8"

Model_CRN = apiUtil.get_latest_deployment_details(model_name=model_name)["model_crn"]
Deployment_CRN = apiUtil.get_latest_deployment_details(model_name=model_name)["latest_deployment_crn"]

# Get the various Model Endpoint details
HOST = os.getenv("CDSW_API_URL").split(":")[0] + "://" + os.getenv("CDSW_DOMAIN")
model_endpoint = (
    HOST.split("//")[0] + "//modelservice." + HOST.split("//")[1] + "/model"
)


# Read in the model metrics dict
model_metrics = metrics.read_metrics(
    model_crn=Model_CRN, model_deployment_crn=Deployment_CRN
)

# This is a handy way to unravel the dict into a big pandas dataframe
metrics_df = pd.json_normalize(model_metrics["metrics"])
metrics_df.tail().T

# This is a handy way to unravel the dict into a big pandas dataframe

# Write the data to SQL lite for visualization
if not (os.path.exists("model_metrics.db")):
    conn = sqlite3.connect("model_metrics.db")
    metrics_df.to_sql(name="model_metrics", con=conn)

# Do some conversions & calculations on the raw metrics
metrics_df["startTimeStampMs"] = pd.to_datetime(
    metrics_df["startTimeStampMs"], unit="ms"
)
metrics_df["endTimeStampMs"] = pd.to_datetime(metrics_df["endTimeStampMs"], unit="ms")
metrics_df["processing_time"] = (
    metrics_df["endTimeStampMs"] - metrics_df["startTimeStampMs"]
).dt.microseconds * 1000

# Create plots for different tracked metrics
sns.set_style("whitegrid")
sns.despine(left=True, bottom=True)

# Plot metrics.probability
prob_metrics = metrics_df.dropna(subset=["metrics.accuracy"]).sort_values(
    "startTimeStampMs"
)
sns.lineplot(
    x=range(len(prob_metrics)), y="metrics.accuracy", data=prob_metrics, color="grey"
)

# Plot processing time
time_metrics = metrics_df.dropna(subset=["processing_time"]).sort_values(
    "startTimeStampMs"
)
sns.lineplot(
    x=range(len(prob_metrics)), y="processing_time", data=prob_metrics, color="grey"
)

# Plot model accuracy drift over the simulated time period
agg_metrics = metrics_df.dropna(subset=["metrics.accuracy"]).sort_values(
    "startTimeStampMs"
)
sns.barplot(
    x=list(range(1, len(agg_metrics) + 1)),
    y="metrics.accuracy",
    color="grey",
    data=agg_metrics,
)
