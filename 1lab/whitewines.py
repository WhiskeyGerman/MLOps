import os
from datetime import datetime, timedelta

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator

from train_model_whitewines import train


def download_data() -> None:
    url = (
        "https://raw.githubusercontent.com/stedy/"
        "Machine-Learning-with-R-datasets/refs/heads/master/whitewines.csv"
    )
    df = pd.read_csv(url, delimiter=",")
    df.to_csv("whitewines.csv", index=False)
    print(f"Downloaded data shape: {df.shape}")


def clear_data() -> bool:
    if not os.path.exists("whitewines.csv"):
        raise FileNotFoundError("File whitewines.csv not found.")

    df = pd.read_csv("whitewines.csv")

    df = df.drop_duplicates()

    questionable_sugar = df[df["residual sugar"] > 50]
    df = df.drop(questionable_sugar.index)

    questionable_sulfur = df[df["free sulfur dioxide"] > 200]
    df = df.drop(questionable_sulfur.index)

    df = df.reset_index(drop=True)
    df.to_csv("df_clear.csv", index=False)
    print(f"Cleaned data shape: {df.shape}")

    return True


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag_wines = DAG(
    dag_id="whitewines_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 2, 3),
    schedule=timedelta(hours=1),
    max_active_runs=1,
    max_active_tasks=3,
    catchup=False
)

download_task = PythonOperator(
    task_id="download_wines_data",
    python_callable=download_data,
    dag=dag_wines,
)

clear_task = PythonOperator(
    task_id="clear_wines_data",
    python_callable=clear_data,
    dag=dag_wines,
)

train_task = PythonOperator(
    task_id="train_wines_model",
    python_callable=train,
    dag=dag_wines,
)

download_task >> clear_task >> train_task
