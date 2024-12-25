# preprocess_and_train_dag.py
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime

default_args = {
    'start_date': datetime(2024, 1, 1),
    'retries': 0
}

def preprocess_data_callable():
    """
    Calls the preprocess_data function from preprocess_data.py.
    """
    import sys
    sys.path.append("/opt/airflow/scripts")  # or wherever you store preprocess_data.py
    from preprocess_data import preprocess_data
    preprocess_data()

def train_model_callable():
    """
    Calls the run_training function from train_model.py.
    """
    import sys
    sys.path.append("/opt/airflow/scripts")  # or wherever you store train_model.py
    from train_model import run_training
    run_training()

with DAG(
    dag_id='preprocess_and_train_dag',
    default_args=default_args,
    schedule_interval='@once',  # or whatever schedule you desire
    catchup=False,
) as dag:

    preprocess_data_task = PythonOperator(
        task_id='preprocess_data_task',
        python_callable=preprocess_data_callable
    )

    train_model_task = PythonOperator(
        task_id='train_model_task',
        python_callable=train_model_callable
    )

    # The training depends on the preprocessing
    preprocess_data_task >> train_model_task