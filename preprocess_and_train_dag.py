from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'start_date': datetime(2024, 12, 25),
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
    Calls the run_training function from train_model.py
    (e.g. using RandomForest + GridSearchCV).
    """
    import sys
    sys.path.append("/opt/airflow/scripts")  # or wherever you store train_model.py
    from train_model import run_training  # now uses GridSearchCV
    run_training()

def train_model_xgb_callable():
    """
    Calls the run_training_xgb_threshold function from train_model_xgb_threshold.py
    (XGB + threshold + multi-metric approach).
    """
    import sys
    sys.path.append("/opt/airflow/scripts")  # path for train_model_xgb_threshold.py
    from train_model_xgb_threshold import run_training_xgb_threshold
    run_training_xgb_threshold()

with DAG(
    dag_id='preprocess_and_train_dag',
    default_args=default_args,
    schedule_interval=None,  # or set a cron or '@once' if desired
    catchup=False,
) as dag:

    preprocess_data_task = PythonOperator(
        task_id='preprocess_data_task',
        python_callable=preprocess_data_callable
    )

    train_model_task = PythonOperator(
        task_id='train_model_task',
        python_callable=train_model_callable, 
        execution_timeout=timedelta(hours=2)
    )

    train_model_xgb_threshold_task = PythonOperator(
        task_id='train_model_xgb_threshold_task',
        python_callable=train_model_xgb_callable,
        execution_timeout=timedelta(hours=2)
    )

    # The training depends on the preprocessing
    # -> Once data is preprocessed, it can trigger both RF training (train_model_task)
    #    and XGB threshold training (train_model_xgb_threshold_task)
    preprocess_data_task >> [train_model_task, train_model_xgb_threshold_task]