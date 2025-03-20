from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    "draft_model_server_dag",
    start_date=datetime(2025, 3, 18),
    schedule_interval="@daily",
    catchup=False,
) as dag:
    # Start FastAPI server
    start_fastapi = BashOperator(
        task_id="start_fastapi",
        bash_command="uvicorn app:app --host 0.0.0.0 --port 8000 &",
    )

    # Start Celery worker
    start_celery = BashOperator(
        task_id="start_celery",
        bash_command="celery -A app.celery_app worker --loglevel=info &",
    )

    start_fastapi >> start_celery