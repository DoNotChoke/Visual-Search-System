FROM apache/airflow:3.1.5

USER airflow

RUN pip install uv --upgrade


RUN uv pip install \
    "apache-airflow==3.1.5" \
    "apache-airflow-providers-docker" \
    "torch" \
    "torchvision" \
    "feast" \
    "pymilvus"
