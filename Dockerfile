FROM python:3.9-slim
WORKDIR /app

ENV ROOT_FOLDER_PATH "/app"
ENV MODEL_DIR_PATH "/app/training/model_files"
COPY ./requirements/requirements-api.txt requirements/requirements-api.txt
COPY ./app app
COPY ./training/model_files training/model_files
RUN pip install --upgrade -r /app/requirements/requirements-api.txt


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

