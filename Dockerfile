FROM python:3.7-buster

RUN mkdir my-model
ENV MODEL_PATH=my-model/catboost_model.cbm

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt 

ENV AIP_STORAGE_URI=gs://model-8990/heart
ENV AIP_BUCKET_ID=model-8990
ENV AIP_MODEL_URI=heart/model.joblib
ENV AIP_HEALTH_ROUTE=/ping
ENV AIP_PREDICT_ROUTE=/predict
ENV AIP_HTTP_PORT=8080

COPY app.py ./app.py

# Expose port 8080
EXPOSE 8080

CMD flask run --host=0.0.0.0 --port=8080 