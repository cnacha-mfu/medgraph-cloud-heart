import joblib
import flask
import os
import pandas as pd
import json
from google.cloud import storage
from tempfile import TemporaryFile
import shap

app = flask.Flask(__name__)
port = int(os.getenv("AIP_HTTP_PORT", 8080))

storage_client = storage.Client()
bucket_name=os.environ['AIP_BUCKET_ID']
model_bucket=os.environ['AIP_MODEL_URI']

bucket = storage_client.get_bucket(bucket_name)
#select bucket file
blob = bucket.blob(model_bucket)
with TemporaryFile() as temp_file:
    #download blob into temp file
    blob.download_to_file(temp_file)
    temp_file.seek(0)
    #load into joblib
    model=joblib.load(temp_file)
    explainer = shap.TreeExplainer(model)

#model = joblib.load("model.joblib")

@app.route(os.environ['AIP_HEALTH_ROUTE'], methods=['GET'])
def health_check():
   return {"status": "healthy"}

@app.route(os.environ['AIP_PREDICT_ROUTE'], methods=['POST'])
def predict():
    features = flask.request.get_json(force=True)
    df = pd.DataFrame.from_dict(features['instances'])
    prediction = model.predict(df)
    shap_values = explainer.shap_values(df)
    #print(shap_values)
    explaination = []
    i = 0
    feature_names = df.columns
    for pred in prediction:
        ind_shape_value = shap_values[i].tolist()
        indv_expl = {}
        j=0
        for fn in feature_names:
            indv_expl[fn] = ind_shape_value[j]
            j = j+1
        explaination.append(indv_expl)
        i = i+1
    
    #print(explaination)
    
    return flask.jsonify({"predictions": {"result":prediction.tolist(),"explainations":explaination}})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)

