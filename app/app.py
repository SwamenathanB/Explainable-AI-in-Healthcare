from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import pickle
import joblib
import numpy as np
from firebase_admin import credentials, firestore, initialize_app, storage
from google.cloud import storage as gcp_storage
import json
import pickle
import xgboost

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "healthcure-9f50b-eaeed22fd8f2.json"

# Initialize Firestore DB
cred = credentials.Certificate('key.json')
default_app = initialize_app(cred, {
    'storageBucket': 'healthcure-9f50b.appspot.com'
})
bucket = storage.bucket()
db = firestore.client()
users_ref = db.collection('users')

app = Flask(__name__)


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = gcp_storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )


diabetes_model = pickle.load(open('models/diabetes_model.sav', 'rb'))
heart_model = pickle.load(open('models/heart_disease_model.dat', 'rb'))
breastcancer_model = joblib.load('models/breast_cancer_model.pkl')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('homepage.html')


@app.route('/breastcancer')
def breast_cancer():
    return render_template('breastcancer.html')


@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')


@app.route('/heartdisease')
def heartdisease():
    return render_template('heartdisease.html')


@app.route('/resultd', methods=['POST'])
def resultd():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        pregnancies = request.form['pregnancies']
        glucose = request.form['glucose']
        bloodpressure = request.form['bloodpressure']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        diabetespedigree = request.form['diabetespedigree']
        age = request.form['age']
        skinthickness = request.form['skin']

        # Insert Data into Firestore DB
        firestore_entry = {}
        firestore_entry['name'] = firstname + " " + lastname
        firestore_entry['email'] = email
        firestore_entry['phone'] = phone
        firestore_entry['gender'] = gender
        firestore_entry['age'] = age
        json_string = json.dumps(firestore_entry)
        json_object = json.loads(json_string)

        try:
            users_ref.document(email).set(json_object)
        except Exception as e:
            print(e)

        pred = diabetes_model.predict(
            [[insulin, bmi, diabetespedigree, age]])
        # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Diabetes test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
        return render_template('resultd.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)


@app.route('/resultbc', methods=['POST'])
def resultbc():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        cpm = request.form['concave_points_mean']
        am = request.form['area_mean']
        rm = request.form['radius_mean']
        pm = request.form['perimeter_mean']
        cm = request.form['concavity_mean']

        # Insert Data into Firestore DB
        firestore_entry = {}
        firestore_entry['name'] = firstname + " " + lastname
        firestore_entry['email'] = email
        firestore_entry['phone'] = phone
        firestore_entry['gender'] = gender
        firestore_entry['age'] = age
        json_string = json.dumps(firestore_entry)
        json_object = json.loads(json_string)

        try:
            users_ref.document(email).set(json_object)
        except Exception as e:
            print(e)

        pred = breastcancer_model.predict(
            np.array([cpm, am, rm, pm, cm]).reshape(1, -1))
        return render_template('resultbc.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)


@app.route('/resulth', methods=['POST'])
def resulth():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        nmv = float(request.form['nmv'])
        tcp = float(request.form['tcp'])
        eia = float(request.form['eia'])
        thal = float(request.form['thal'])
        op = float(request.form['op'])
        mhra = float(request.form['mhra'])
        age = float(request.form['age'])
        print(np.array([nmv, tcp, eia, thal, op, mhra, age]).reshape(1, -1))

        # Insert Data into Firestore DB
        firestore_entry = {}
        firestore_entry['name'] = firstname + " " + lastname
        firestore_entry['email'] = email
        firestore_entry['phone'] = phone
        firestore_entry['gender'] = gender
        firestore_entry['age'] = age
        json_string = json.dumps(firestore_entry)
        json_object = json.loads(json_string)

        try:
            users_ref.document(email).set(json_object)
        except Exception as e:
            print(e)

        pred = heart_model.predict(
            np.array([nmv, tcp, eia, thal, op, mhra, age]).reshape(1, -1))
        if pred == 0:
            prediction = "POSITIVE"
        elif pred == 1:
            prediction = "NEGATIVE"
        #pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Diabetes test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
        return render_template('resulth.html', fn=firstname, ln=lastname, age=age, r=prediction, gender=gender)


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


if __name__ == '__main__':
    app.run(debug=True)
