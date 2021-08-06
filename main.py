import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
my_model = pickle.load(open('inter_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]

    final_features = np.array([int_features])
    final = my_model.predict(final_features)
    if final == 0:
        final = "c-CS-m"
    elif final == 1:
        final = "c-CS-s"
    elif final == 2:
        final = "c-SC-m"
    elif final == 3:
        final = "c-SC-s"
    elif final == 4:
        final = "t-CS-m"
    elif final == 5:
        final = "t-CS-s"
    elif final == 6:
        final = "t-SC-m"
    elif final == 7:
        final = "t-SC-s"

    return render_template('index.html', prediction_text='Class should be  {}'.format(final))


if __name__ == '__main__':
    app.run()

