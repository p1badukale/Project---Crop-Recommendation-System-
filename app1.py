from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

app = Flask(__name__)
data=pd.read_csv("C:\\Users\\pavan\\Downloads\\Crop_recommendation.csv")
x = data.drop(['label'], axis=1)
# Load your model and other necessary libraries here
x = data.drop(['label'], axis=1)
km =KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(x)
# Lets find out the Results
a = data['label']
y_means = pd.DataFrame(y_means)
z = pd.concat([y_means, a], axis = 1)
z = z.rename(columns = {0: 'cluster'})
y = data['label']
x = data.drop(['label'], axis = 1)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Process the input using your prediction logic (include all 7 features)
        prediction = model.predict([[N, P, K, temperature, humidity, ph, rainfall]])

        # Pass the prediction to the result template
        return render_template('result.html', prediction=prediction[0])


if __name__ == '__main__':
    app.run(debug=True)
