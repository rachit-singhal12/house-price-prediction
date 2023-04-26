from flask import Flask, render_template,request,redirect,url_for
import pickle
import numpy as np
import pandas as pd
app = Flask(__name__)

model= pickle.load(open('python/models/model.pkl','rb'))

@app.route('/')
def index():
    return render_template('UI.html')

@app.route('/predict',methods = ['POST'])
def predict():
    if "subform" in request.form:
        bedrooms = request.form.get('bedrooms', 0)
        bathrooms = request.form.get('bathroom', 0)
        sqft_living = request.form.get('sqft_living', 0)
        sqft_lot = request.form.get('sqft_lot', 0)
        floors = request.form.get('floors', 0)
        waterfront = request.form.get('waterfront', 0)
        view = request.form.get('view', 0)
        condition = request.form.get('condition', 0)
        sqft_above = request.form.get('sqft_above', 0)
        sqft_basement = request.form.get('sqft_basement', 0)
        yr_built = request.form.get('yr_built', 0)
        yr_renovated = request.form.get('yr_renovated', 0)

        int_features = [[bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, sqft_above, sqft_basement, yr_built, yr_renovated]]
        f = pd.DataFrame(int_features,columns=['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','sqft_above','sqft_basement','yr_built','yr_renovated'])

        f['bedrooms'] = f['bedrooms'].astype(float)
        f['bathrooms'] = f['bathrooms'].astype(float)
        f['sqft_living'] = f['sqft_living'].astype(float)
        f['sqft_lot'] = f['sqft_lot'].astype(float)
        f['floors'] = f['floors'].astype(float)
        f['waterfront'] = f['waterfront'].astype(float)
        f['view'] = f['view'].astype(float)
        f['condition'] = f['condition'].astype(float)
        f['sqft_above'] = f['sqft_above'].astype(float)
        f['yr_built'] = f['yr_built'].astype(float)
        f['yr_renovated'] = f['yr_renovated'].astype(float)
        f['sqft_basement'] = f['sqft_basement'].astype(float)

        pred = model.predict(f)
        output = round(pred[0]*1000000)

        return render_template('UI.html',data = output)

if __name__ == '__main__':
    app.run(debug=True)