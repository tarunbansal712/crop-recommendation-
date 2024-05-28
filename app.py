from flask import Flask,request,render_template
import numpy as np
import pandas 
import sklearn
import pickle
import joblib
# importing model
model = pickle.load(open('model.pkl','rb'))

# creating a flask app
app=Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST','GET'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

      
    values=[N,P,K,temp,humidity,ph,rainfall]
    single_pred = np.array(values).reshape(1, -1)
    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}
    print(type(ph))
    print(type(temp))
    print(type(humidity))
    if float(ph)>0 and float(ph)<=14 and float(temp)<100 and float(humidity)>0:
        joblib.load('crop_app','r')
        model = joblib.load(open('crop_app','rb'))
        arr = [single_pred]
        acc = model.predict(arr[0])
        if acc[0] in crop_dict:
            crop = crop_dict[acc[0]]
        return render_template('index.html', prediction=crop)
    else:
        return "Sorry...  Error in entered values in the form Please check the values and fill it again"

if __name__ == "__main__":
    app.run(debug=True)