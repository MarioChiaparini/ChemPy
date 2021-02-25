import numpy as np
from flask import Flask, request, jsonify, render_template
import _pickle as cPickle

app = Flask(__name__) #Initialize the flask App
model = cPickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    print(output)
    if output == 0:
        return render_template('index.html', prediction_text='Gold')
    elif output == 1:
        return render_template('index.html', prediction_text='Quartz')
    elif output == 2:
        return render_template('index.html', prediction_text='Potassium Dihydrogen')
    elif output == 3:
        return render_template('index.html', prediction_text='Iron')
    elif output == 4:
        return render_template('index.html', prediction_text='Diamond')
    
        
if __name__ == "__main__":
    app.run(debug=True)