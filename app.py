from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

model = joblib.load('model/model.pkl')

@app.route('/')
def home():
    return '''
    <h2>House Price Prediction</h2>
    <form action="/predict_ui" method="post">
        <input type="number" name="size" placeholder="Enter size" required>
        <button type="submit">Predict</button>
    </form>
    '''

@app.route('/predict_ui', methods=['POST'])
def predict_ui():
    size = int(request.form['size'])
    prediction = model.predict([[size]])

    return f"<h3>Predicted Price: {prediction[0]}</h3>"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    size = data['size']
    prediction = model.predict([[size]])

    return jsonify({"predicted_price": float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)