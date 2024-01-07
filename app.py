from flask import Flask, request, jsonify,make_response
import joblib
import spamclassify
from sklearn.preprocessing import MinMaxScaler
from flask_cors import CORS, cross_origin
from flask import Response
from flask.helpers import send_from_directory
app = Flask(__name__,static_folder='sp/build',static_url_path='')
CORS(app,origins='*')


#CORS(app, resources={r"/*": {"origins": "*", "methods": ["POST"]}})
# Load the model and vectorizer
CORS(app)

""" def home():
    response = jsonify({'message': 'Hello, Flask is running!'})
    response
    return response """


@app.route('/predict',methods =['POST'] )
def predict():
    # response = jsonify({"prediction": "po"})
    data = request.get_json(force=True)
    text = data['text']
    #turn the message into vector 
    classifier = joblib.load('spam_classifier_model.joblib')
    ovect = spamclassify.message_to_count_vector2(text)
    print(ovect)
    print(ovect.shape)
    #turn ovect a number from 0-1   
    #fitted = MinMaxScaler.transform(ovect)
    
    #print(fitted)
    # Make predictions
    prediction = classifier.predict(ovect)
    print (prediction[0])
    response = jsonify(int(prediction[0]))
    #response= jsonify({"pee":"pee runs"})
    response
    return response

@app.route('/')
@cross_origin()
def serve():
    return send_from_directory(app.static_folder,'index.html')

      
       
       

    


if __name__ == '__main__':
    app.run()