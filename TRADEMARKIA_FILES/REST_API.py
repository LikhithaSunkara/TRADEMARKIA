from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input description from the POST request
    description = request.json['description']

    # Preprocess the input description, tokenize it, and encode it using the trained BERT model
    processed_description = preprocess(description)
    tokenized_description = tokenize(processed_description)
    encoded_description = encode(tokenized_description)

    # Feed the BERT-encoded representation to the trained classification model to predict the class
    output = model(encoded_description)
    predicted_class = get_predicted_class(output)

    # Return the predicted class as a response
    response = {
        'predicted_class': predicted_class
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run()