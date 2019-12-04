from flask import Flask,jsonify,request
import torch
from transformers import BertTokenizer, BertModel
from predict import predict_sentiment, init_token_idx, eos_token_idx
import json
from BERTGRUSentiment import BERTGRUSentiment

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained('bert-base-uncased')
model = torch.load("../notebooks/yelp_entire_model.pt", map_location=torch.device('cpu'))

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Return json response with sentiment 
    for a client sentence from PyTorch
    Transformer + GRU model trained on 
    Yelp Reviews dataset
    """
    if request.method== "POST":
        sentence = request.get_data()
        sentence = str(sentence)
        sentiment = predict_sentiment(model, tokenizer, sentence)
        return jsonify({'sentiment': sentiment})

if __name__ == "__main__":
    app.run(debug=True)
