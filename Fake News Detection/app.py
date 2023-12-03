from flask import Flask, render_template, request
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
from bs4 import BeautifulSoup
from newspaper import Article

app = Flask(__name__)

# Load pre-trained models
text_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

image_model = models.resnet50(pretrained=True)
image_model.fc = torch.nn.Linear(in_features=2048, out_features=2)
image_model.eval()

def preprocess_text(text):
    return text_tokenizer(text, padding=True, truncation=True, return_tensors='pt')

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension
    return input_batch

def preprocess_url(url):
    # Download the article content from the URL
    article = Article(url)
    article.download()
    article.parse()

    # Get the text content of the article
    text_content = article.text

    output = text_model(**preprocess_text(text_content)).logits

    return output

def combine_outputs(text_output, image_output):
    combined_output = torch.cat((text_output, image_output), dim=1)
    return combined_output

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_type = request.form['input_type']

        if input_type == 'text':
            input_data = request.form['text_input']
            output = text_model(**preprocess_text(input_data)).logits
        elif input_type == 'url':
            input_data = request.form['url_input']
            output = text_model(**preprocess_text(input_data)).logits
        elif input_type == 'image':
            input_data = request.files['image_input']
            output = image_model(preprocess_image(input_data))

        # Extract class prediction and confidence level
        class_prediction = torch.argmax(output).item()
        confidence_level = torch.nn.functional.softmax(output, dim=1)[0][class_prediction].item()

        # Determine whether the news is real or fake based on the class prediction
        result = "FAKE" if class_prediction == 1 else "REAL"

        # Further processing or return the result as needed
        return render_template('result.html', result=result, confidence=confidence_level)

if __name__ == '__main__':
    app.run(debug=True)