# Fake News Detection App

## Overview

This Flask application is designed for detecting fake news using two different models: one for text-based inputs and another for image-based inputs. The app uses pre-trained models, including BERT for text classification and a ResNet model for image classification.

## Setup Instructions

Follow these steps to set up and run the Fake News Detection App.

### Prerequisites

- Python 3.6 or later
- Pip (Python package installer)
- Flask
- Transformers (Hugging Face library for natural language processing models)
- Torch (PyTorch library)
- Torchvision (PyTorch computer vision library)
- Pillow (Python Imaging Library)
- Requests (HTTP library for sending requests)
- BeautifulSoup (Library for pulling data out of HTML and XML files)
- Newspaper3k (Library for article scraping and parsing)

### Installation

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/your-username/fake-news-detection-app.git
    cd fake-news-detection-app
    ```

2. Create a virtual environment (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Running the App

1. Start the Flask app:

    ```bash
    python app.py
    ```

2. Open your web browser and go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/).

## Usage

The Fake News Detection App provides a simple web interface. Users can input text, a URL, or an image to determine whether the news is fake or real. The app leverages pre-trained models to make predictions based on the provided input.

## Folder Structure

- **templates**: HTML templates for rendering web pages.
- **static**: Static files such as stylesheets.
- **app.py**: The main Flask application file.
- **requirements.txt**: List of Python packages required for the app.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)
- [Flask](https://flask.palletsprojects.com/)

## Watch the Demo

Check out the Fake News Detection Demo on [YouTube](https://youtu.be/iJTOrjOy2as)!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.