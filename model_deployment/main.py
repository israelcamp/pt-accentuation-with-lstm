import json

from flask import request, Flask
from flask_cors import CORS

from model import ModelHandler

MODEL_HANDLER = ModelHandler()

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['POST'])
def predict():
    print(request)
    original_text = request.form['text']
    predicted_text = MODEL_HANDLER(original_text)
    print(predicted_text)
    return {
        'status': 200,
        'predicted_text': predicted_text
    }


if __name__ == '__main__':
    app.run()
#     model_handler = ModelHandler()
#     txt = model_handler.pred_text_from_text(
#         )
#     print(txt)
