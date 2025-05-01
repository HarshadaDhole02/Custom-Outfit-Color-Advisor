import os
import base64
import pickle
from flask import Flask,render_template,request
import numpy as np
from skin_tone_detector import predict_skin_tone,recommend_colors
import io
from PIL import Image

app = Flask(__name__)

with open("skin_tone_model.pkl","rb") as f:
    model = pickle.load(f)


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    image_data = request.form['image']
    image_data=image_data.split(',')[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))

    image_path = "static/uploads/captured_image.jpg"
    image.save(image_path)

    r,g,b =np.array(image).mean(axis=(0,1))[:3]

    skin_tone,suggested_colors = predict_skin_tone(r,g,b)

    print(f"Predicted Skin Tone: {skin_tone}")
    print(f"Recommended Colors: {suggested_colors}")
    
    return render_template("result.html",tone=skin_tone,colors=suggested_colors,r=int(r),g=int(g),b=int(b))

if __name__ == '__main__':
    app.run(debug=True)