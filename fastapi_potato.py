from fastapi import FastAPI
from fastapi import  File, UploadFile
import uvicorn
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf

app=FastAPI()
class_names=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
model=tf.keras.models.load_model('final_model.keras')

def read_image(data):
    image = np.array(Image.open(BytesIO(data)))
    return image
@app.get("/ping")
async def ping():
    return "hi"


@app.post("/predict")
async def predict(file: UploadFile):
    bn= await file.read()
    arr=read_image(bn)
    img_array = tf.expand_dims(arr, 0)

    predictions = model.predict(img_array)
    confidence = round(100 * (np.max(predictions[0])), 2)
    predicted_class = class_names[np.argmax(predictions[0])]
    return {'class':predicted_class, 'confidence':confidence}


if __name__== "__main__" :
    uvicorn.run(app,host='localhost',port=8000)