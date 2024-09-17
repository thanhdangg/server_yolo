from flask import Flask, jsonify
import torch
import firebase_admin
from firebase_admin import db, credentials, storage
import cv2
import numpy as np
import json
from PIL import Image
import os


app = Flask(__name__)

cred = credentials.Certificate('yolo-c6728-firebase-adminsdk-ul5x4-02d7ba17d8.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'yolo-c6728.appspot.com',
    'databaseURL': 'https://yolo-c6728-default-rtdb.firebaseio.com' 
})


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

def listener(event):
    try:
        if isinstance(event.data, dict):
            file_name = event.data.get('fileName') 
            if file_name:
                process_image(file_name)
        else:
            print(f'Unexpected data format: {event.data}')
    except UnicodeDecodeError:
        print("Error: Could not decode data from Firebase")
    except Exception as e:
        print(f"Error processing event: {e}")


def download_image(file_name):
    bucket = storage.bucket()
    blob = bucket.blob(f"input/{file_name}")  

    local_directory = "./input"
    if not os.path.exists(local_directory):
        os.makedirs(local_directory)
    
    local_path = os.path.join(local_directory, file_name)  
    blob.download_to_filename(local_path)
    
    print(f"Downloaded {file_name} to {local_path}")
    return local_path

def upload_image(file_name, processed_image_path):
    bucket = storage.bucket()
    blob = bucket.blob(f"output/{file_name}")
    blob.upload_from_filename(processed_image_path)
    
    blob.make_public()
    
    new_image_url = blob.public_url
    print("Image processed : ", new_image_url)
    
    update_database(file_name, new_image_url)

def update_database(file_name, new_image_url):
    ref = db.reference(f'output/')
    print("ref: ",ref)
    ref.update({
        'file_name':file_name,
        'processed_image_url': new_image_url
    })

def process_image(file_name):
    local_path = download_image(file_name)
    
    results = model(local_path)
    print("result: ",results)
    
    results_json = results.pandas().xyxy[0].to_json(orient='records')
    print("results_json: ",results_json)

    results_data = json.loads(results_json)
    
    image = cv2.imread(local_path)
    
    for result in results_data:
        xmin = int(result['xmin'])
        ymin = int(result['ymin'])
        xmax = int(result['xmax'])
        ymax = int(result['ymax'])
        confidence = result['confidence']
        class_name = result['name']
        
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    
    processed_image_path = f"./output/processed_{file_name}"
    cv2.imwrite(processed_image_path, image)
    
    upload_image(file_name, processed_image_path)
ref = db.reference('/')
ref.listen(listener)

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'predictions': 'test'})

if __name__ == "__main__":
    app.run(debug=True)
