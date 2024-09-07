from flask import Flask, request, jsonify
import io
from PIL import Image
import torch
import firebase_admin
from firebase_admin import credentials, storage
import google.api_core.exceptions
from concurrent.futures import ThreadPoolExecutor, as_completed


app = Flask(__name__)

# Initialize Firebase Admin SDK
cred = credentials.Certificate('yolo-c6728-d395a2537229.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'yolo-c6728.appspot.com'
})


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# Initialize thread pool for asynchronous Firebase uploads
executor = ThreadPoolExecutor(max_workers=5)


def upload_image_to_firebase(image):
    try:
        bucket = storage.bucket()
        blob = bucket.blob(f'input/{image.filename}')
        image.seek(0)
        blob.upload_from_string(image.read(), content_type=image.content_type)
        blob.make_public()
    except google.api_core.exceptions.NotFound as e:
        return {'filename': image.filename, 'status': 'error', 'details': 'Bucket not found or invalid URL'}
    except Exception as e:
        return {'filename': image.filename, 'status': 'error', 'details': str(e)}   


@app.route('/predict', methods=['POST'])
def predict():
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400

    images = request.files.getlist('images')
    results_list = []

    for file in images:
        img = Image.open(io.BytesIO(file.read()))

        # Predict using YOLO model
        results = model(img)
        results_json = results.pandas().xyxy[0].to_json(orient='records')
        print('results_json:', results_json)

        results_list.append({
            'filename': file.filename,
            'results': results_json
        })

        # Asynchronously upload image to Firebase
        executor.submit(upload_image_to_firebase, file)

    return jsonify({'predictions': results_list})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)