from flask import Flask, request, jsonify
import io
import os
from faster_whisper import WhisperModel

app = Flask(__name__)

# Initialize the Whisper model

# Load parameters from environment variables
device = "cuda"
model = "large-v3"
compute_type = "auto"
if os.environ.get('INFERENCE_DEVICE') is not None:
    device = os.environ.get('INFERENCE_DEVICE')
if os.environ.get('INFERENCE_MODEL') is not None:
    device = os.environ.get('INFERENCE_MODEL')

# Load model
model = WhisperModel(model, device=device, compute_type=compute_type)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/inference', methods=['POST'])
def inference():
    # Check if a file is provided
    if 'file' not in request.files:
        return "No file provided", 400
    file = request.files['file']

    # Convert the file to a binary stream
    file_stream = io.BytesIO(file.read())

    # Perform transcription
    try:
        # Note: Ensure that your Whisper model can handle the file stream directly.
        # If it requires a file path, you might need to modify the model's code.
        segments, info = model.transcribe(file_stream, beam_size=5)
        segments = list(segments)
        return jsonify({'text': list(segments)[0].text})
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000)
