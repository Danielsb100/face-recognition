from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
from recognizer import FacialRecognizer
import os
import sys
import time

# Determina o diretório base (funciona tanto no script quanto no .exe do PyInstaller)
if getattr(sys, 'frozen', False):
    # Se for o executável, o diretório base é onde o .exe está
    base_dir = os.path.dirname(sys.executable)
else:
    # Se for o script .py, o diretório base é onde o arquivo está
    base_dir = os.path.dirname(os.path.abspath(__file__))

webapp_path = os.path.join(base_dir, 'webapp_dist')
databank_path = os.path.join(base_dir, 'data_bank')

app = Flask(__name__)
CORS(app)

# Initialize the recognizer with absolute path
recognizer = FacialRecognizer(data_bank_dir=databank_path)

# Callback to update GUI text status
gui_callback = None
# Callback to update GUI image display
gui_callback_image = None

# Store the latest analysis result for polling from Even App
last_analysis_result = {
    "timestamp": 0.0,
    "names": [],
    "count": 0
}

@app.route('/')
def index():
    return send_from_directory(webapp_path, 'index.html')

@app.route('/<path:path>')
def serve_file(path):
    import os
    if os.path.exists(os.path.join(webapp_path, path)):
        return send_from_directory(webapp_path, path)
    return send_from_directory(webapp_path, 'index.html')

# Remover CORS(app) e recognizer duplicados


@app.route('/analyze', methods=['POST'])
def analyze():
    global gui_callback
    if gui_callback:
        gui_callback()

    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    
    try:
        # Read image from request to OpenCV format
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400
        
        # Resize to prevent tiny bounding boxes on 12MP phone photos
        h, w = frame.shape[:2]
        max_dim = 1000
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        
        # Run recognition and get annotated frame and names
        annotated_frame, names = recognizer.recognize_in_frame(frame, process_rescale=1.0, return_names=True)
        
        global gui_callback_image, last_analysis_result
        if gui_callback_image:
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            gui_callback_image(rgb_frame)
            
        last_analysis_result = {
            "timestamp": time.time(),
            "names": names,
            "count": len(names)
        }
            
        return jsonify({
            "names": names,
            "count": len(names)
        })
        
    except Exception as e:
        print(f"Server error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    global gui_callback
    if gui_callback:
        gui_callback()
    return jsonify({
        "status": "online",
        "known_faces_count": len(recognizer.known_face_names)
    })

@app.route('/latest_results', methods=['GET'])
def latest_results():
    return jsonify(last_analysis_result)

if __name__ == '__main__':
    # Run on all interfaces (0.0.0.0) so it's accessible via Wi-Fi IP
    app.run(host='0.0.0.0', port=5000, debug=True)
