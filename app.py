from flask import Flask, request, render_template, jsonify
from emotion_detection import detect_emotion_from_frame
import cv2
import numpy as np
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data_url = request.json['image']
        header, encoded = data_url.split(",", 1)
        img_data = base64.b64decode(encoded)
        np_data = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

        emotion = detect_emotion_from_frame(frame)
        return jsonify({"emotion": emotion})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/juego')
def juego():
    return render_template('juego.html')

@app.route('/estado')
def estado():
    return jsonify(rompe.get_game_state())
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
