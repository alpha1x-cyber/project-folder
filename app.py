from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from scipy.fft import fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# 1. تحميل البيانات بشكل تدريجي
def load_data_in_chunks(file_path, chunk_size=1000000):
    data_chunks = []
    with open(file_path, 'rb') as f:
        while True:
            chunk = np.fromfile(f, dtype=np.float32, count=chunk_size)
            if len(chunk) == 0:
                break
            data_chunks.append(chunk)
    return np.concatenate(data_chunks)

# 2. تحليل الإشارات الراديوية
def analyze_rf_signals(data):
    fft_data = fft(data)
    frequencies = np.fft.fftfreq(len(data))
    return fft_data, frequencies

# 3. تمييز الإشارات باستخدام Random Forest
def classify_signals(fft_data, labels):
    X_train, X_test, y_train, y_test = train_test_split(fft_data, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train.reshape(-1, 1), y_train)
    predictions = model.predict(X_test.reshape(-1, 1))
    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy

# 4. تحديد موقع الطائرة بدون طيار باستخدام TDOA
def tdoa_localization(signal_times, sensor_positions):
    time_differences = np.diff(signal_times)
    estimated_position = np.mean(sensor_positions, axis=0)
    return estimated_position

# 5. تنفيذ الإجراءات المضادة (التشويش)
def jamming_signal(frequency, duration=1000):
    jamming_signal = np.sin(2 * np.pi * frequency * np.arange(duration))
    return jamming_signal

# صفحة الرئيسية
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            # تحميل البيانات
            data = load_data_in_chunks(file_path)
            fft_data, frequencies = analyze_rf_signals(data)
            
            # افتراض أن لديك تسميات (labels) للبيانات
            labels = np.random.randint(0, 2, size=len(data))
            
            # تمييز الإشارات
            model, accuracy = classify_signals(fft_data, labels)
            
            # تحديد موقع الطائرة بدون طيار
            signal_times = np.random.rand(3)
            sensor_positions = np.array([[0, 0], [1, 0], [0, 1]])
            estimated_position = tdoa_localization(signal_times, sensor_positions)
            
            # تنفيذ إجراءات مضادة (التشويش)
            jamming_frequency = 2.4e9
            jamming_signal_output = jamming_signal(jamming_frequency)
            
            # عرض النتائج
            results = {
                'accuracy': accuracy * 100,
                'estimated_position': estimated_position,
                'jamming_signal': jamming_signal_output[:10].tolist()
            }
            return render_template('index.html', results=results)
    
    return render_template('index.html', results=None)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
