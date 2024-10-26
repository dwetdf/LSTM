from flask import Flask, render_template
from flask_socketio import SocketIO
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import psutil
from cpu_train import LSTMModel 

app = Flask(__name__)
socketio = SocketIO(app)

def load_model(model_path, input_size, hidden_size, num_layers, output_size):
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

def prepare_input(data, sequence_length):
    return np.array(data[-sequence_length:]).reshape(1, sequence_length, 2)

def predict_next_usage(model, input_data, scaler):
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_data).to(next(model.parameters()).device)
        output = model(input_tensor)
        predicted = output.cpu().numpy()
    predicted = scaler.inverse_transform(predicted.reshape(-1, 2))
    return predicted[0]

def get_current_usage():
    return psutil.cpu_percent(interval=1), psutil.virtual_memory().percent

@app.route('/')
def index():
    return render_template('LR.html')

def background_task():
    # 设置参数
    model_path = 'cpu_memory_lstm_model.pth'
    sequence_length = 10
    hidden_size = 50
    num_layers = 2

    # 加载模型
    model = load_model(model_path, input_size=2, hidden_size=hidden_size, num_layers=num_layers, output_size=2)

    # 初始化scaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # 初始化数据列表
    usage_data = []

    while True:
        current_cpu, current_memory = get_current_usage()
        usage_data.append([current_cpu, current_memory])

        if len(usage_data) > sequence_length:
            usage_data = usage_data[-sequence_length:]

        if len(usage_data) == sequence_length:
            scaler.fit(np.array(usage_data))
            scaled_data = scaler.transform(np.array(usage_data))
            input_data = prepare_input(scaled_data, sequence_length)
            predicted_cpu, predicted_memory = predict_next_usage(model, input_data, scaler)

            # 将 float32 转换为原生 Python float
            current_cpu = float(current_cpu)
            current_memory = float(current_memory)
            predicted_cpu = float(predicted_cpu)
            predicted_memory = float(predicted_memory)

            socketio.emit('update_data', {
                'current_cpu': current_cpu, 
                'current_memory': current_memory,
                'predicted_cpu': predicted_cpu, 
                'predicted_memory': predicted_memory
            })

        time.sleep(5)  # 每5秒更新一次

if __name__ == '__main__':
    socketio.start_background_task(background_task)
    socketio.run(app, debug=True)
