import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
import psutil
from cpu_train import LSTMModel

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

def main():
    # 设置参数
    model_path = 'cpu_memory_lstm_model.pth'
    sequence_length = 10
    hidden_size = 50
    num_layers = 2
    prediction_interval = 5  # 每5秒进行一次预测

    # 加载模型
    model = load_model(model_path, input_size=2, hidden_size=hidden_size, num_layers=num_layers, output_size=2)

    # 初始化scaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # 初始化数据列表
    usage_data = []

    print("开始实时CPU和内存使用率预测...")
    while True:
        # 获取当前CPU和内存使用率
        current_cpu, current_memory = get_current_usage()
        usage_data.append([current_cpu, current_memory])

        # 保持数据列表长度为sequence_length
        if len(usage_data) > sequence_length:
            usage_data = usage_data[-sequence_length:]

        # 当收集到足够的数据时开始预测
        if len(usage_data) == sequence_length:
            # 重新拟合scaler
            scaler.fit(np.array(usage_data))

            # 准备输入数据
            scaled_data = scaler.transform(np.array(usage_data))
            input_data = prepare_input(scaled_data, sequence_length)

            # 进行预测
            predicted_cpu, predicted_memory = predict_next_usage(model, input_data, scaler)

            print(f"当前CPU使用率: {current_cpu:.2f}%, 内存使用率: {current_memory:.2f}%")
            print(f"预测的下一个时间步的CPU使用率: {predicted_cpu:.2f}%, 内存使用率: {predicted_memory:.2f}%")
            print("-" * 50)

        # 等待一段时间再进行下一次预测
        time.sleep(prediction_interval)

if __name__ == "__main__":
    main()
