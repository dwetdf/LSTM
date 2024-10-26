import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 假设我们已经定义了LSTMModel类
from cpu_train import LSTMModel

def load_model(model_path, input_size, hidden_size, num_layers, output_size):
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def prepare_input(data, sequence_length):
    return np.array(data[-sequence_length:]).reshape(1, sequence_length, 2)

def predict_next_usage(model, input_data, scaler):
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_data).to(next(model.parameters()).device)
        output = model(input_tensor)
        predicted = output.cpu().numpy()
    # 反归一化预测结果
    predicted = scaler.inverse_transform(predicted.reshape(-1, 2))
    return predicted[0]

def main():
    # 设置参数
    model_path = 'cpu_memory_lstm_model.pth'
    data_path = 'data/cpu_memory_random.csv'
    sequence_length = 10
    hidden_size = 50
    num_layers = 2

    # 加载模型
    model = load_model(model_path, input_size=2, hidden_size=hidden_size, num_layers=num_layers, output_size=2)

    # 加载数据
    df = pd.read_csv(data_path)
    data = df[['cpu', 'memory']].values

    # 创建并拟合scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data)

    # 准备输入数据
    scaled_data = scaler.transform(data)
    input_data = prepare_input(scaled_data, sequence_length)

    # 进行预测
    predicted_usage = predict_next_usage(model, input_data, scaler)

    print(f"预测的下一个时间步的CPU使用率: {predicted_usage[0]:.2f}%")
    print(f"预测的下一个时间步的内存使用率: {predicted_usage[1]:.2f}%")

    # 预测未来多个时间步（例如，预测未来5个时间步）
    future_predictions = []
    current_input = input_data.copy()
    for _ in range(5):
        next_pred = predict_next_usage(model, current_input, scaler)
        future_predictions.append(next_pred)
        # 更新输入序列
        current_input = np.roll(current_input, -1, axis=1)
        current_input[0, -1, :] = scaler.transform([next_pred])[0]

    print("未来5个时间步的使用率预测:")
    for i, pred in enumerate(future_predictions, 1):
        print(f"  步骤 {i}: CPU: {pred[0]:.2f}%, 内存: {pred[1]:.2f}%")

if __name__ == "__main__":
    main()
