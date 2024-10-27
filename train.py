import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 1. 数据集类
class CPUUsageDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 2. 数据加载和预处理
def load_data(file_path, sequence_length=10):
    # 从CSV文件加载数据
    df = pd.read_csv(file_path)
    
    # 假设CSV文件有两列，分别是CPU使用率和内存使用率
    data = df[['cpu', 'memory']].values
    
    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    
    # 创建序列数据
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length), :])
        y.append(data[i + sequence_length, :])
    
    X, y = np.array(X), np.array(y)
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, scaler

# 3. LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 4. 训练函数
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}] (Train)')
        for X_batch, y_batch in train_pbar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'Train Loss': f'{train_loss/len(train_loader):.4f}'})
        
        # 验证
        model.eval()
        test_loss = 0
        test_pbar = tqdm(test_loader, desc=f'Epoch [{epoch+1}/{num_epochs}] (Test)')
        with torch.no_grad():
            for X_batch, y_batch in test_pbar:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                test_loss += loss.item()
                test_pbar.set_postfix({'Test Loss': f'{test_loss/len(test_loader):.4f}'})
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Test Loss: {test_loss/len(test_loader):.4f}')

# 主函数
def main():
    # 设置参数
    file_path = 'data/cpu_memory_random.csv'  # 请替换为包含CPU和内存使用率数据的文件路径
    sequence_length = 10
    hidden_size = 50
    num_layers = 2
    num_epochs = 5
    batch_size = 32
    learning_rate = 0.001
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载和预处理数据
    X_train, X_test, y_train, y_test, scaler = load_data(file_path, sequence_length)
    
    # 创建数据加载器
    train_dataset = CPUUsageDataset(X_train, y_train)
    test_dataset = CPUUsageDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    model = LSTMModel(input_size=2, hidden_size=hidden_size, num_layers=num_layers, output_size=2)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device)
    
    # 保存模型
    torch.save(model.state_dict(), 'cpu_memory_lstm_model.pth')
    
    print("模型训练完成并保存.")

if __name__ == "__main__":
    main()
