from torch.utils.data import Dataset
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더
import torch

class CustomDataset(Dataset):
    def __init__(self, x, y):
        super(CustomDataset, self).__init__()

        self.x_data = x
        self.y_data = y
    
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])

        return x, y

'''
# 텐서로 변환
X_train_tensor = torch.FloatTensor(X_train)
Y_train_tensor = torch.FloatTensor(Y_train)

X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(Y_test)

# 텐서 형태로 데이터 정의
dataset = TensorDataset(X_train_tensor, Y_train_tensor)

# 데이터로더는 기본적으로 2개의 인자를 입력받으며 배치크기는 통상적으로 2의 배수를 사용
dataloader = DataLoader(dataset,
                        batch_size=batch,
                        shuffle = False,  
                        drop_last = False)
'''