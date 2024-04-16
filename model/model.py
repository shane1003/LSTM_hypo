import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class LSTM(nn.Module):
    def __init__(self, device, input_size, hidden_size, num_layer, dropout = 0):
        super(LSTM, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.dropout = dropout
        self.lstm_layer_cnt = len(hidden_size)
        
        if self.lstm_layer_cnt == 1:
            self.LSTM0 = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size[0], num_layers = self.num_layer[0], dropout = self.dropout, batch_first = True)

        elif self.lstm_layer_cnt == 2:
            self.LSTM0 = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size[0], num_layers = self.num_layer[0], dropout = self.dropout, batch_first = True)        
            self.LSTM1 = nn.LSTM(input_size = self.hidden_size[0], hidden_size = self.hidden_size[1], num_layers = self.num_layer[1], dropout = self.dropout)

        elif self.lstm_layer_cnt == 3:
            self.LSTM0 = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size[0], num_layers = self.num_layer[0], dropout = self.dropout, batch_first = True)        
            self.LSTM1 = nn.LSTM(input_size = self.hidden_size[0], hidden_size = self.hidden_size[1], num_layers = self.num_layer[1], dropout = self.dropout)
            self.LSTM2 = nn.LSTM(input_size = self.hidden_size[1], hidden_size = self.hidden_size[2], num_layers = self.num_layer[2], dropout = self.dropout)
        
        self.fc = nn.Linear(self.hidden_size[self.lstm_layer_cnt - 1], 1)

    def lstm_result(self, x_input):
        if self.lstm_layer_cnt == 1:
            out, hidden = self.LSTM0(x_input)
            
        elif self.lstm_layer_cnt == 2:
            out, hidden = self.LSTM0(x_input)
            out, hideen = self.LSTM1(out)

        elif self.lstm_layer_cnt == 3:
            out, hidden = self.LSTM0(x_input)
            out, hidden = self.LSTM1(out)
            out, hidden = self.LSTM2(out)

        return out
        
    def forward(self, x_input, targets, target_len, teacher_forcing_ratio = 0.5):
        #print("Forward start")
        batch_size = x_input.shape[0]

        # Full teacher forcing for decoder or not.
        # Teacher Forcning(Give the answer)

        inputs = x_input.to(self.device)
        outputs = torch.zeros(batch_size, target_len, 1).to(self.device)

        if random.random() <= teacher_forcing_ratio:
            #print("Teacher Forcing!", "input shape", x_input.shape)

            out = self.lstm_result(x_input)
            out = self.fc(out)

            outputs = out
        
        # Non-Teacher Forcing(Using past output)
        else:
            print("Non-Teacher forcing")
            
            for t in range(target_len): #0 1 2 3 4 5
                out = self.lstm_result(x_input)
                out = self.fc(out)

                outputs[:, t, 0] = out[:, t, 0]

                if t < target_len - 1 : 
                    
                    x_input[:, t + 1, 0] = out[:, t , 0]

        return outputs.squeeze()
        
    def predict(self, input, target_len):
        outputs = torch.zeros(1, target_len, 1)

        if self.input_size == 1:
            input = input.unsqueeze(0)
            input = input.unsqueeze(2)
        else:
            input = input.squeeze().unsqueeze(0)
        self.eval()
        batch_size = input.shape[0]

        for t in range(target_len):
            out = self.lstm_result(input)
            out = self.fc(out)

            if t < target_len - 1 : 
                input[:, t + 1, 0] = out[:, t , 0]

            outputs[:, t, 0] = out[:, t, 0]

        return outputs.detach().numpy().squeeze()
            