from tqdm import tqdm
from torch import optim
import torch
import torch.nn as nn
import torch.nn.functional as F

def train(device, model, dataloader, learning_rate, epochs, output_window):
    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 0.0001)
    criterion = nn.L1Loss()
    torch.autograd.set_detect_anomaly(True)

    epoch = 0
    with tqdm(range(epochs)) as tr:
        for i in tr:
            total_loss = 0.0
            for x,y in dataloader:
                optimizer.zero_grad()
                x = x.to(device).float()
                y = y.to(device).float()
                #1 - (epoch/epochs)
                output = model(x, y, output_window, 1).to(device)
                #print("model output shape : ", output.shape)
                #print("label shape : ", y.shape)
                loss = criterion(output, y)
                #print("loss : ", loss)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                total_loss = total_loss + loss.cpu().item()
            tr.set_postfix(loss="{0:.5f}".format(total_loss/len(dataloader)))
            epoch = epoch + 1

    return model, optimizer

def save_model(PATH, model, optimizer):
    torch.save(model, PATH + '/model.pt')
    torch.save(model.state_dict(), PATH + 'model_state_dict.pt')
    torch.save({
        'model' : model.state_dict(),
        'optimizer' : optimizer.state_dict()
    }, PATH + 'all.tar')


def load_model(PATH):
    model = torch.load(PATH + '/model.pt')
    
    # load state_dict and save at the model
    #model.load_state_dict(torch.load(PATH + 'model_state_dict.pt'))

    '''
    # load checkpoint and do more training
    checkpoint = torch.load(PATH + 'all.tar')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    '''

    return model