# -*- coding: gbk -*-

import torch
import numpy as np
from tqdm import tqdm

def normalize(dataset,train_ratio=0.8):
    
    data_norm=[]
    
    all_data = np.concatenate(dataset[:int(len(dataset)*train_ratio)], axis=0)
    data_mean=np.mean(all_data,axis=0)[1:]
    data_std=np.std(all_data,axis=0)[1:]
    np.savez(r"C:\Users\34362\Desktop\LSTM\results\norm.npz",mean=data_mean,std=data_std)
    for i in range(0,len(dataset)):
        data=np.array(dataset[i])[:,1:]
        data=(data-data_mean)/data_std
        data_tensor=torch.tensor(data)
        data_norm.append(data_tensor)
    
    return data_norm



def prepare_data(dataset,train_ratio=0.8,batch_size=32,train_steps=20):
    
    dataset=dataset
    data_list_x=[]
    data_list_y=[]


    def data_x(pos):
            return data_tensor[pos-train_steps:pos]
    def data_y(pos):
        return data_tensor[pos-train_steps+1:pos+1]
    
    for i in range(0,int(len(dataset)*train_ratio)):
        data_tensor=dataset[i]
        for pos in range(train_steps,len(data_tensor)): 
            data_list_x.append(data_x(pos))
            data_list_y.append(data_y(pos))
            
    train_data_x=torch.stack(data_list_x)
    train_data_y=torch.stack(data_list_y)
    
    for batch in range(len(train_data_x)//batch_size):
        train_iter_x=train_data_x[batch*batch_size:(batch+1)*batch_size]
        train_iter_y=train_data_y[batch*batch_size:(batch+1)*batch_size]
        
        yield train_iter_x,train_iter_y
        
        
           
def prepare_val(dataset,train_ratio=0.8,test_num=10,batch_size=32,train_steps=20): 
    
    val_list_x=[]
    val_list_y=[]  

    
    def data_x(pos):
            return data_tensor[pos-train_steps:pos]
    def data_y(pos):
        return data_tensor[pos-train_steps+1:pos+1]
    
    for i in range(int(len(dataset)*(train_ratio)),len(dataset)-test_num):
        data_tensor=dataset[i]
        
        for pos in range(train_steps,len(data_tensor)): 
            val_list_x.append(data_x(pos))
            val_list_y.append(data_y(pos))
            
    val_data_x=torch.stack(val_list_x)
    val_data_y=torch.stack(val_list_y)
    
    for batch in range(len(val_data_x)//batch_size):
        val_iter_x=val_data_x[batch*batch_size:(batch+1)*batch_size]
        val_iter_y=val_data_y[batch*batch_size:(batch+1)*batch_size]
        
        yield val_iter_x,val_iter_y




def multi_steps_data(dataset,test_num=10,train_step=20): 

    data_true_list=[]
    
    for i in range(int(len(dataset)-test_num),len(dataset)):
    
        data_tensor=dataset[i]
        end_idx=len(data_tensor)-train_step

        data_true = torch.zeros((end_idx, train_step,3))
        
        for j in range(end_idx):
            data_true[j]=data_tensor[j:j+train_step]
            
        data_true_list.append(data_true)
        
    return data_true_list



class MyDataLoader(object):
    def __init__(self,dataset,train_ratio=0.8,test_num=10,batch_size=32,train_steps=20,pred_steps=12):
        self.dataset,self.train_ratio,self.test_num,self.batch_size,self.train_steps=dataset,train_ratio,test_num,batch_size,train_steps
        self.pred_steps=pred_steps
    def __iter__(self):
        return self.train_iter()
    
    def train_iter(self):
        return prepare_data(self.dataset,self.train_ratio,self.batch_size,self.train_steps)
    
    def val_iter(self):
        return prepare_val(self.dataset,self.train_ratio,self.test_num,self.batch_size,self.train_steps)
    
    def test_iter(self):
        data_true_list=multi_steps_data(self.dataset,self.test_num,self.train_steps)
        return data_true_list
    
    
def init_state(batch_size, hidden_dim, device, num_layer=3, type="lstm"):
    if type == "rnn":
        return torch.zeros((num_layer, batch_size, hidden_dim)).to(device)
    elif type == "lstm":
        return (torch.zeros((num_layer, batch_size, hidden_dim)).to(device),
                torch.zeros((num_layer, batch_size, hidden_dim)).to(device))
        
        
        
def train_time_seq(time_seq_loader, net, state, criterion, optimizer, device):
    

    total_loss = []
    
    for x, y in time_seq_loader:
        if isinstance(state, tuple):
            state = tuple(torch.zeros_like(item) for item in state)
        else:
            state = torch.zeros_like(state, device=device)
        x, y = x.float().to(device), y.flatten().reshape(-1,y.shape[-1]).float().to(device)
        y_hat, state = net(x, state)
        loss = torch.sum(criterion(y_hat, y).to(device))
        # loss = torch.sum(criterion(y_hat, y) * torch.cat([net.loss_scale for _ in range(time_seq_loader.batch_size)], dim=0).to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss.append(loss)
    return sum(total_loss) / len(total_loss)


def val_time_seq(time_seq_loader, net, state, criterion, device):
    net.eval()
    total_loss = []
    with torch.no_grad():
        for x,y in time_seq_loader:
        
            if isinstance(state, tuple):
                state = tuple(torch.zeros_like(item) for item in state)
            else:
                state = torch.zeros_like(state, device=device)
            batch_size = y.shape[0]
            x, y = x.float().to(device), y.flatten().reshape(-1,y.shape[-1]).float().to(device)
            y_hat,state=net(x,state)
            loss = torch.sum(criterion(y_hat, y).to(device))
            # loss = torch.sum(criterion(y_hat, y) * torch.cat([net.loss_scale for _ in range(batch_size)], dim=0).to(device))
            total_loss.append(loss)
            
    return sum(total_loss) / len(total_loss)


def pred_time_seq(time_seq_loader, net,device,test_num,train_steps,hidden_size=300,pred_step=20):
    net.eval()
    pred_list = []
    with torch.no_grad():
        for i in range(test_num):
            x=time_seq_loader[i]

            state=init_state(len(x), hidden_size, device)
            x=x[:,:train_steps]
            tqdm_iter = tqdm(range(pred_step), desc="Multi step predict")
            for j in tqdm_iter:
                train_data=x[:,j:j+train_steps]
                # print(train_data.shape)
                y_hat,_=net(train_data,state)
                y_hat=y_hat.reshape(-1,train_steps,3)[:,-1,:].unsqueeze(1)

                x=torch.cat([x,y_hat],dim=1)

            pred_data=x[:,-pred_step-1:]
            pred_list.append(pred_data) 
        return pred_list
    
    

def denorm_and_pad(mutil_steps_pred,pred_steps,train_steps,input_size):
    
    datas=[]
    norm_data=np.load(r"C:\Users\34362\Desktop\LSTM\results\norm.npz")
    data_mean=norm_data["mean"]
    data_std=norm_data["std"]
    print(data_mean)
    print(data_std)
    
    pad_first=np.zeros((train_steps,pred_steps+1,input_size))
    for i in range(10):
        data=mutil_steps_pred[i].cpu().numpy() * data_std
        data=data+data_mean
        data=np.concatenate([pad_first,data],axis=0)
        datas.append(data)

    return datas



__all__ = ["normalize","prepare_data","prepare_val","multi_steps_data","MyDataLoader","init_state"
           ,"train_time_seq","val_time_seq","pred_time_seq","denorm_and_pad"]
  