import torch
import numpy as np
from tqdm import tqdm

class StandardScaler():
    
    def __init__(self):
        self.mean = 0.
        self.std = 1.
        
    def fit(self,dataset,train_ratio=0.8): 
        train_data = dataset[:int(len(dataset)*train_ratio)]
        if len(train_data) == 0:
            raise ValueError("Train dataset is empty, check train_ratio or dataset size")
        
        all_data = np.concatenate(train_data, axis=0)
        self.mean = np.mean(all_data,axis=0)[1:]
        self.std = np.std(all_data,axis=0)[1:]
        self.std = np.where(self.std == 0, 1e-8, self.std)

    def transform(self,dataset): 
        trans_data = []
          
        for i in range(0,len(dataset)):
            #标准化，去掉时间维度
            data = np.array(dataset[i])[:,1:]
            data = (data-self.mean)/self.std
            data_tensor = torch.tensor(data)
            trans_data.append(data_tensor)
        
        return trans_data
    
    def inverse_transform(self,mutil_steps_pred,train_steps,pred_steps,input_size,test_num=10):
        
        datas=[]
        #填充前20个时间步，方便可视化
        pad_first=np.zeros((train_steps,pred_steps,input_size))
        
        for i in range(test_num):
            data = mutil_steps_pred[i].cpu().numpy() * self.std
            data = data+self.mean
            data = np.concatenate([pad_first,data],axis=0)
            datas.append(data)

        return datas



def prepare_data(dataset,test_num=10,train_ratio=0.8,batch_size=32,train_steps=20,pred_steps=12,type="train"):

    data_list_x=[]
    data_list_y=[]


    def data_x(pos):
            return data_tensor[pos-train_steps:pos]
    def data_y(pos):
        return data_tensor[pos:pos+pred_steps]
    
    if type=="train":
        for i in range(0,int(len(dataset)*train_ratio)):
            data_tensor=dataset[i]
            for pos in range(train_steps,len(data_tensor)-pred_steps+1): 
                data_list_x.append(data_x(pos))
                data_list_y.append(data_y(pos))
                
    elif type=="val":
        for i in range(int(len(dataset)*(train_ratio)),len(dataset)-test_num):
            data_tensor=dataset[i]
            for pos in range(train_steps,len(data_tensor)-pred_steps+1): 
                data_list_x.append(data_x(pos))
                data_list_y.append(data_y(pos))
    else:
        raise ValueError(f"unknown:{type}")

    train_data_x=torch.stack(data_list_x)
    train_data_y=torch.stack(data_list_y)
    total = len(train_data_x)
    
    for i in range(0, total, batch_size):
        batch_x = train_data_x[i : i + batch_size]
        batch_y = train_data_y[i : i + batch_size]
        
        yield batch_x,batch_y
        



def multi_steps_data(dataset,test_num=10,train_step=20,input_size=3): 

    data_true_list=[]
    
    for i in range(int(len(dataset)-test_num),len(dataset)):
    
        data_tensor=dataset[i]
        end_idx=len(data_tensor)-train_step
        data_true = torch.zeros((end_idx, train_step,input_size))
        
        for j in range(end_idx):
            data_true[j]=data_tensor[j:j+train_step]
            
        data_true_list.append(data_true)
        
    return data_true_list



class MyDataLoader(object):
    def __init__(self,dataset,train_ratio=0.8,test_num=10,batch_size=32,train_steps=20,pred_steps=12,type="train"):
        self.dataset,self.train_ratio,self.test_num,self.batch_size,self.train_steps=dataset,train_ratio,test_num,batch_size,train_steps
        self.pred_steps=pred_steps
        self.type=type
        
    def __iter__(self):
        return self.data_iter()
    
    def data_iter(self):
        return prepare_data(self.dataset,self.test_num,self.train_ratio,self.batch_size,self.train_steps,self.pred_steps,self.type)
    
    def test_iter(self,input_size):
        data_true_list=multi_steps_data(self.dataset,self.test_num,self.train_steps,input_size)
        return data_true_list
    
    
def init_state(batch_size, hidden_dim, device, num_layer, type="lstm"):
    if type == "rnn":
        return torch.zeros((num_layer, batch_size, hidden_dim)).to(device)
    elif type == "lstm":
        return (torch.zeros((num_layer, batch_size, hidden_dim)).to(device),
                torch.zeros((num_layer, batch_size, hidden_dim)).to(device))
        
        
        
def train_time_seq(time_seq_loader, net, hidden_size,num_layer, criterion, optimizer, device):
    
    net.train()
    total_loss = []
    
    for x, y in time_seq_loader:
        batch_size=x.shape[0]
        state=init_state(batch_size,hidden_size,device,num_layer)
        x, y = x.float().to(device), y.float().to(device)
        y_hat, state = net(x, state)
        loss = torch.sum(criterion(y_hat, y).to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss.append(loss)
    return sum(total_loss) / len(total_loss)


def val_time_seq(time_seq_loader, net,hidden_size,num_layer,criterion, device):
    
    net.eval()
    total_loss = []
    
    with torch.no_grad():
        for x,y in time_seq_loader:

            batch_size = x.shape[0]
            state=init_state(batch_size,hidden_size,device,num_layer)
            x, y = x.float().to(device), y.float().to(device)
            y_hat,state=net(x,state)
            loss = torch.sum(criterion(y_hat, y).to(device))
            total_loss.append(loss)
            
    return sum(total_loss) / len(total_loss)


def pred_time_seq(time_seq_loader, net,device,test_num,train_steps,num_layer,hidden_size=300,pred_step=20):
    
    net.eval()
    pred_list = []
    
    with torch.no_grad():
        for i in range(test_num):
            
            x=time_seq_loader[i].to(device)
            state=init_state(len(x), hidden_size, device,num_layer)
            y_hat,_ = net(x,state)
            pred_list.append(y_hat)
           
        return pred_list
    
    




__all__ = ["prepare_data","multi_steps_data","MyDataLoader","init_state"
           ,"train_time_seq","val_time_seq","pred_time_seq","StandardScaler"]
  