import torch
import pickle
from torch import nn
from tqdm import tqdm
from MyDataLoader import *
from animation_marking import plot_loss_curve


class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,linear_size,output_size,train_steps=20):
        super().__init__()
        
        self.rnn=nn.LSTM(input_size,hidden_size,num_layers=3,batch_first=True,dropout=0.1)
        self.linear=nn.Sequential(nn.Dropout(0.1),
            nn.Linear(hidden_size, linear_size),
            nn.BatchNorm1d(linear_size), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(linear_size, output_size)
        )
        
        self.hidden_size=hidden_size
        
        # self.loss_scale = torch.arange(1,train_steps+1).reshape(-1,1).float()
    
    def forward(self,x,state):
        y_hat,new_state=self.rnn(x,state)
        y_hat=y_hat.reshape(-1,y_hat.shape[-1])
        y_hat=self.linear(y_hat)
        
        if isinstance(new_state,tuple):
            new_state=tuple(item.detach() for item in new_state)
        else:
            new_state=new_state.detach()
        return y_hat,new_state



with open(r"C:\Users\34362\Desktop\LSTM\results\circular.pkl","rb")as f:
    circular=pickle.load(f)
with open(r"C:\Users\34362\Desktop\LSTM\results\infinity_like.pkl","rb")as f:
    infinity_like=pickle.load(f)
    
    

if __name__=='__main__':
    batch_size,train_steps,input_size,hidden_size,linear_size,output_size,lr=64,20,3,300,168,3,0.001
    pred_steps=15
    test_num=10
    epochs=100
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    circular=normalize(circular)
    data_loder=MyDataLoader(circular,0.8,10,batch_size,train_steps,pred_steps)
    net=LSTM(input_size,hidden_size,linear_size,output_size).to(device)
    criterion=nn.MSELoss(reduction="none")
    optimizer=torch.optim.AdamW(net.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2)
    start_state=init_state(batch_size,hidden_size,device=device,type='lstm')
    
   
    loss_list=[]
    val_loss_list=[]
    train_loss_best=500
    val_loss_best=500
    patience=5
    stop_count = 0
    
    tqdm_iter=tqdm(range(epochs))
    for epoch in tqdm_iter:
        train_loss=train_time_seq(data_loder,net,start_state,criterion,optimizer,device).cpu().item()
        val_loss=val_time_seq(data_loder.val_iter(),net,start_state,criterion,device).cpu().item()
        loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        scheduler.step(val_loss)
    
        if val_loss < val_loss_best:
            val_loss_best = val_loss
            stop_count = 0
            torch.save(net.state_dict(), 'time_seq_best.opt')
        else:
            stop_count += 1
        if stop_count >= patience:
            print(f"Early stopping at epoch {epoch}, best validation loss: {val_loss_best}")
            print(f"Training loss: {train_loss}")
            break
        
        tqdm_iter.set_postfix(train_loss=f"{train_loss}",val_loss=f"{val_loss}")
    
    torch.save(net.state_dict(), 'time_seq_final.opt')        
    plot_loss_curve(loss_list,val_loss_list)

