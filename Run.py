import torch
import pickle
from torch import nn
from tqdm import tqdm
from mydataloader2 import *
from animation_marking import plot_loss_curve
from seq2seq import Seq2Seq


with open(r"C:\Users\34362\Desktop\LSTM\results\circular.pkl","rb")as f:
    circular=pickle.load(f)
with open(r"C:\Users\34362\Desktop\LSTM\results\infinity_like.pkl","rb")as f:
    infinity_like=pickle.load(f)
    
    
if __name__=='__main__':
    batch_size,train_steps,input_size,hidden_size,linear_size,output_size,lr=64,20,3,300,168,3,0.000005
    num_layer=2
    pred_steps=12
    test_num=10
    epochs=100
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler()
    scaler.fit(circular)
    circular_trans=scaler.transform(circular)
    data_loder=MyDataLoader(circular_trans,0.8,10,batch_size,train_steps,pred_steps)
    val_loder=MyDataLoader(circular_trans,0.8,10,batch_size,train_steps,pred_steps,type="val")
    net=Seq2Seq(input_size,hidden_size,output_size,num_layer,linear_size,pred_steps,train_steps).to(device)
    criterion=nn.MSELoss(reduction="none")
    optimizer=torch.optim.AdamW(net.parameters(),lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2)
    
    
   
    loss_list=[]
    val_loss_list=[]
    val_loss_best=500000
    patience=5
    stop_count = 0
    net.load_state_dict(
    torch.load("time_seq_best.opt", map_location=device, weights_only=False)
)
    
    tqdm_iter=tqdm(range(epochs))
    for epoch in tqdm_iter:
        train_loss=train_time_seq(data_loder,net,hidden_size,num_layer,criterion,optimizer,device).cpu().item()
        val_loss=val_time_seq(val_loder,net,hidden_size,num_layer,criterion,device).cpu().item()
        loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        scheduler.step(val_loss)
        print(f"epoch:{epoch},train_loss:{train_loss},val_loss:{val_loss}")
    
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
