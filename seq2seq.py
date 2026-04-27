import torch
from torch import nn



class Encoder(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super().__init__()
    
        self.rnn=nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,dropout=0.1)

    def forward(self,x,state):
        y_hat,new_state =self.rnn(x,state)
        (h,c) = new_state
        return h,c
    
    
class Decoder(nn.Module):
    def __init__(self,input_size,hidden_size,linear_size,output_size,num_layers):
        super().__init__()
        
        self.rnn=nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,dropout=0.1)
        self.linear=nn.Sequential(nn.Dropout(0.1),
            nn.Linear(hidden_size, linear_size), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(linear_size, output_size)
        )
        
        self.hidden_size=hidden_size
    
    def forward(self,x,state):
        y_hat,new_state=self.rnn(x,state)
        y_hat=y_hat.squeeze(1)
        y_hat=self.linear(y_hat)

        return y_hat,new_state
    
    
class Seq2Seq(nn.Module):
    
    def __init__(self,input_size,hidden_size,output_size,num_layers,linear_size,pred_steps,train_steps):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.pred_steps=pred_steps
        self.train_steps=train_steps
        self.encoder = Encoder(input_size,hidden_size,num_layers)
        self.decoder = Decoder(input_size, hidden_size,linear_size, output_size, num_layers)

    def forward(self,x,state):
        batch_size = x.shape[0]
        target_len = self.pred_steps
        h,c = self.encoder(x,state)
        outputs = torch.zeros(batch_size,self.pred_steps,self.output_size).to(x.device)
        decoder_input = x[:,-1,:].unsqueeze(1)
        for t in range(target_len):
            decoder_output,(h,c) = self.decoder(decoder_input,(h,c))
            outputs[:,t,:] = decoder_output
            decoder_input = decoder_output.unsqueeze(1)  
            
        h = h.detach()
        c = c.detach()
        return outputs,(h,c)
    
    