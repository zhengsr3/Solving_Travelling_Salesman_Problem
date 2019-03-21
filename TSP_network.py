import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
class Ptr_Net(nn.Module):
    def __init__(self,hidden_size=128, embedding_size=128,num_directions=2,
                 input_size=2,batch_size=128,initialization_stddev=0.1,
                 dropout_p=0):
        super(Ptr_Net, self).__init__()
        #Define Embedded
        self.Embed=torch.nn.Linear(input_size, embedding_size, bias=False)
        #Define Encoder
        self.Encoder=torch.nn.LSTM(input_size=embedding_size,hidden_size=hidden_size,batch_first=True,
                              bidirectional=True)
        #Define Attention
        self.W_ref=torch.nn.Linear(num_directions * hidden_size, num_directions * hidden_size, bias=False)
        self.W_q=torch.nn.Linear(num_directions * hidden_size, num_directions * hidden_size, bias=False)
        self.v=torch.nn.Linear(num_directions * hidden_size, 1, bias=False)
        #Define Decoder
        self.Decoder=torch.nn.LSTM(input_size=embedding_size*2,hidden_size=hidden_size,batch_first=True,
                              bidirectional=True)
        self.DropOut1= nn.Dropout(p=dropout_p)
        self.DropOut2= nn.Dropout(p=dropout_p)
        self.W_ref2=torch.nn.Linear(num_directions * hidden_size, num_directions * hidden_size, bias=False)
        self.W_q2=torch.nn.Linear(num_directions * hidden_size, num_directions * hidden_size, bias=False)
        self.v2=torch.nn.Linear(num_directions * hidden_size, 1, bias=False)
        self.Softmax_Cross_Entrophy=torch.nn.CrossEntropyLoss(reduction='none')
    def forward(self, inputs, test_roads):
        #inputs:(batch,city,2),tensor
        #test_roads:(batch,city),numpy
        #output:(batch),tensor
        #basic parameter
        city=inputs.size()[1]
        max_time_steps=inputs.size()[1]
        already_played_penalty=1e6
        #Embedding
        #inputs:(batch,city,coor)
        City_Embedding=self.Embed(inputs)
        #Encoder
        #City_Embedding:(batch,city,embedding)
        Enc, (hn, cn) = self.Encoder(City_Embedding,None)
        #Attention and Decoder
        #Enc:(batch, city, num_directions * hidden_size)
        #hn: (batch,num_layers * num_directions,  hidden_size)
        #cn: (batch,num_layers * num_directions,  hidden_size)
        decoder_input = torch.zeros(Enc.size()[0],1,Enc.size()[2])
        decoder_state = (hn,cn)
        already_played_actions = torch.zeros(Enc.size()[0],max_time_steps)
        decoder_inputs = [decoder_input]
        path_loss=0
        decoder_outputs=[]
        for i in range(max_time_steps):
            decoder_output,decoder_state=self.Decoder(decoder_input,decoder_state)
            Enc=self.DropOut1(Enc)
            decoder_output=self.DropOut2(decoder_output)
            output_weight=torch.squeeze(
                    self.v(torch.tanh(
                        self.W_ref(Enc)+self.W_q(decoder_output.repeat(1,city,1))
                    ))
                )-already_played_penalty*already_played_actions
            #attention_weight:(batch,city)
            attention_weight=torch.nn.functional.softmax(
                torch.squeeze(
                    self.v2(torch.tanh(
                        self.W_ref2(Enc)+self.W_q2(decoder_input)
                    ))
                ),dim=1
            )
            decoder_outputs.append(torch.argmax(output_weight, dim=1))
            decoder_input=torch.unsqueeze(torch.einsum('ij,ijk->ik',attention_weight,Enc),dim=1)
            already_played_actions+=torch.zeros(Enc.size()[0],max_time_steps).scatter_(1,torch.unsqueeze(decoder_outputs[-1],dim=1),1)
    
            path_loss+=self.Softmax_Cross_Entrophy(output_weight,torch.LongTensor(test_roads[:,i].T).squeeze())
        return path_loss
    def get_road(self,inputs,random=False):
        #inputs:(batch,city,2),tensor
        #output:(batch,city),numpy
        #basic parameter
        city=inputs.size()[1]
        max_time_steps=inputs.size()[1]
        already_played_penalty=1e6
        #Embedding
        #inputs:(batch,city,coor)
        City_Embedding=self.Embed(inputs)
        #Encoder
        #City_Embedding:(batch,city,embedding)
        Enc, (hn, cn) = self.Encoder(City_Embedding,None)
        #Attention and Decoder
        #Enc:(batch, city, num_directions * hidden_size)
        #hn: (batch,num_layers * num_directions,  hidden_size)
        #cn: (batch,num_layers * num_directions,  hidden_size)
        decoder_input = torch.zeros(Enc.size()[0],1,Enc.size()[2])
        decoder_state = (hn,cn)
        already_played_actions = torch.zeros(Enc.size()[0],max_time_steps)
        decoder_inputs = [decoder_input]
        decoder_outputs=[]
        for i in range(max_time_steps):
            decoder_output,decoder_state=self.Decoder(decoder_input,decoder_state)
            #print(decoder_output.size())
            Enc=self.DropOut1(Enc)
            decoder_output=self.DropOut2(decoder_output)
            '''print(torch.tanh(
                        self.W_ref(Enc)+self.W_q(decoder_output.repeat(1,city,1))
                    ))'''
            output_weight=torch.squeeze(
                    self.v(torch.tanh(
                        self.W_ref(Enc)+self.W_q(decoder_output.repeat(1,city,1))
                    ))
                )-already_played_penalty*already_played_actions
            #print(attention_weight)
            attention_weight=torch.nn.functional.softmax(
                torch.squeeze(
                    self.v2(torch.tanh(
                        self.W_ref2(Enc)+self.W_q2(decoder_input)
                    ))
                ),dim=1
            )
            if random==False:
                decoder_outputs.append(torch.argmax(output_weight, dim=1))
            else:
                #print(attention_weight)
                decoder_outputs.append(torch.argmax(output_weight, dim=1))
            decoder_input=torch.unsqueeze(torch.einsum('ij,ijk->ik',attention_weight,Enc),dim=1)
            already_played_actions+=torch.zeros(Enc.size()[0],max_time_steps).scatter_(1,torch.unsqueeze(decoder_outputs[-1],dim=1),1)
        return np.array([list(item) for item in decoder_outputs]).T
    def print_prob(self,inputs,random=False):
        #inputs:(batch,city,2),tensor
        #output:(batch,city),numpy
        #basic parameter
        city=inputs.size()[1]
        max_time_steps=inputs.size()[1]
        already_played_penalty=1e6
        #Embedding
        #inputs:(batch,city,coor)
        City_Embedding=self.Embed(inputs)
        #Encoder
        #City_Embedding:(batch,city,embedding)
        Enc, (hn, cn) = self.Encoder(City_Embedding,None)
        #Attention and Decoder
        #Enc:(batch, city, num_directions * hidden_size)
        #hn: (batch,num_layers * num_directions,  hidden_size)
        #cn: (batch,num_layers * num_directions,  hidden_size)
        decoder_input = torch.zeros(Enc.size()[0],1,Enc.size()[2])
        decoder_state = (hn,cn)
        already_played_actions = torch.zeros(Enc.size()[0],max_time_steps)
        decoder_inputs = [decoder_input]
        decoder_outputs=[]
        for i in range(max_time_steps):
            decoder_output,decoder_state=self.Decoder(decoder_input,decoder_state)
            #print(decoder_output.size())
            Enc=self.DropOut1(Enc)
            decoder_output=self.DropOut2(decoder_output)
            output_weight=torch.squeeze(
                    self.v(torch.tanh(
                        self.W_ref(Enc)+self.W_q(decoder_output.repeat(1,city,1))
                    ))
                )-already_played_penalty*already_played_actions
            print(output_weight)
            print(torch.nn.functional.softmax(output_weight))
            attention_weight=torch.nn.functional.softmax(
                torch.squeeze(
                    self.v2(torch.tanh(
                        self.W_ref2(Enc)+self.W_q2(decoder_input)
                    ))
                ),dim=1
            )
            if random==False:
                decoder_outputs.append(torch.argmax(output_weight, dim=1))
            else:
                print(attention_weight)
                decoder_outputs.append(torch.argmax(output_weight, dim=1))
            decoder_input=torch.unsqueeze(torch.einsum('ij,ijk->ik',attention_weight,Enc),dim=1)
            already_played_actions+=torch.zeros(Enc.size()[0],max_time_steps).scatter_(1,torch.unsqueeze(decoder_outputs[-1],dim=1),1)
        #return np.array([list(item) for item in decoder_outputs]).T
class Critic_Net(nn.Module):
    def __init__(self,hidden_size=128, embedding_size=128,num_directions=2,
                 input_size=2,batch_size=128,initialization_stddev=0.1,mid_size=100):
        super(Critic_Net, self).__init__()
        #Define Embedded
        self.Embed=torch.nn.Linear(input_size, embedding_size, bias=False)
        #Define Encoder
        self.Encoder=torch.nn.LSTM(input_size=embedding_size,hidden_size=hidden_size,batch_first=True,
                              bidirectional=True)
        #Define Attention
        self.W_ref=torch.nn.Linear(num_directions * hidden_size, num_directions * hidden_size, bias=False)
        self.W_q=torch.nn.Linear(num_directions * hidden_size, num_directions * hidden_size, bias=False)
        self.v=torch.nn.Linear(num_directions * hidden_size, 1, bias=True)
        #Define Decoder
        self.Processor=torch.nn.LSTM(input_size=embedding_size*2,hidden_size=hidden_size,batch_first=True,
                              bidirectional=True)
        self.last_layer=torch.nn.Sequential(
            torch.nn.Linear(embedding_size*2, mid_size, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(mid_size,1, bias=True)
        )
    def forward(self, inputs):
        #inputs:(batch,city,2),tensor
        #output:(batch),tensor
        #basic parameter
        city=inputs.size()[1]
        max_time_steps=inputs.size()[1]
        #Embedding
        #inputs:(batch,city,coor)
        City_Embedding=self.Embed(inputs)
        #Encoder
        #City_Embedding:(batch,city,embedding)
        Enc, (hn, cn) = self.Encoder(City_Embedding,None)
        #Attention and Decoder
        #Enc:(batch, city, num_directions * hidden_size)
        #hn: (batch,num_layers * num_directions,  hidden_size)
        #cn: (batch,num_layers * num_directions,  hidden_size)
        processor_input = torch.zeros(Enc.size()[0],1,Enc.size()[2])
        #processor_input:(batch, city, num_directions * hidden_size)
        processor_state = (hn,cn)
        processor_inputs = [processor_input]
        for i in range(max_time_steps):
            processor_output,processor_state=self.Processor(processor_input,processor_state)
            attention_weight=torch.nn.functional.softmax(
                torch.squeeze(
                    self.v(torch.tanh(
                        self.W_ref(Enc)+self.W_q(processor_output.repeat(1,city,1))
                    ))
                ),dim=1
            )
            processor_input=torch.unsqueeze(torch.einsum('ij,ijk->ik',attention_weight,Enc),dim=1)
        output=torch.squeeze(self.last_layer(processor_output))
        return output
def weights_init(m):
    if isinstance(m, torch.nn.LSTM):
        torch.nn.init.uniform_(m.weight_ih_l0.data, a=-0.08, b=0.08)
        torch.nn.init.uniform_(m.weight_hh_l0.data, a=-0.08, b=0.08)
        torch.nn.init.uniform_(m.bias_ih_l0.data, a=-0.08, b=0.08)
        torch.nn.init.uniform_(m.bias_hh_l0.data, a=-0.08, b=0.08)
        torch.nn.init.uniform_(m.weight_ih_l0_reverse.data, a=-0.08, b=0.08)
        torch.nn.init.uniform_(m.weight_hh_l0_reverse.data, a=-0.08, b=0.08)
        torch.nn.init.uniform_(m.bias_ih_l0_reverse.data, a=-0.08, b=0.08)
        torch.nn.init.uniform_(m.bias_hh_l0_reverse.data, a=-0.08, b=0.08)
    else:
        try:
            torch.nn.init.uniform_(m.weight.data, a=-0.08, b=0.08)
            torch.nn.init.uniform_(m.bias.data, a=-0.08, b=0.08)
        except Exception:
            1+1
