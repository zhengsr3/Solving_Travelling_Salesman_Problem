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
def get_point(batch,city,coor):
    #output:(batch,city,coor),tensor
    return torch.abs(torch.FloatTensor(np.random.normal(size=(batch,city,coor))))
def get_length(point,road):
    #point:(batch,city,coor),tensor
    #road:(batch,city),numpy
    #output:(batch,city),tensor
    try:
        length=torch.zeros(torch.IntTensor(road).size())
    except TypeError:
        length=torch.zeros(torch.LongTensor(road).size())
    batch=length.size()[0]
    city=length.size()[1]
    for i in range(batch):
        for j in range(city):
            if j!=city-1:
                length[i,j]=float(torch.sum(torch.pow(point[i,road[i,j],:]-point[i,road[i,j+1],:],2)))
            else:
                length[i,j]=float(torch.sum(torch.pow(point[i,road[i,j],:]-point[i,road[i,0],:],2)))
    return length
def get_length_sum(point,road):
    #point:(batch,city,coor),tensor
    #road:(batch,city),numpy
    #output:(batch),tensor
    try:
        length=torch.zeros(torch.IntTensor(road).size())
    except TypeError:
        length=torch.zeros(torch.LongTensor(road).size())
    batch=length.size()[0]
    city=length.size()[1]
    for i in range(batch):
        for j in range(city):
            if j!=city-1:
                length[i,j]=float(torch.sqrt(torch.sum(torch.pow(point[i,road[i,j],:]-point[i,road[i,j+1],:],2))))
            else:
                length[i,j]=float(torch.sqrt(torch.sum(torch.pow(point[i,road[i,j],:]-point[i,road[i,0],:],2))))
    return torch.sum(length,dim=1)
def draw(points,roads):
    #point:(batch,city,coor)
    #road:(batch,city)
    batch=min(roads.shape[0],2)
    #print(batch)
    city=len(roads[0])
    fig=plt.figure()
    for j in range(batch):
        ax=plt.subplot(1,batch,j+1)
        point=points[j].numpy()
        road=roads[j]
        for i in range(city-1):
            ax.plot(point[[road[i],road[i+1]],0],point[[road[i],road[i+1]],1],color='b')
        ax.plot(point[[road[city-1],road[0]],0],point[[road[city-1],road[0]],1],color='b')
    fig.show()
def training(ptr_net,critic_net,batch=3,city=5,coors=2,lr_c = 0.001,lr_p = 0.01,beta1=0.5,
             n_baseline_gradient_steps=5,train_steps=20,fix_point=False,loss_sign=1,policy_only=False,
             show_i_time=10,show_j_time=10,show_j=False,draw=True,
             print_training_log=True,log_file_name='train_log.csv',
            save_net=True,save_name='train_',
             shuffle_prob=0):
    critic_op=optim.Adam(critic_net.parameters(), lr=lr_c, betas=(beta1, 0.999))
    ptr_op=optim.Adam(ptr_net.parameters(), lr=lr_p, betas=(beta1, 0.999))
    points_test=get_point(batch,city,coors)
    draw_data_length=[]
    draw_data_loss=[]
    draw_data_critic_loss=[]
    if print_training_log:
        log_file=open(log_file_name,'a')
        log_file.write('iter,ptr_loss,ptr_grad,critic_loss,critic_grad,mean_length\n')
        log_file.close()
    for i in range(train_steps):
        if fix_point==False:
            points=get_point(batch,city,coors)
        else:
            points=points_test
        #points:(batch,city,coors)
        roads=ptr_net.get_road(points)
        if np.random.uniform()<shuffle_prob:
            shuffle=torch.randperm(city)
            city_shuffles=shuffle.numpy()
            points=points[:,city_shuffles,:]
            roads=road_shuffle_inverse(roads,city_shuffles)
        
        #roads:(batch,city)
        real_length=get_length_sum(points,roads)
        if policy_only==False:
            for j in range(n_baseline_gradient_steps):
                est_length=critic_net(points)
                critic_loss=torch.sum(torch.pow(est_length-real_length,2))
                draw_data_critic_loss.append(float(critic_loss))
                critic_net.zero_grad()
                critic_loss.backward(retain_graph=True)
                crit_grad=torch.nn.utils.clip_grad_norm_(critic_net.parameters(),1)
                crit_grad=round(float(crit_grad),4)
                critic_op.step()
                if show_j:
                    if i%show_j_time==0:
                        print('j:'+str(j))
                        print('bsln_error:'+str(critic_loss))
        adv=est_length-real_length
        ptr_loss=loss_sign*torch.dot(ptr_net(points,roads),adv)
        ptr_net.zero_grad()
        ptr_loss.backward(retain_graph=True)
        ptr_grad=torch.nn.utils.clip_grad_norm_(ptr_net.parameters(),1)
        ptr_grad=round(float(ptr_grad),4)
        ptr_op.step()
        roads_test=ptr_net.get_road(points_test)
        draw_data_loss.append(float(
            torch.dot(
                ptr_net(points_test,roads_test),critic_net(points_test)-get_length_sum(points_test,roads_test)
            )
        ))
        draw_data_length.append(float(torch.mean(get_length_sum(points_test,ptr_net.get_road(points_test)))))
        if print_training_log:
            roads_test=ptr_net.get_road(points_test)
            mean_length=torch.mean(get_length_sum(points_test,roads_test))
            mean_length=round(float(mean_length),4)
            log_file=open(log_file_name,'a')
            log_file.write(str(i)+','+str(round(float(ptr_loss),4))+','+str(ptr_grad)+','+
                           str(round(float(critic_loss),4))+','+str(crit_grad)+','+str(mean_length)+'\n')
            log_file.close()
        if i%show_i_time==0:
            print('i:'+str(i))
            roads_test=ptr_net.get_road(points_test)
            print(roads_test[0:2])
            #print(ptr_net.Encoder.state_dict())
            #print(ptr_net.state_dict())
            #for item in ptr_net.Encoder.parameters():
            #    print(item.grad,item.size())
            print('ptr_loss:'+str(torch.dot(
                ptr_net(points_test,roads_test),critic_net(points_test)-get_length_sum(points_test,roads_test)
            )))
            print('mean_length:'+str(torch.mean(get_length_sum(points_test,roads_test))))
            print('bsln_error:'+str(
                torch.sum(torch.pow(critic_net(points_test)-get_length_sum(points_test,roads_test),2))
            ))
            if save_net:
                torch.save(ptr_net, save_name+'ptr.pkl')
                torch.save(critic_net, save_name+'critic.pkl')
    if draw==True:
        fig=plt.figure()
        ax1=plt.subplot(1,2,1)
        ax1.plot(draw_data_length)
        ax2=plt.subplot(1,2,2)
        ax2.plot(draw_data_critic_loss)
        fig.show()
    return (ptr_net,critic_net)
def get_length_sum_single(point,road):
    #point:(city,coor),tensor
    #road:(city),numpy
    #output:number
    point=torch.FloatTensor(point)
    city=point.size()[0]
    length=0
    for j in range(city):
        if j!=city-1:
            length+=float(torch.sqrt(torch.sum(torch.pow(point[road[j],:]-point[road[j+1],:],2))))
        else:
            length+=float(torch.sqrt(torch.sum(torch.pow(point[road[j],:]-point[road[0],:],2))))
    print(length)
    return length
def draw_single(points,roads):
    #point:(city,coor)
    #road:(city)
    city=len(roads)
    fig=plt.figure()
    ax=plt.subplot(1,1,1)
    point=points[j].numpy()
    road=roads[j]
    for i in range(city-1):
        ax.plot(point[[road[i],road[i+1]],0],point[[road[i],road[i+1]],1],color='b')
    ax.plot(point[[road[city-1],road[0]],0],point[[road[city-1],road[0]],1],color='b')
    fig.show()
def city_shuffle(points):
    '''
    points:(batch,city,coor)
    shuffle along the city dimension
    each example in batch shuffle in a different way
    '''
    batch_size=points.size()[0]
    city=points.size()[1]
    for i in range(batch_size):
        shuffle=torch.randperm(city)
        points[i,:,:]=points[i,shuffle,:]
    return points
def adjust_road(road_shuffle,point_shuffle,point):
    '''
    road_shuffle:(city),numpy
    point_shuffle:(city,coor),tensor
    point:(city,coor),tensor
    '''
    city=road_shuffle.shape[0]
    road=[]
    for i in range(city):
        point_now=point_shuffle[road_shuffle[i]]
        for j in range(city):
            if torch.all(torch.eq(point_now,point[j])):
                road.append(j)
                break
    return np.array(road)
def active_search(ptr_net,point,road=None,iter_time=300,batch_size=300,
                  lr_p = 0.01,beta1=0.9,alpha=0.01,loss_sign=1,alpha_decay=0.9,
                 plot_comp=True,plot_mean=False,
                  print_searching_log=True,log_file_name='search_log.csv',
                 save_net=True,save_name='search_',
                  shuffle_prob=0):
    '''
    searching for shortest road for a particular points distribution
    point:(city,coor),tensor
    road:(city),numpy
    '''
    ptr_op=optim.Adam(ptr_net.parameters(), lr=lr_p, betas=(beta1, 0.999))
    point_copy=torch.unsqueeze(point,0).repeat(batch_size,1,1)
    if road is None:
        road=ptr_net.get_road(point_copy)[0]
    city=road.shape[0]
    road=torch.IntTensor(road)
    road_copy=torch.unsqueeze(road,0).repeat(batch_size,1)
    road_best=road_copy[0]
    #point_copy:(batch,city,coor),tensor
    #road_copy:(batch,city),tensor
    #road_best:(city)
    length_best=get_length_sum(point_copy,road_copy)[0]
    #length_best:number
    baseline=length_best
    mean=[]
    if print_searching_log:
        log_file=open(log_file_name,'a')
        log_file.write('iter,ptr_loss,ptr_grad,best,mean_length\n')
        log_file.close()
    #length_best:number
    for i in range(iter_time):
        point_input=city_shuffle(point_copy)
        road_output=ptr_net.get_road(point_input)
        if np.random.uniform()<shuffle_prob:
            shuffle=torch.randperm(city)
            city_shuffles=shuffle.numpy()
            point_input=point_input[:,city_shuffles,:]
            road_output=road_shuffle_inverse(road_output,city_shuffles)
        length_all=get_length_sum(point_input,road_output)
        #length_all:(batch)
        j=torch.argmin(length_all)
        if length_all[j]<length_best:
            length_best=length_all[j]
            road_shuffle=road_output[j,:]
            point_shuffle=point_input[j]
            #print(get_length_sum_single(point_shuffle,road_shuffle))
            road_best=adjust_road(road_shuffle,point_shuffle,point)
        adv=baseline-length_all
        ptr_loss=loss_sign*torch.dot(ptr_net(point_input,road_output),adv)
        ptr_net.zero_grad()
        ptr_loss.backward(retain_graph=True)
        ptr_grad=torch.nn.utils.clip_grad_norm_(ptr_net.parameters(),1)
        ptr_op.step()
        if i%10==0:
            print(i)
            if save_net:
                torch.save(ptr_net, save_name+'ptr.pkl')
        mean.append(float((torch.mean(length_all))))
        if print_searching_log:
            log_file=open(log_file_name,'a')
            log_file.write(str(i)+','+str(float(ptr_loss))+','+
                           str(float(ptr_grad))+
                           ','+str(float(length_all[j]))+','+str(float((torch.mean(length_all))))+'\n')
            log_file.close()
        baseline=baseline*alpha_decay+(1-alpha_decay)*torch.mean(length_all)
    fig=plt.figure()
    if plot_comp==True:
        point=point.numpy()
        fig1,ax=plt.subplots(1,2)
        ax_init=ax[0]
        ax_init.set_title('init:'+str(round(float(get_length_sum(point_copy,road_copy)[0]),4)), 
                          fontsize=14, fontweight='bold')
        for i in range(city-1):
            ax_init.plot(point[[road[i],road[i+1]],0],point[[road[i],road[i+1]],1],color='b')
        ax_init.plot(point[[road[city-1],road[0]],0],point[[road[city-1],road[0]],1],color='b')
        road_best_copy=torch.unsqueeze(torch.IntTensor(road_best),0).repeat(batch_size,1)
        ax_after=ax[1]
        ax_after.set_title('after:'+str(round(float(get_length_sum_single(point,road_best)),4)), 
                          fontsize=14, fontweight='bold')
        for i in range(city-1):
            ax_after.plot(point[[road_best[i],road_best[i+1]],0],point[[road_best[i],road_best[i+1]],1],color='b')
        ax_after.plot(point[[road_best[city-1],road_best[0]],0],point[[road_best[city-1],road_best[0]],1],color='b')
    elif plot_mean==True:
        plt.plot(mean)
    fig.show()
    return {
        'ptr_net':ptr_net,
        'mean':mean,
        'road_best':road_best,
        'length_best':length_best
    }
def road_shuffle_inverse(road,shuffle):
    road_new=np.zeros(road.shape)
    city=len(shuffle)
    for i in range(city):
        j=shuffle[i]
        road_new[road==shuffle[i]]=j
    road_new=road_new.astype(int)
    return road_new
