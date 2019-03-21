import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from TSP_env import *
def training(ptr_net,critic_net,
             batch=3,city=5,coors=2,lr_c = 0.001,lr_p = 0.01,beta1=0.5,
             n_baseline_gradient_steps=5,train_steps=20,
             show_i_time=10,show_j_time=10,show_j=False,draw=True,
             print_training_log=True,log_file_name='train_log.csv',
            save_net=True,save_name='train_'):
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
        points=get_point(batch,city,coors)
        #points:(batch,city,coors)
        roads=ptr_net.get_road(points)
        
        #roads:(batch,city)
        real_length=get_length_sum(points,roads)
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
        ptr_loss=torch.dot(ptr_net(points,roads),adv)
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
                  lr_p = 0.01,beta1=0.9,alpha=0.01,alpha_decay=0.9,
                 plot_comp=True,plot_mean=False,
                  print_searching_log=True,log_file_name='search_log.csv',
                 save_net=True,save_name='search_'):
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
    length_init=get_length_sum(point_copy,road_copy)[0]
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
        ptr_loss=torch.dot(ptr_net(point_input,road_output),adv)
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
        ax_init.set_title('init:'+str(round(float(length_init),4)), 
                          fontsize=14, fontweight='bold')
        for i in range(city-1):
            ax_init.plot(point[[road[i],road[i+1]],0],point[[road[i],road[i+1]],1],color='b')
        ax_init.plot(point[[road[city-1],road[0]],0],point[[road[city-1],road[0]],1],color='b')
        road_best_copy=torch.unsqueeze(torch.IntTensor(road_best),0).repeat(batch_size,1)
        ax_after=ax[1]
        ax_after.set_title('after:'+str(round(float(get_length_sum(point,road_best)),4)), 
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
def learning_rate_adjusting_train(ptr_net,critic_net
            ,batch=128,city=20,coors=2,lr_c_init = 0.001
               ,lr_p_init = 0.001,beta1=0.9,n_baseline_gradient_steps=1,
               train_steps_per_iter=1000,train_iter=10,decay=0.99,
               log_file_name='train_log_decay.csv',
             show_i_time=10,show_j_time=10,show_j=False,draw=True,
             print_training_log=True,save_net=True,save_name='train_deacy_'):
    critic_op=optim.Adam(critic_net.parameters(), lr=lr_c_init, betas=(beta1, 0.999))
    ptr_op=optim.Adam(ptr_net.parameters(), lr=lr_p_init, betas=(beta1, 0.999))
    scheduler_cr = torch.optim.lr_scheduler.StepLR(critic_op, step_size=train_steps_per_iter, gamma=decay)
    scheduler_ptr = torch.optim.lr_scheduler.StepLR(ptr_op, step_size=train_steps_per_iter, gamma=decay)
    points_test=get_point(batch,city,coors)
    draw_data_length=[]
    draw_data_loss=[]
    draw_data_critic_loss=[]
    if print_training_log:
        log_file=open(log_file_name,'a')
        log_file.write('iter,ptr_loss,ptr_grad,critic_loss,critic_grad,mean_length\n')
        log_file.close()
    for i in range(train_iter*train_steps_per_iter):
        scheduler_cr.step()
        scheduler_ptr.step()
        points=get_point(batch,city,coors)
        #points:(batch,city,coors)
        roads=ptr_net.get_road(points)
        
        #roads:(batch,city)
        real_length=get_length_sum(points,roads)
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
        ptr_loss=torch.dot(ptr_net(points,roads),adv)
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
                torch.save(ptr_net, save_name+'ptr'+str(int(i/train_steps_per_iter))+'.pkl')
                torch.save(critic_net, save_name+'critic'+str(int(i/train_steps_per_iter))+'.pkl')
    if draw==True:
        fig=plt.figure()
        ax1=plt.subplot(1,2,1)
        ax1.plot(draw_data_length)
        ax2=plt.subplot(1,2,2)
        ax2.plot(draw_data_critic_loss)
        fig.show()
    return (ptr_net,critic_net)
def ReduceLROnPlateau_train(ptr_net,critic_net
            ,batch=128,city=20,coors=2,lr_c_init = 0.001
               ,lr_p_init = 0.001,beta1=0.9,n_baseline_gradient_steps=1,
               max_train_steps=10000,min_lr=1e-8,decay=0.99,train_steps_per_iter=1000,
               log_file_name='train_log_rlop.csv',
             show_i_time=10,show_j_time=10,show_j=False,draw=True,
             print_training_log=True,save_net=True,save_name='train_rlop_'):
    critic_op=optim.Adam(critic_net.parameters(), lr=lr_c_init, betas=(beta1, 0.999))
    ptr_op=optim.Adam(ptr_net.parameters(), lr=lr_p_init, betas=(beta1, 0.999))
    scheduler_ptr = torch.optim.lr_scheduler.ReduceLROnPlateau(ptr_op,factor=decay)
    points_test=get_point(batch*3,city,coors)
    draw_data_length=[]
    draw_data_loss=[]
    draw_data_critic_loss=[]
    if print_training_log:
        log_file=open(log_file_name,'a')
        log_file.write('iter,ptr_loss,ptr_grad,critic_loss,critic_grad,mean_length\n')
        log_file.close()
    for i in range(max_train_steps):
        points=get_point(batch,city,coors)
        #points:(batch,city,coors)
        roads=ptr_net.get_road(points)
        
        #roads:(batch,city)
        real_length=get_length_sum(points,roads)
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
        ptr_loss=torch.dot(ptr_net(points,roads),adv)
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
                torch.save(ptr_net, save_name+'ptr'+str(int(i/train_steps_per_iter))+'.pkl')
                torch.save(critic_net, save_name+'critic'+str(int(i/train_steps_per_iter))+'.pkl')
        scheduler_ptr.step(mean_length)
    if draw==True:
        fig=plt.figure()
        ax1=plt.subplot(1,2,1)
        ax1.plot(draw_data_length)
        ax2=plt.subplot(1,2,2)
        ax2.plot(draw_data_critic_loss)
        fig.show()
    return (ptr_net,critic_net)
