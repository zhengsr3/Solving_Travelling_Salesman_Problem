# Reinforcement_Learning_Pointer_Networks_TSP_Pytorch
​	The algorithm and network architect are designed as the following paper said:

​	 https://arxiv.org/abs/1611.09940

​	This repository is consisted of two part:TSP_pytoch.py contain the defination of networks and function for envioronment , training,and search. Reinforcement_Learning_Pointer_Networks_TSP_Pytorch_visuallization.ipynb use those function and visualizing the outcome.

​	There are two network used in the procedure: policy network and critic network.Policy networks is a pointer networks. It use LSTM as Encoder,use LSTM and attention as Decoder.Besides this,it has a module which take all the outcome from encoder and an output from decoder to compute(using linear layer and softmax) a probability distribution over input,which indicate which input we will choose in the next step.We use this prob to decide the city to go. Critic network  use LSTM as Encoder,use LSTM and attention as Decoder and two fc layer to estimate the length saleman need to go through.

​	The train function mainly take policy network critic network and the number of city as input and train two networks by turns.

​	The active_search take policy network and coordinate of cities as the main input and use policy-gradient to guide the searching.

​	Some function like draw, get_length take coordinates and visiting plan as input and return a picture or the length of plan. These function act as the environment.

​	The version of pytorch used in this py is 1.0.

​	If you have any question or think that there are bug in these code,please feel free to contact me. Thank you very much.