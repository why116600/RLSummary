import torch
import torch.nn as nn
import gym
import numpy as np


class BaseNet(nn.Module):#定义多层感知机网络进行强化学习
	def __init__(self,in_size,out_size,linear):
		super(BaseNet,self).__init__()
		self.i2h=nn.Linear(in_size,40)
		self.h2o=nn.Linear(40,out_size)
		self.linear=linear
		self.relu=nn.ReLU()
		if not linear:
			self.activate=nn.Softmax()

	def forward(self,x):
		out=self.i2h(x)
		out=self.relu(out)
		out=self.h2o(out)
		if not self.linear:
			out=self.activate(out)
		return out

class A2CAgent:#基于优势Actor Critic算法进行训练的智能体
	def __init__(self,state_n,action_n,gamma=0.99):
		self.critic=BaseNet(state_n,1,True)
		self.actor=BaseNet(state_n,action_n,False)
		self.c_loss=nn.MSELoss()
		self.a_loss=nn.CrossEntropyLoss()
		self.c_optim=torch.optim.Adam(self.critic.parameters(),lr=0.001)
		self.a_optim=torch.optim.Adam(self.actor.parameters(),lr=0.001)
		self.gamma=gamma
		self.lamb=1.0

	def get_action(self,state):#按照actor网络提供的概率分布随机取动作值
		ts=torch.Tensor(state)
		ta=self.actor(ts)
		action=ta.detach().numpy()
		return np.random.choice(len(action),p=action)#np.argmax(action)

	def get_value_tensor(self,state):#计算state的状态价值
		ts=torch.Tensor(state)
		return self.critic(ts)
	
	def learn(self,state,action,reward,next):
		ts1=torch.Tensor(state)
		value=self.get_value_tensor(state)
		#计算优势函数和价值函数的目标值
		advan=torch.ones((1,))*reward-self.get_value_tensor(state)
		# next_value=torch.ones((1,))*reward
		if not next is None:
			advan+=self.gamma*self.get_value_tensor(next)
			# next_value+=self.gamma*self.get_value_tensor(next)
		ta=self.actor(ts1)
		#计算两个网络的损失函数
		a_loss=-ta[action].log()*advan.detach()*self.lamb
		c_loss=advan**2#self.c_loss(value,next_value)
		# 控制衰减系数
		if next is None:
			self.lamb=1.0
		else:
			self.lamb*=0.95			
		# print('a_loss=',float(a_loss[0].detach()),',c_loss=',float(c_loss[0].detach()))
		#按梯度更新网络
		self.a_optim.zero_grad()
		a_loss.backward()
		self.a_optim.step()
		self.c_optim.zero_grad()
		c_loss.backward()
		self.c_optim.step()


EPOCH=10000

def main():
	env=gym.make('CartPole-v1')
	agent=A2CAgent(4,2)
	for i in range(EPOCH):
		done=False
		state=env.reset()
		whole_reward=0
		while not done:
			action=agent.get_action(state)
			next,reward,done,_=env.step(action)
			if done:
				next=None
			agent.learn(state,action,reward,next)
			if not done:
				state=next
			whole_reward+=reward
		print('epoch',i,' got reward:',whole_reward)

if __name__=='__main__':
	main()

