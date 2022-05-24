#!/home/wanghy/anaconda3/envs/mmd/bin/python

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random

def TargetFunc(x):
	return (x-1)*(x-25)

class SimpleNet(nn.Module):
	def __init__(self):
		super(SimpleNet,self).__init__()#定义好每一层，其参数变量会进入parameters中，以供optim下的优化器调参
		self.fc1=nn.Linear(1,10)
		self.relu=nn.LeakyReLU()
		self.fc2=nn.Linear(10,1)

	def forward(self,x):#定义网络的结构，即前馈计算过程
		out=self.fc1(x)
		out=self.relu(out)
		return self.fc2(out)

class CustomNet(nn.Module):#通过带梯度的变量自定义网络
	def __init__(self):
		super(CustomNet,self).__init__()
		self.i2h=Variable(torch.randn((1,10)),requires_grad=True)
		self.h2o=Variable(torch.randn((10,1)),requires_grad=True)
		self.bias1=Variable(torch.randn((1,10)),requires_grad=True)
		self.bias2=Variable(torch.randn((1,1)),requires_grad=True)
		self.relu=nn.LeakyReLU()
		self.optim=torch.optim.SGD([self.i2h,self.h2o,self.bias1,self.bias2],lr=0.0001)
		# self.weights=Variable(torch.randn((3,1)),requires_grad=True)
		# self.optim=torch.optim.SGD([self.weights,],lr=0.5)

	def forward(self,x):
		out=self.relu(x.mm(self.i2h)+self.bias1)
		ret=out.mm(self.h2o)+self.bias2
		# xs=Variable(torch.Tensor([x.pow(2),x,1]).reshape((1,3)))
		# ret=xs.mm(self.weights)
		return ret

	def learn(self,x,y0):
		self.optim.zero_grad()
		y=self.forward(x)
		loss=(y-y0).pow(2)
		# print('loss=',float(loss.detach()))
		loss.backward()
		self.optim.step()


if __name__=='__main__':
	print('original values:')
	result0=[TargetFunc(x) for x in range(100)]
	print(result0)
	print('simple net:')
	net1=SimpleNet()
	param=net1.parameters()
	print('params:',param)
	crit=nn.MSELoss()
	optim=torch.optim.SGD(net1.parameters(),lr=0.5)
	for i in range(1000):
		xs=np.array([random.randint(0,101) for j in range(10)])
		ys=Variable(torch.Tensor((TargetFunc(xs)+random.gauss(0.3,0.1))/10000).reshape((len(xs),1)))
		optim.zero_grad()
		ys0=net1(Variable(torch.Tensor(xs/100).reshape((len(xs),1))))
		loss=crit(ys,ys0)#torch.sum((ys-ys0)*(ys-ys0))/len(xs)
		loss.backward()
		optim.step()
	result1=[float(net1(torch.ones((1,))*x/100).detach())*10000 for x in range(100)]
	print(result1)
	print('custom net:')
	net2=CustomNet()
	param=net2.parameters()
	print('params:',param)
	for i in range(10000):
		x=random.randint(0,101)
		vy0=Variable(torch.ones((1,1))*(TargetFunc(x)+random.gauss(0.3,0.1))/10000)
		vx=Variable(torch.ones((1,1))*x/100)
		net2.learn(vx,vy0)
	ys=[float(net2(torch.ones((1,1))*x).detach())*10000 for x in range(100)]
	print(ys)