import numpy as np
import torch
import gym
from torch.autograd import Variable

HIDDEN_SIZE=64
EBSILON=0.2

class PPOActor(torch.nn.Module):
	def __init__(self,state_n,action_n):
		super(PPOActor,self).__init__()
		self.input_layer=torch.nn.Linear(state_n,HIDDEN_SIZE)
		#self.hidden_layer=torch.nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE)
		self.output_layer=torch.nn.Linear(HIDDEN_SIZE,action_n)
		
	def forward(self,state):
		if type(state)!=torch.Tensor:
			state=torch.Tensor(state)
		y=self.input_layer(state)
		y=torch.relu(y)
		#y=self.hidden_layer(y)
		#y=torch.relu(y)
		y=self.output_layer(y)
		return torch.softmax(y,len(y.shape)-1)


	
class CriticNet(torch.nn.Module):
	def __init__(self,state_n):
		super(CriticNet,self).__init__()
		self.input_layer=torch.nn.Linear(state_n,HIDDEN_SIZE)
		#self.hidden_layer=torch.nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE)
		self.output_layer=torch.nn.Linear(HIDDEN_SIZE,1)

	def forward(self,state):
		if type(state)!=torch.Tensor:
			state=torch.Tensor(state)
		y=self.input_layer(state)
		y=torch.relu(y)
		#y=self.hidden_layer(y)
		#y=torch.relu(y)
		y=self.output_layer(y)
		return y#torch.softmax(y,len(y.shape)-1)

	def learn(self,state,G,optim):
		loss=(self.forward(state)-G)**2
		loss.backward()
		optim.step()
		optim.zero_grad()

class PPOAgent:
	def __init__(self,state_n,action_n):
		self.actor=PPOActor(state_n,action_n)
		self.critic=CriticNet(state_n)
		self.state_n=state_n
		self.action_n=action_n
		self.actor_optim=torch.optim.Adam(self.actor.parameters())
		self.critic_optim=torch.optim.Adam(self.critic.parameters())
		self.GAMMA=0.99
		
	def GetAction(self,state):
		pi=self.actor(state)
		return np.random.choice(self.action_n,p=pi.data.numpy())

	#攒够固定步长的轨迹再进行更新，即使中间有终结，也reset后继续收集
	def update(self,states,states_,actions,rewards,end_flags):
		t_states=torch.Tensor(states)
		t_next=torch.Tensor(states_)
		t_actions=torch.unsqueeze(torch.LongTensor(actions,),1)
		t_rewards=torch.unsqueeze(torch.Tensor(rewards),1)
		t_ends=torch.unsqueeze(torch.Tensor(end_flags),1)
		t_pis=self.actor(t_states).data
		t_pi=t_pis.gather(-1,t_actions)
		t_G=torch.zeros(len(states),1)
		t_values=self.critic(t_states).data
		t_advan=t_rewards-t_values+self.GAMMA*self.critic(t_next).detach()*t_ends
		G=0.0
		#for i in range(len(states)-1,-1,-1):
			#G=rewards[i]+self.GAMMA*G
			#t_G[i,0]=G
		#对一组轨迹数据进行一定次数的子迭代
		for i in range(10):
			pi=self.actor(t_states).gather(-1,t_actions)
			ratio=torch.exp(torch.log(pi)-torch.log(t_pi))#pi/t_pi
			loss=-torch.min(ratio*t_advan,torch.clamp(ratio,1-EBSILON,1+EBSILON)*t_advan)#-torch.log(pi)*t_advan#
			loss=loss.mean()
			loss.backward()
			self.actor_optim.step()
			self.actor_optim.zero_grad()
			t_error=t_rewards-self.critic(t_states)+self.GAMMA*self.critic(t_next)*t_ends
			loss=torch.square(t_error)#torch.square(self.critic(t_states)-t_G)#
			loss=loss.mean()
			loss.backward()
			self.critic_optim.step()
			self.critic_optim.zero_grad()

	def rollout(self,env,episode_n,batch_n):
		first=env.reset()
		scores=0
		all=0
		round=0
		maxscore=0
		for i in range(episode_n):
			states=[]
			states_=[]
			actions=[]
			rewards=[]
			end_flags=[]
			for j in range(batch_n):
				a=self.GetAction(first)
				second,reward,done,_=env.step(a)
				states.append(first)
				states_.append(second)
				actions.append(a)
				rewards.append(reward)
				scores+=reward
				if done:
					round+=1
					all+=scores
					if scores>maxscore:
						maxscore=scores
					scores=0
					first=env.reset()
					end_flags.append(0.0)
				else:
					first=second
					end_flags.append(1.0)
			self.update(states,states_,actions,rewards,end_flags)
		return all,round,maxscore
	

def main():
	env=gym.make('CartPole-v1')
	agent=PPOAgent(4,2)
	maxstep=0
	for round in range(50):
		score,n,maxscore=agent.rollout(env,100,10)
		print('got average score ',score/n,'and max score',maxscore,' at round ',round)
		if (score/n)>400.0 and maxscore>400.0:
			break
		#if step>maxstep:
			#maxstep=step
			#print('got breakthrough[',step,'] at ',round)
		#elif step>100:
			#print('got score',step,'at ',round)
	print('start test')
	allscore=0
	for i in range(10):
		score=0
		first=env.reset()
		done=False
		while not done:
			action=agent.GetAction(first)
			second,reward,done,_=env.step(action)
			first=second
			score+=reward
		print('got score:',score)
		allscore+=score
		score=0
	print('the end,averge:',allscore/10)

if __name__=='__main__':
	main()