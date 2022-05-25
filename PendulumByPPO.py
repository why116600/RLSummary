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
		self.hidden_layer=torch.nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE)
		self.output_layer=torch.nn.Linear(HIDDEN_SIZE,action_n)
		self.logstd=Variable(torch.zeros(action_n),True)
		
	def forward(self,state):
		if type(state)!=torch.Tensor:
			state=torch.Tensor(state)
		y=self.input_layer(state)
		y=torch.relu(y)
		y=self.hidden_layer(y)
		y=torch.relu(y)
		y=self.output_layer(y)
		return y


	
class CriticNet(torch.nn.Module):
	def __init__(self,state_n):
		super(CriticNet,self).__init__()
		self.input_layer=torch.nn.Linear(state_n,HIDDEN_SIZE)
		self.hidden_layer=torch.nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE)
		self.output_layer=torch.nn.Linear(HIDDEN_SIZE,1)

	def forward(self,state):
		if type(state)!=torch.Tensor:
			state=torch.Tensor(state)
		y=self.input_layer(state)
		y=torch.relu(y)
		y=self.hidden_layer(y)
		y=torch.relu(y)
		y=self.output_layer(y)
		return y#torch.softmax(y,len(y.shape)-1)

	def learn(self,optim,advan):
		if type(advan)!=list:
			advan=[advan,]
		for adv in advan:
			loss=adv**2
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

	def get_action(self,state):
		means=self.actor(state)
		std=torch.exp(self.actor.logstd)
		pi=torch.distributions.Normal(means,std)
		action=pi.sample()#torch.normal(means,std)
		return action,pi.log_prob(action)

	def update(self,states,states_,actions,rewards,end_flags,step_n):
		t_states=torch.Tensor(states)
		t_next=torch.Tensor(states_)
		t_actions=torch.unsqueeze(torch.Tensor(actions),1)
		t_rewards=torch.unsqueeze(torch.Tensor(rewards),1)
		t_ends=torch.unsqueeze(torch.Tensor(end_flags),1)
		means=self.actor(t_states)
		stds=torch.ones_like(means)*torch.exp(self.actor.logstd)
		t_pis=torch.distributions.Normal(means,stds)
		t_logpi=t_pis.log_prob(t_actions).data
		t_values=self.critic(t_states).data
		t_advan=t_rewards-t_values+self.GAMMA*self.critic(t_next).detach()*t_ends
		for i in range(step_n):
			means=self.actor(t_states)
			stds=torch.ones_like(means)*torch.exp(self.actor.logstd)
			pis=torch.distributions.Normal(means,stds)
			logpi=pis.log_prob(t_actions)
			ratio=torch.exp(logpi-t_logpi)
			loss=-torch.min(ratio*t_advan,torch.clip(ratio,1-EBSILON,1+EBSILON)*t_advan)#-logpi*t_advan#
			loss=loss.mean()
			loss.backward()
			self.actor_optim.step()
			self.actor_optim.zero_grad()
			t_error=t_rewards-self.critic(t_states)+self.GAMMA*self.critic(t_next)*t_ends
			loss=torch.square(t_error)
			loss=loss.mean()
			loss.backward()
			self.critic_optim.step()
			self.critic_optim.zero_grad()

	def train(self,env,episode_n,batch_n,step_n=10):
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
				action,_=self.get_action(first)
				a=action.detach()
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
			self.update(states,states_,actions,rewards,end_flags,step_n)
		return first,all,round,maxscore

def main():
	env=gym.make('Pendulum-v0')
	agent=PPOAgent(env.observation_space.shape[0],env.action_space.shape[0])
	state=None
	first_train=True
	maxrewards=0.0
	for i in range(100):
		done=False
		#env.reset()
		state,scores,n,maxscore=agent.train(env,200,1,1)
		#while not done:
			#state,reward,done,info=env.step(np.random.randn(env.action_space.shape[0]))
			#state,reward,done=agent.train(env,20,state)
			#rewards+=reward
		#state=None
		print('average rewards:',scores/n,'at round',i)
		if first_train:
			maxrewards=maxscore
			first_train=False
		if maxscore>maxrewards:
			maxrewards=maxscore
			print('max rewards:',maxrewards,'at round ',i)

if __name__=='__main__':
	main()