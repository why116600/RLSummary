import numpy as np
import torch
import gym
from torch.autograd import Variable

HIDDEN_SIZE=64

class DDPGActor(torch.nn.Module):
	def __init__(self,state_n,action_n):
		super(DDPGActor,self).__init__()
		self.input_layer=torch.nn.Linear(state_n,HIDDEN_SIZE)
		self.hidden_layer=torch.nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE)
		self.output_layer=torch.nn.Linear(HIDDEN_SIZE,action_n)
		
	def forward(self,state):
		if type(state)!=torch.Tensor:
			state=torch.Tensor(state)
		y=self.input_layer(state)
		y=torch.relu(y)
		y=self.hidden_layer(y)
		y=torch.relu(y)
		y=self.output_layer(y)
		
		return torch.softmax(y,len(y.shape)-1)


class DDPGCritic(torch.nn.Module):
	def __init__(self,state_n,action_n):
		super(DDPGCritic,self).__init__()
		self.input_layer=torch.nn.Linear(state_n+action_n,HIDDEN_SIZE)
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
		return y

class EMA:
	def __init__(self,model,optim):
		self.model=model
		self.optim=optim
		self.old_para={}

	def snapshot(self):
		para1=self.model.named_parameters()
		for name,para in para1:
			self.old_para[name]=para.data

	def update(self,rou):
		self.optim.step()
		self.optim.zero_grad()
		para1=self.model.named_parameters()
		for name,para in para1:
			para.data=rou*para.data+(1-rou)*self.old_para[name]

def EMA_update(net1,optim,rou):
	para1=net1.named_parameters()
	old_para={}
	for name,para in para1:
		old_para[name]=para.data
	optim.step()
	optim.zero_grad()
	para1=net1.named_parameters()
	for name,para in para1:
		para.data=rou*para.data+(1-rou)*old_para[name]

class DDPGAgent:
	def __init__(self,state_n,action_n):
		self.state_n=state_n
		self.action_n=action_n
		self.actor=DDPGActor(state_n,action_n)
		self.critic=DDPGCritic(state_n,action_n)
		self.actor_optim=torch.optim.Adam(self.actor.parameters())
		self.critic_optim=torch.optim.Adam(self.critic.parameters())
		self.actor_ema=EMA(self.actor,self.actor_optim)
		self.critic_ema=EMA(self.critic,self.critic_optim)
		self.states=[]
		self.actions=[]
		self.states_=[]
		self.rewards=[]
		self.ends=[]

	def add_exp(self,s,a,r,s_,done):
		self.states.append(s)
		self.actions.append(a)
		self.rewards.append(np.ones((1,))*r)
		self.states_.append(s_)
		if done:
			self.ends.append(np.zeros((1,)))
		else:
			self.ends.append(np.ones((1,)))

	def get_exp_len(self):
		return len(self.states)

	def learn(self,batch_size,rou=0.99):
		assert len(self.states)>=batch_size
		indices=np.random.choice(len(self.states),size=batch_size)
		t_states=torch.Tensor(np.array(self.states)[indices])
		t_actions=torch.Tensor(np.array(self.actions)[indices])
		t_rewards=torch.Tensor(np.array(self.rewards)[indices])
		t_next=torch.Tensor(np.array(self.states_)[indices])
		t_end=torch.Tensor(np.array(self.ends)[indices])

		#actor_=DDPGActor(self.state_n,self.action_n)
		#actor_.load_state_dict(self.actor.state_dict())
		#critic_=DDPGCritic(self.state_n,self.action_n)
		#critic_.load_state_dict(self.critic.state_dict())

		a_=self.actor(t_next).detach()#actor_(t_next)
		q_=self.critic(torch.cat((t_next,a_),1)).detach()#critic_(torch.cat((t_next,a_),1))
		y=t_rewards+0.99*q_*t_end
		q=self.critic(torch.cat((t_states,t_actions),1))
		td_error=torch.square(y-q).mean()
		self.critic_optim.zero_grad()
		td_error.backward()
		EMA_update(self.critic,self.critic_optim,rou)

		a=self.actor(t_states)
		q=self.critic(torch.cat((t_states,a),1))
		loss=-q.mean()
		loss.backward()
		EMA_update(self.actor,self.actor_optim,rou)


	def get_action(self,state):
		a=self.actor(state).detach()
		action=np.random.choice(env.action_space.n,p=a.numpy())
		return action

	def explore(self,env,first):
		probs=self.actor(first).detach()
		action=np.random.choice(env.action_space.n,p=probs.numpy())
		second,r,done,_=env.step(action)
		self.add_exp(first,probs.numpy(),r,second,done)
		return second,r,done

	def save_actor(self,path):
		torch.save(self.actor,path)

def main():
	env=gym.make('CartPole-v1')
	agent=DDPGAgent(env.observation_space.shape[0],env.action_space.n)
	new_train=True
	max_rewards=0.0
	for i in range(1000):
		first=env.reset()
		
		rewards=0.0
		done=False
		while not done:
			second,r,done=agent.explore(env,first)
			if agent.get_exp_len()>=100:
				agent.learn(100)
			first=second
			rewards+=r
		if new_train:
			max_rewards=rewards
			new_train=False
			print('first rewards:',rewards)
		elif rewards>max_rewards:
			max_rewards=rewards
			print('got breakthrough ',rewards,'at round ',i)
		elif rewards>100:
			print('got reward ',rewards,'at round ',i)

if __name__=='__main__':
	main()