import torch
import torch.nn as nn

def main():
	c=nn.Conv2d(3,10,3,padding=1)
	m1=torch.randn((1,3,5,6))
	m2=c(m1)
	print('m2 shape:',m2.shape)
	m3=torch.flatten(m2,2,3)
	print('m3 shape:',m3.shape)

if __name__=='__main__':
	main()