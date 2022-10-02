import numpy as np 
import tensorflow as tf 

TAU = 1e-3

def update_target_network(q_network , target_q_network) : 
	for q_weight , target_q_weight in zip(q_network , target_q_network) :
		target_q_weight.assign(TAU * q_weight + (1 - TAU) * target_q_weight) 


