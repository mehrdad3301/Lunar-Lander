import random 
import numpy as np 
import tensorflow as tf 

TAU = 1e-3
MINIBATCH_SIZE = 64
EPSILON_DECAY_RATE = 0.995
MIN_EPSILON = 0.01
def update_target_network(q_network , target_q_network) : 
	for q_weight , target_q_weight in zip(q_network , target_q_network) :
		target_q_weight.assign(TAU * q_weight + (1 - TAU) * target_q_weight) 


def get_action(q_values , epsilon) : 
	
	if np.random.rand() < epsilon : 
		return np.random.randint(q_values.shape[1]) 
	return np.argmax(q_values.numpy()[0]) 

def check_update_conditions(timestep , num_steps_per_update , len_buffer) : 

	return (timestep + 1) % num_steps_per_update == 0 and \
			len_buffer > MINIBATCH_SIZE 

def get_experiences(memory_buffer) : 
	
	experiences = random.sample(memory_buffer , MINIBATCH_SIZE) 
	states = tf.convert_to_tensor(np.array([e.state for e in experiences if e is not None]),dtype=tf.float32)                                         
	actions = tf.convert_to_tensor(np.array([e.action for e in experiences if e is not None]), dtype=tf.float32)                                      
	rewards = tf.convert_to_tensor(np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32)                                      
	next_states = tf.convert_to_tensor(np.array([e.next_state for e in experiences if e is not None]),dtype=tf.float32)                               
	done_vals = tf.convert_to_tensor(np.array([e.done for e in experiences if e is not None]).astype(np.uint8),                                       
							dtype=tf.float32)                                                                                                

	return (states, actions, rewards, next_states, done_vals)                                                                                         

def get_new_epsilon(epsilon) : 
	return max(MIN_EPSILON , EPSILON_DECAY_RATE * epsilon) 
                                                                 

def print_episode_info(episode , latest_avg , num_avg_points) : 
		
	end = "\n" if (episode + 1) % 100 == 0 else "\r" 
	print(f"Episode {episode} | Total point average of the last 100 episodes: {latest_avg}" , end=end) 

