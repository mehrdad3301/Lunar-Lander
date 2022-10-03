import base64
import random 
import imageio 
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import IPython

TAU = 1e-3
MINIBATCH_SIZE = 64
EPSILON_DECAY_RATE = 0.995
MIN_EPSILON = 0.01

def update_target_network(q_network, target_q_network) : 
	for q_weight , target_q_weight in zip(q_network.weights , target_q_network.weights) :
		target_q_weight.assign(TAU * q_weight + (1 - TAU) * target_q_weight) 

def get_action(q_values , epsilon) : 
	
	if np.random.rand() < epsilon : 
		return np.random.randint(q_values.shape[1]) 
	return np.argmax(q_values.numpy()[0]) 

def check_update_conditions(timestep , num_steps_per_update , len_buffer) : 

	return ((timestep + 1) % num_steps_per_update == 0) and \
			(len_buffer > MINIBATCH_SIZE)

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
                                                                 

def print_episode_info(episode , num_avg_points , latest_avg ) : 
		
	end = "\n" if (episode + 1) % 100 == 0 else "\r" 
	print(f"Episode {episode + 1} | Total point average of the last 100 episodes: {latest_avg}" , end=end) 

def plot_history(total_avg_points) : 
	
	
	plt.figure(figsize=(10 , 7)) 
	plt.plot(total_avg_points , color="cyan") 
	plt.plot(get_exp_weighted_avg(total_avg_points)[20:] , color="magenta") 
	plt.xlabel("Episode")
	plt.ylabel("Total Points") 
	plt.gca().set_facecolor("black") 

def get_exp_weighted_avg(points , beta=0.9) : 
	
	avg = [0] 
	for x in points : 
		v = beta * avg[-1] + ( 1 - beta ) * x 
		avg.append(v)
	return avg 	
	
def make_video(filename , env , q_network , fps=30) : 

	with imageio.get_writer(filename , fps=fps) as video : 
		
		state = env.reset() 
		frame = env.render(mode="rgb_array") 	
		video.append_data(frame)	
		done = False 

		while not done :

			q_vals = q_network(state.reshape(1 , -1)) 
			action = get_action(q_vals , epsilon=0.0) 
			state , _ , done , _ = env.step(action)			
			frame = env.render(mode="rgb_array") 
			video.append_data(frame) 
	
