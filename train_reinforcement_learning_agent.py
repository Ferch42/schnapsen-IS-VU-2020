
from api import State, util
#  from bots.rdeep import rdeep
from bots.dqn.dqn import features

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from keras.models import load_model

import numpy as np
import random	
import sys

number_of_episodes = 100000

opponent = util.load_player('rdeep')

possible_moves = [(None, None)]
possible_moves += [(x,None) for x in range(20)]
possible_moves+= [(None,x) for x in range(20)]
possible_moves += [(2,3),(3,2),(7,8),(8,7),(12,13),(13,12),(17,18),(18,17)]

possible_moves_dict = dict()

print(possible_moves)
for i, move in enumerate(possible_moves):
	possible_moves_dict[move] = i

def move_hot_form(move):
	
	hot_form = [0 for _ in range(len(possible_moves))]
	hot_form[possible_moves_dict[move]] =1
	
	return hot_form


Qnet = Sequential()
Qnet.add(Dense(64, input_dim = 154+len(possible_moves), activation = 'relu'))
Qnet.add(Dense(64, activation = 'relu'))
Qnet.add(Dense(64, activation = 'relu'))
Qnet.add(Dense(1, activation = 'linear'))
rms = optimizers.RMSprop(lr=0.00025)

Qnet = load_model('Qnet.h5')
print('loading modellll')
Qnet.compile(optimizer=rms, loss='mse',metrics=['accuracy'])


print(Qnet.summary())
rms2 = optimizers.RMSprop(lr=0.00025)
Qnet_copy= keras.models.clone_model(Qnet)
Qnet_copy.compile(optimizer=rms2, loss='mse')
Qnet_copy.set_weights(Qnet.get_weights())


def e_greedy(state, e = 0.1):

	state_features = features(state)
	moves = state.moves()
	state_vectors = [state_features+ move_hot_form(m) for m in moves]
	Qvalues = Qnet.predict(np.array(state_vectors)).flatten()

	if np.random.uniform()< e:
		return random.choice(moves)

	else:
		return moves[np.argmax(Qvalues)]






experience_buffer = []
n_steps  = 0
C = 1000
n_epi = 0
wins = 0
win_history = []

def train(sample_size = 32, gamma = 0.99):


	global experience_buffer

	training_data= random.sample(experience_buffer, sample_size)

	data = []
	target = []
	for given_state, action, reward, next_state, finished in training_data:

		data.append(features(given_state)+ move_hot_form(action))
		
		if finished:
			target.append(reward)
			continue
		
		elif not finished:
			# compute the max over the available actions
			state_features = features(next_state)
			state_vectors = [state_features+ move_hot_form(m) for m in next_state.moves()]
			Qtarget = Qnet_copy.predict(np.array(state_vectors)).max()

			target.append(reward+ gamma*Qtarget)

	data = np.array(data)
	target = np.array(target)
	Qnet.fit(data, target, verbose = 0)

# Qnet.save('Qnet.h5')

for _ in range(number_of_episodes):

	state = State.generate(phase=1)

	dqn_number = random.choice([1,2])
	n_epi+=1
	while not state.finished():

		given_state = state.clone(signature=state.whose_turn()) if state.get_phase() == 1 else state

		action = (None,None)
		n_steps+=1

		if state.whose_turn() ==dqn_number:
			#print('ai')
			action = e_greedy(given_state)
			state = state.next(action)

		while(state.whose_turn()==util.other(dqn_number)):
			#print('o')
			state2 = state.clone(signature=state.whose_turn()) if state.get_phase() == 1 else state
			if state.finished():
				break
			move = opponent.get_move(state2)
			state = state.next(move)

		reward = 0
		if state.finished():
			
			winner, score = state.winner()
			reward = 1 if winner == dqn_number else 0	
			
			win_history.append(reward)
			if reward ==1:
				wins+=1

			mean_reward_last_100 = np.array(win_history[-100:]).mean()
			sys.stdout.write('\r')
			sys.stdout.write('Number of episodes '+str(n_epi)+ ' wins '+str(wins)+ ' win rate ' +str(mean_reward_last_100))
			sys.stdout.flush()
			if n_epi%1000== 0:
				print('\nsaving network')
				Qnet.save('Qnet.h5')


		next_state = state.clone(signature=state.whose_turn()) if state.get_phase() == 1 else state
		experience_buffer.append((given_state, action, reward, next_state, state.finished()))
		experience_buffer = experience_buffer[-100000:]

		if n_steps>10000:
			train()

		if n_steps% 10000:
			# update target network
			Qnet_copy.set_weights(Qnet.get_weights())
		



Qnet.save('Qnet.h5')


