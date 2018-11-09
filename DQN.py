#! /usr/bin/env python
#coding=utf-8

import keras
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

np.random.seed(1)
weight_file="weight.h5"

class DQN:
    def __init__(
            self,
            state_size,
            action_size,
            epsilon=1,
            epsilon_decay=0.97,
            epsilon_min=0.01,
            learning_rate=0.001,
            gamma=0.9):
        self.state_size=state_size
        self.action_size=action_size
        self.epsilon=epsilon
        self.epsilon_decay=epsilon_decay
        self.epsilon_min=epsilon_min
        self.learning_rate=learning_rate
        self.gamma=gamma
        self.memory=deque(maxlen=5000)
        self.test=False
        self.batch_count=0
        self.count=0
        self.max_count=10
        self.model=self._build_model()
        self.model_target=self._build_model()

    def _build_model(self):
        model=Sequential()
        model.add(Dense(32,input_dim=self.state_size,activation='relu'))
        model.add(Dense(32,activation='relu'))
        model.add(Dense(self.action_size,activation='linear'))
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

    def act(self,state):
        if np.random.uniform() < self.epsilon and self.test==False:
            return np.random.randint(self.action_size)
        act_value=self.model.predict(state)
        return np.argmax(act_value[0])

    def replay(self,batch_size):
        self.batch_count += 1
        if self.batch_count < batch_size:
            return
        self.batch_count=0

        self.count += 1
        if self.count==self.max_count:
            self.copy_weight()
        self.count = 0

        if len(self.memory)>3*batch_size and False:
            batch = random.sample(self.memory, batch_size * 3)
        else:
            batch = random.sample(self.memory, batch_size)
        # for state,action,reward,next_state,done in batch:
        #     target=reward
        #     if not done:
        #         target=reward+self.gamma*np.max(self.model_target.predict(next_state)[0])
        #     target_f=self.model.predict(state)   #can't understand
        #     target_f[0][action]=target
        #     self.model.fit(state,target_f,epochs=1,verbose=1,
        #                    callbacks=[keras.callbacks.ModelCheckpoint(weight_file,monitor='loss',verbose=0,
        #                                                               save_best_only=True,mode='auto')])
        states=[]
        target_fs=[]
        for state, action, reward, next_state, done in batch:
            states.append(state[0])
            target = reward
            if not done:
                target = reward + self.gamma * np.max(self.model_target.predict(next_state)[0])
            target_f = self.model.predict(state)  # can't understand
            target_f[0][action] = target
            target_fs.append(target_f[0])
        states=np.asarray(states)
        target_fs=np.asarray(target_fs)
        self.model.fit(states, target_fs, batch_size=batch_size,epochs=1, verbose=1,
                       callbacks=[keras.callbacks.ModelCheckpoint(weight_file, monitor='loss', verbose=0,
                                                                  save_best_only=True, mode='auto')])
        if self.epsilon > self.epsilon_min:
            self.epsilon*=self.epsilon_decay

    def load(self):
        self.model.load_weights(weight_file)
        self.test=True

    def copy_weight(self):
        self.model_target.set_weights(self.model.get_weights())
