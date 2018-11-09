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

class DQN:
    def __init__(
            self,
            state_size,
            action_size,
            epsilon=1,
            epsilon_decay=0.998,
            epsilon_min=0.05,
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
        self.model=self._build_model()

    def _build_model(self):
        model=Sequential()
        model.add(Dense(24,input_dim=self.state_size,activation='relu'))
        model.add(Dense(24,activation='relu'))
        model.add(Dense(self.action_size,activation='linear'))
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

    def act(self,state):
        if np.random.uniform() < self.epsilon:
            return np.random.randint(self.action_size)
        act_value=self.model.predict(state)
        return np.argmax(act_value[0])

    def replay(self,batch_size):
        batch=random.sample(self.memory,batch_size)
        for state,action,reward,next_state,done in batch:
            target=reward
            if not done:
                target=reward+self.gamma*np.max(self.model.predict(next_state)[0])
            target_f=self.model.predict(state)   #can't understand
            target_f[0][action]=target
            self.model.fit(state,target_f,epochs=1,verbose=1)
        if self.epsilon > self.epsilon_min:
            self.epsilon*=self.epsilon_decay
