from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from constants import (
    BATCH_SIZE,
    GAMMA,
    REPLAY_BUFFER_SIZE,
    LEARNING_RATE,
    EPSILON_START,
    EPSILON_END,
    EPSILON_DECAY,
    TARGET_UPDATE,
)



class TrafficLightAgent:
    """Agente de aprendizaje por refuerzo para control de semáforos"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=REPLAY_BUFFER_SIZE)
        self.epsilon = EPSILON_START
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Redes neuronales para Q-learning
        self.policy_net = self.build_model().to(self.device)
        self.target_net = self.build_model().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.loss_function = nn.MSELoss()
        self.update_count = 0
        
    def build_model(self):
        """Construye la red neuronal para aproximar la función Q"""
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model
    
    def select_action(self, state):
        """Selecciona una acción usando política epsilon-greedy"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.max(1)[1].item()
    
    def remember(self, state, action, reward, next_state, done):
        """Almacena experiencia en el buffer de replay"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Entrena la red con experiencias aleatorias del buffer"""
        if len(self.memory) < BATCH_SIZE:
            return
        
        minibatch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Q-learning
        current_q = self.policy_net(states).gather(1, actions)
        next_q = self.target_net(next_states).max(1)[0].unsqueeze(1).detach()
        target_q = rewards + (1 - dones) * GAMMA * next_q
        
        # Actualizar la red
        loss = self.loss_function(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Actualizar red objetivo periódicamente
        self.update_count += 1
        if self.update_count % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decaimiento de epsilon
        self.epsilon = max(EPSILON_END, EPSILON_START - 
                           (self.update_count / EPSILON_DECAY))
    
    def save(self, path):
        """Guarda el modelo entrenado"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """Carga un modelo previamente entrenado"""
        if torch.cuda.is_available():
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
        
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.target_net.eval()