import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 24)
        self.fc2 = nn.Linear(24, 24)
        self.out = nn.Linear(24, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class DQNAgent:
    def __init__(self, state_size, action_size, device=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = device or get_device()

        # Model and target
        self.model = QNetwork(self.state_size, self.action_size).to(self.device)
        self.target_model = QNetwork(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        n_int = len(state) // 4
        # Epsilon-greedy
        if random.random() <= self.epsilon:
            return [random.randrange(2) for _ in range(n_int)]

        self.model.eval()
        with torch.no_grad():
            state_v = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.model(state_v).cpu().numpy()[0]
        # Split into 2-action groups
        actions = []
        for i in range(n_int):
            slice_vals = q_vals[i*2:(i+1)*2]
            actions.append(int(np.argmax(slice_vals)))
        return actions

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return False

        minibatch = random.sample(self.memory, batch_size)
        self.model.train()

        for state, action_list, reward, next_state, done in minibatch:
            # Prepare tensors
            state_v = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            next_state_v = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Current Q-values
            q_vals = self.model(state_v).squeeze(0)
            target_q = q_vals.clone().detach()

            # Compute target value
            if done:
                target_val = reward
            else:
                with torch.no_grad():
                    next_q = self.target_model(next_state_v).squeeze(0)
                # Max per intersection
                n_int = len(action_list)
                next_q_max = []
                for i in range(n_int):
                    next_q_max.append(next_q[i*2:(i+1)*2].max())
                target_val = reward + self.gamma * torch.stack(next_q_max).mean().item()

            # Update only taken actions
            for i, a in enumerate(action_list):
                idx = i*2 + a
                target_q[idx] = target_val

            # Compute loss and optimize
            self.optimizer.zero_grad()
            loss = self.criterion(q_vals.unsqueeze(0), target_q.unsqueeze(0))
            loss.backward()
            self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return True

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.update_target_model()

    def save(self, path):
        torch.save(self.model.state_dict(), path)


class RLTrafficController:
    def __init__(self, env, num_intersections):
        self.env = env
        self.num_intersections = num_intersections
        self.state_size = num_intersections * 4
        self.action_size = 2 * num_intersections
        self.device = get_device()

        self.agent = DQNAgent(self.state_size, self.action_size, device=self.device)
        self.batch_size = 32
        self.train_interval = 4
        self.step_counter = 0
        self.update_target_interval = 100

    def train(self, episodes=1000, max_steps=1000, render=False):
        scores = []
        print("\n===== INICIANDO ENTRENAMIENTO =====")
        print(f"Total de episodios: {episodes}")
        print(f"Pasos máximos por episodio: {max_steps}")
        print(f"Batch size: {self.batch_size}")
        print(f"Intervalo de entrenamiento: cada {self.train_interval} pasos")
        print(f"Actualización del modelo objetivo: cada {self.update_target_interval} pasos")
        print("===================================\n")

        for e in range(episodes):
            state = self.env.reset()
            total_reward = 0
            step_rewards = []
            log_interval = max(1, max_steps // 10)

            print(f"\n[Episodio {e+1}/{episodes}] Epsilon: {self.agent.epsilon:.4f}")

            for step in range(max_steps):
                self.step_counter += 1
                action = self.agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                step_rewards.append(reward)

                # Logging
                if (step + 1) % log_interval == 0 or step == 0 or done:
                    avg_recent = sum(step_rewards[-log_interval:]) / min(log_interval, len(step_rewards))
                    mem_usage = len(self.agent.memory) / self.agent.memory.maxlen * 100
                    print(f"  Paso {step+1}/{max_steps} - Reward: {reward:.2f} - Avg últimos {min(log_interval,len(step_rewards))}: {avg_recent:.2f} - Memoria: {mem_usage:.1f}%")

                # Training
                if self.step_counter % self.train_interval == 0:
                    trained = self.agent.replay(self.batch_size)
                    if (step + 1) % log_interval == 0:
                        status = "Replay realizado" if trained else "Memoria insuficiente"
                        print(f"  [Entrenamiento] {status} (batch={self.batch_size}, memoria={len(self.agent.memory)})")

                # Update target model
                if self.step_counter % self.update_target_interval == 0:
                    self.agent.update_target_model()
                    print(f"  [Actualización] Modelo objetivo actualizado en paso {step+1}")

                if render:
                    self.env.render()
                if done:
                    break

            scores.append(total_reward)
            print(f"  [Fin Episodio {e+1}] Score: {total_reward:.2f}, Epsilon: {self.agent.epsilon:.4f}")

            # Guardar cada 50 episodios
            if (e + 1) % 50 == 0:
                path = f"traffic_model_ep{e+1}.pt"
                self.agent.save(path)
                print(f"  [Guardado] Modelo en: {path}")

        return scores

    def test(self, episodes=10, render=True):
        self.agent.epsilon = 0.0
        scores = []
        for e in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.agent.act(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                if render:
                    self.env.render()
            scores.append(total_reward)
            print(f"Prueba {e+1}/{episodes}, Score: {total_reward}")
        return scores
