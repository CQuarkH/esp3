import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # factor de descuento
        self.epsilon = 1.0   # tasa de exploración inicial
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Red neuronal para aproximación de función Q
        model = tf.keras.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        # Actualizar modelo objetivo
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Política epsilon-greedy
        if np.random.rand() <= self.epsilon:
            # Exploración: acciones aleatorias
            return [random.randrange(2) for _ in range(len(state) // 4)]
        
        # Explotación: mejor acción según el modelo
        act_values = self.model.predict(np.array([state]), verbose=0)
        # Convertir la salida en un array de acciones para cada semáforo
        actions = []
        for i in range(len(state) // 4):
            # Cada semáforo tiene 2 posibles acciones (0 o 1)
            action_idx = np.argmax(act_values[0][i*2:(i+1)*2])
            actions.append(action_idx)
        return actions

    def replay(self, batch_size):
        # Entrenar con experiencias pasadas
        if len(self.memory) < batch_size:
            return False
        
        minibatch = random.sample(self.memory, batch_size)
        loss_values = []
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Predicción de Q-values para el siguiente estado
                next_q_values = self.target_model.predict(np.array([next_state]), verbose=0)
                
                # Obtener el mejor Q-value para cada semáforo
                next_q_max = []
                for i in range(len(next_state) // 4):
                    next_q_max.append(np.max(next_q_values[0][i*2:(i+1)*2]))
                
                # Actualizar el objetivo Q
                target = reward + self.gamma * np.mean(next_q_max)
            
            # Actualizar el valor Q para la acción tomada
            target_f = self.model.predict(np.array([state]), verbose=0)
            
            for i, a in enumerate(action):
                target_f[0][i*2 + a] = target
            
            # Entrenar el modelo
            history = self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
            loss_values.append(history.history['loss'][0])
        
        # Reducir epsilon para favorecer explotación
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return True

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class RLTrafficController:
    def __init__(self, env, num_intersections):
        self.env = env
        self.num_intersections = num_intersections
        self.state_size = num_intersections * 4  # 4 valores por intersección
        self.action_size = 2 * num_intersections  # 2 acciones por semáforo (0 o 1)
        self.agent = DQNAgent(self.state_size, self.action_size)
        self.batch_size = 32
        self.train_interval = 4  # Entrenar cada 4 pasos
        self.step_counter = 0
        self.update_target_interval = 100  # Actualizar modelo objetivo cada 100 pasos
        
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
            avg_reward_per_step = 0
            step_rewards = []
            log_interval = max(1, max_steps // 10)  # Mostrar logs en 10 puntos durante cada episodio
            
            print(f"\n[Episodio {e+1}/{episodes}] Iniciando... Epsilon: {self.agent.epsilon:.4f}")
            
            for step in range(max_steps):
                self.step_counter += 1
                
                # Seleccionar acción
                action = self.agent.act(state)
                
                # Ejecutar acción en el entorno
                next_state, reward, done, _ = self.env.step(action)
                
                # Guardar experiencia en memoria
                self.agent.remember(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                step_rewards.append(reward)
                
                # Mostrar progreso durante el episodio
                if (step + 1) % log_interval == 0 or step == 0 or done:
                    avg_reward_per_step = sum(step_rewards[-log_interval:]) / min(log_interval, len(step_rewards))
                    memory_usage = len(self.agent.memory) / self.agent.memory.maxlen * 100
                    print(f"  Paso {step+1}/{max_steps} - Reward: {reward:.2f} - Avg últimos {min(log_interval, len(step_rewards))} pasos: {avg_reward_per_step:.2f} - Memoria: {memory_usage:.1f}%")
                    
                    # Información del estado actual
                    features = []
                    for i in range(self.num_intersections):
                        features.append({
                            "NS_queue": state[i*4],
                            "NS_speed": state[i*4+1],
                            "EW_queue": state[i*4+2],
                            "EW_speed": state[i*4+3]
                        })
                    print(f"  Estado actual de intersecciones: {features}")
                    print(f"  Acciones tomadas: {action}")
                
                # Entrenar con experiencias pasadas
                if self.step_counter % self.train_interval == 0:
                    replays = self.agent.replay(self.batch_size)
                    if (step + 1) % log_interval == 0:
                        mem_size = len(self.agent.memory)
                        if replays:
                            print(f"  [Entrenamiento] Replay realizado (batch={self.batch_size}, memoria={mem_size})")
                        else:
                            print(f"  [Entrenamiento] Memoria insuficiente ({mem_size}/{self.batch_size} ejemplos necesarios)")
                
                # Actualizar modelo objetivo
                if self.step_counter % self.update_target_interval == 0:
                    self.agent.update_target_model()
                    print(f"  [Actualización] Modelo objetivo actualizado en paso {step+1}")
                
                # Renderizar entorno solo si se solicita
                if render:
                    self.env.render()
                
                if done:
                    print(f"  [Episodio terminado en paso {step+1}]")
                    break
            
            scores.append(total_reward)
            
            # Resumen del episodio
            print(f"\n[Resumen Episodio {e+1}]")
            print(f"  Puntuación total: {total_reward:.2f}")
            print(f"  Epsilon actual: {self.agent.epsilon:.4f}")
            print(f"  Pasos completados: {step+1}")
            print(f"  Reward promedio por paso: {total_reward/(step+1):.4f}")
            
            # Comprobación de la exploración vs explotación
            exploration_ratio = sum(1 for v in step_rewards if v < -10) / len(step_rewards) if step_rewards else 0
            print(f"  Ratio de exploración (aprox): {exploration_ratio:.2f}")
            
            # Estado final de las intersecciones
            features = []
            for i in range(self.num_intersections):
                features.append({
                    "NS_queue": state[i*4],
                    "NS_speed": state[i*4+1],
                    "EW_queue": state[i*4+2],
                    "EW_speed": state[i*4+3]
                })
            print(f"  Estado final: {features}")
            
            # Progreso del entrenamiento
            if e > 0:
                avg_last_5 = sum(scores[-min(5, len(scores)):]) / min(5, len(scores))
                print(f"  Promedio últimos 5 episodios: {avg_last_5:.2f}")
                print(f"  Mejora respecto al episodio anterior: {total_reward - scores[-2]:.2f}")
                
                # Tendencia de aprendizaje
                if len(scores) >= 3:
                    last_3 = scores[-3:]
                    is_improving = last_3[2] > last_3[1] > last_3[0]
                    print(f"  Tendencia: {'MEJORANDO ↑' if is_improving else 'ESTABLE/FLUCTUANTE ↔'}")
            
            # Guardar modelo cada 50 episodios
            if (e + 1) % 50 == 0:
                model_path = f"traffic_model_ep{e+1}.h5"
                self.agent.save(model_path)
                print(f"  [Guardado] Modelo guardado en: {model_path}")
        
        return scores
    
    def test(self, episodes=10, render=True):
        """Probar el modelo entrenado"""
        self.agent.epsilon = 0.0  # Sin exploración durante pruebas
        scores = []
        
        for e in range(episodes):
            state = self.env.reset()
            total_reward = 0
            
            done = False
            while not done:
                action = self.agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                total_reward += reward
                
                if render:
                    self.env.render()
            
            scores.append(total_reward)
            print(f"Prueba {e+1}/{episodes}, Puntuación: {total_reward}")
        
        return scores