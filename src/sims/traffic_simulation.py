from matplotlib import animation, pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from IPython.display import HTML
from tqdm import tqdm
from models.car import Car
from models.city import City
from agents.traffic_light_agent import TrafficLightAgent

from constants import (
    STATE_SIZE,
    ACTION_SIZE,
    MAX_CARS,
)

class TrafficSimulation:
    """Clase principal para ejecutar la simulación de tráfico"""
    
    def __init__(self):
        self.city = City()
        self.agents = {}  # Un agente por intersección {(x,y): agente}
        self.episode = 0
        self.step_count = 0
        self.last_action = {i: 0 for i in self.city.traffic_lights.keys()}
        self.switch_penalty = 0.2  # ajuste fino: cuánto penalizar cada cambio
        self.metrics = {
            'congestion': [],
            'avg_waiting_time': [],
            'throughput': []
        }
        
        # Inicializar agentes para cada intersección
        for intersection in self.city.traffic_lights.keys():
            self.agents[intersection] = TrafficLightAgent(STATE_SIZE, ACTION_SIZE)
        
        # Inicializar coches
        self.spawn_cars(MAX_CARS)
    
    def spawn_cars(self, num_cars):
        """Añade un número específico de coches a la simulación"""
        for i in range(num_cars):
            car = Car(self.city, id=i)
            self.city.add_car(car)
    
    def run_step(self, training=True):
        """Ejecuta un paso de la simulación"""
        self.step_count += 1
        
        # Registrar estado anterior para cada intersección
        prev_states = {}
        for intersection, agent in self.agents.items():
            prev_states[intersection] = self.city.get_state_for_intersection(intersection)
        
        # Los agentes seleccionan acciones para cada semáforo
        actions = {}
        for intersection, agent in self.agents.items():
            state = prev_states[intersection]
            action = agent.select_action(state)
            actions[intersection] = action
            
            # Aplicar la acción (modificar duración del ciclo del semáforo)
            switched = 1 if action == 1 else 0
            self.apply_traffic_light_action(intersection, action)
            self.last_action[intersection] = switched
        
        # Actualizar el estado de la ciudad
        self.city.update()
        
        # Calcular recompensas y nuevos estados
        for intersection, agent in self.agents.items():
            new_state = self.city.get_state_for_intersection(intersection)
            reward = self.calculate_reward(intersection)
            
            # Almacenar experiencia
            if training:
                agent.remember(
                    prev_states[intersection], 
                    actions[intersection], 
                    reward, 
                    new_state, 
                    False  # done siempre es False en esta simulación continua
                )
                
                # Entrenar el agente con experiencias pasadas
                agent.replay()
        
        # Recolectar métricas
        self.collect_metrics()
        
        # Cada ciertos pasos, reemplazar coches que llegaron a su destino
        if self.step_count % 20 == 0:
            self.replace_finished_cars()
    
    def apply_traffic_light_action(self, intersection, action):
        """Aplica la acción del agente al semáforo correspondiente"""
        # Acción 0: Mantener estado actual
        # Acción 1: Cambiar estado
        # Acción 2: Extender fase verde actual
        # Acción 3: Acortar fase verde actual
        
        current_state = self.city.traffic_lights[intersection]
        
        if action == 1:  # Cambiar estado
            self.city.traffic_lights[intersection] = 1 - current_state
        # Las acciones 2 y 3 requerirían una lógica más compleja para la duración de los ciclos
    
    def calculate_reward(self, intersection):
        """Calcula la recompensa para un agente basado en el estado del tráfico"""
        x, y = intersection
        reward = 0.0
        
        # 1. Penalizar por alta densidad de tráfico alrededor de la intersección
        nearby_density = 0
        for i in range(max(0, x-2), min(self.city.size, x+3)):
            for j in range(max(0, y-2), min(self.city.size, y+3)):
                nearby_density += self.city.traffic_density[i, j]
        
        reward -= nearby_density * 2  # Penalización por congestión
        
        # 2. Recompensar por flujo de tráfico (coches moviéndose vs. esperando)
        cars_waiting = 0
        cars_moving = 0
        
        for car in self.city.cars:
            car_x, car_y = int(car.position[0]), int(car.position[1])
            # Si el coche está cerca de esta intersección
            if abs(car_x - x) <= 2 and abs(car_y - y) <= 2:
                if car.waiting_time > 0:
                    cars_waiting += 1
                else:
                    cars_moving += 1
        
        reward += cars_moving * 0.5  # Recompensa por coches en movimiento
        reward -= cars_waiting * 1.0  # Penalización por coches esperando
        
        # 3. Penalizar cambios frecuentes e innecesarios de semáforo
        if self.last_action[intersection] == 1:
            reward -= self.switch_penalty
        
        return reward
    
    def collect_metrics(self):
        """Recolecta métricas de rendimiento de la simulación"""
        # 1. Congestión global
        congestion = self.city.calculate_congestion()
        self.metrics['congestion'].append(congestion)
        
        # 2. Tiempo promedio de espera
        waiting_times = [car.waiting_time for car in self.city.cars]
        avg_waiting = sum(waiting_times) / max(1, len(waiting_times))
        self.metrics['avg_waiting_time'].append(avg_waiting)
        
        # 3. Throughput (coches que completan su viaje)
        throughput = sum(1 for car in self.city.cars if car.reached_destination)
        self.metrics['throughput'].append(throughput)
    
    def replace_finished_cars(self):
        """Reemplaza los coches que han llegado a su destino con nuevos"""
        for i, car in enumerate(list(self.city.cars)):
            if car.reached_destination:
                self.city.remove_car(car)
                new_car = Car(self.city, id=car.id)
                self.city.add_car(new_car)
    
    def train(self, episodes=10, steps_per_episode=100):
        """Entrena los agentes durante varios episodios"""
        for episode in range(episodes):
            self.episode = episode
            print(f"Episodio {episode+1}/{episodes}")
            
            # Reiniciar métricas para este episodio
            for key in self.metrics:
                self.metrics[key] = []
            
            # Ejecutar episodio
            for _ in tqdm(range(steps_per_episode)):
                self.run_step(training=True)
            
            # Guardar modelo periódicamente
            if (episode + 1) % 5 == 0:
                for intersection, agent in self.agents.items():
                    agent.save(f"agent_intersection_{intersection[0]}_{intersection[1]}_ep{episode+1}.pth")
            
            # Mostrar métricas al final del episodio
            # self.plot_metrics()
    
    def test(self, steps=100):
        """Ejecuta una simulación de prueba con agentes ya entrenados"""
        # Reiniciar métricas
        for key in self.metrics:
            self.metrics[key] = []
        
        # Ejecutar simulación
        print("Ejecutando simulación de prueba...")
        for _ in tqdm(range(steps)):
            self.run_step(training=False)
        
        # Mostrar métricas y visualizaciones
        self.plot_metrics()
        self.visualize_traffic()
    
    def plot_metrics(self):
        """Visualiza las métricas recolectadas durante la simulación"""
        plt.figure(figsize=(15, 10))
        
        # Congestión
        plt.subplot(3, 1, 1)
        plt.plot(self.metrics['congestion'])
        plt.title('Congestión Global')
        plt.xlabel('Paso de simulación')
        plt.ylabel('Nivel de congestión')
        
        # Tiempo promedio de espera
        plt.subplot(3, 1, 2)
        plt.plot(self.metrics['avg_waiting_time'])
        plt.title('Tiempo Promedio de Espera')
        plt.xlabel('Paso de simulación')
        plt.ylabel('Pasos')
        
        # Throughput
        plt.subplot(3, 1, 3)
        plt.plot(self.metrics['throughput'])
        plt.title('Throughput (Coches que completaron su viaje)')
        plt.xlabel('Paso de simulación')
        plt.ylabel('Cantidad')
        
        plt.tight_layout()
        plt.savefig(f"metrics_episode_{self.episode}.png")
        plt.show()
    
    def visualize_traffic(self):
        """Genera una visualización de la ciudad y el tráfico"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Visualizar la estructura de la ciudad
        city_colors = {
            0: 'white',  # Vacío
            1: 'gray',   # Calle
            2: 'black',  # Edificio
            3: 'blue'    # Intersección
        }
        
        # Crear un colormap personalizado
        cmap = LinearSegmentedColormap.from_list('city_cmap', 
                                               [city_colors[i] for i in range(4)])
        
        # Mostrar la cuadrícula de la ciudad
        ax.imshow(self.city.grid, cmap=cmap)
        
        arrow_length = 0.3
        for x in range(self.city.size):
            for y in range(self.city.size):
                dir_code = self.city.lane_dir[x, y]
                if self.city.grid[x, y] == 1 and dir_code >= 0:
                    # centro de la celda
                    cx, cy = y, x  
                    # vectores dx,dy según dir_code
                    if dir_code == 0:   # Norte
                        dx, dy = 0, -arrow_length
                    elif dir_code == 1: # Este
                        dx, dy = arrow_length, 0
                    elif dir_code == 2: # Sur
                        dx, dy = 0, arrow_length
                    else:               # Oeste
                        dx, dy = -arrow_length, 0

                    ax.arrow(cx - dx/2, cy - dy/2, dx, dy,
                             head_width=0.1, head_length=0.1, length_includes_head=True)
        
        # Mostrar semáforos
        for (x, y), state in self.city.traffic_lights.items():
            color = 'green' if state == 0 else 'red'
            circle = plt.Circle((y, x), 0.2, color=color)
            ax.add_patch(circle)
        
        # Mostrar coches
        for car in self.city.cars:
            x, y = car.position
            rect = plt.Rectangle((y - 0.3, x - 0.3), 0.6, 0.6, 
                               color='yellow' if not car.reached_destination else 'green')
            ax.add_patch(rect)
            ax.text(y, x, str(car.id), va='center', ha='center', fontsize=8, color='black', weight='bold')
        
        # Mostrar densidad de tráfico como mapa de calor transparente
        density_mask = self.city.traffic_density > 0
        ax.imshow(np.ma.masked_where(~density_mask, self.city.traffic_density), 
                 cmap='Reds', alpha=0.5)
        
        plt.title('Simulación de Tráfico Urbano')
        plt.tight_layout()
        plt.savefig(f"traffic_visualization_step_{self.step_count}.png")
        plt.show()

    def generate_animation(self, steps=100, interval=200):
        """Genera y muestra en el notebook una animación interactiva de la simulación."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Precrear colormap
        city_colors = {
            0: 'white',  # Vacío
            1: 'gray',   # Calle
            2: 'black',  # Edificio
            3: 'blue'    # Intersección
        }
        cmap = LinearSegmentedColormap.from_list(
            'city_cmap', [city_colors[i] for i in range(4)]
        )

        def update(frame):
            ax.clear()
            # 1) Avanzar simulación un paso sin entrenamiento
            self.run_step(training=False)

            # 2) Dibujar grid
            ax.imshow(self.city.grid, cmap=cmap)

            # 3) Dibujar flechas de sentido
            arrow_length = 0.3
            for x in range(self.city.size):
                for y in range(self.city.size):
                    dir_code = self.city.lane_dir[x, y]
                    if self.city.grid[x, y] == 1 and dir_code >= 0:
                        cx, cy = y, x
                        if dir_code == 0:   dx, dy = 0, -arrow_length
                        elif dir_code == 1: dx, dy = arrow_length, 0
                        elif dir_code == 2: dx, dy = 0, arrow_length
                        else:               dx, dy = -arrow_length, 0

                        ax.arrow(cx - dx/2, cy - dy/2, dx, dy,
                                 head_width=0.1, head_length=0.1,
                                 length_includes_head=True)

            # 4) Dibujar semáforos
            for (x, y), state in self.city.traffic_lights.items():
                color = 'green' if state == 0 else 'red'
                ax.add_patch(plt.Circle((y, x), 0.2, color=color))

            # 5) Dibujar coches
            for car in self.city.cars:
                x, y = car.position
                color = 'yellow' if not car.reached_destination else 'green'
                ax.add_patch(plt.Rectangle((y-0.3, x-0.3), 0.6, 0.6, color=color))
                ax.text(y, x, str(car.id), va='center', ha='center', fontsize=8, color='black', weight='bold')

            # 6) Capa de densidad
            density_mask = self.city.traffic_density > 0
            ax.imshow(
                np.ma.masked_where(~density_mask, self.city.traffic_density),
                cmap='Reds', alpha=0.5
            )

            ax.set_title(f'Simulación - Paso {self.step_count}')
            ax.set_xticks([])
            ax.set_yticks([])
            return []

        ani = animation.FuncAnimation(
            fig, update, frames=steps, interval=interval, blit=False
        )

        # En lugar de guardar, devolvemos el HTML interactivo
        plt.show()  
        return HTML(ani.to_jshtml())