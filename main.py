import gym
import numpy as np
import pygame
import sys
import random
import tensorflow as tf
from rl_traffic_controller import RLTrafficController

# Colores en formato RGB
WHITE = (255, 255, 255)
GRAY = (160, 160, 160)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)

# Dimensiones de la ventana
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800

# Tamaño de los vehículos
CAR_WIDTH = 15
CAR_HEIGHT = 25

# Velocidad base de movimiento de los vehículos (píxeles por frame)
BASE_SPEED = 2.0

# Ancho de cada carril
LANE_WIDTH = 30

# Posiciones de las intersecciones
INTERSECTIONS = [
    {"x": SCREEN_WIDTH // 3, "y": SCREEN_HEIGHT // 3, "size": 120},
    {"x": SCREEN_WIDTH * 2 // 3, "y": SCREEN_HEIGHT // 3, "size": 120},
    {"x": SCREEN_WIDTH // 3, "y": SCREEN_HEIGHT * 2 // 3, "size": 120},
    {"x": SCREEN_WIDTH * 2 // 3, "y": SCREEN_HEIGHT * 2 // 3, "size": 120}
]

class CityTrafficSimEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, max_vehicles=50):
        super(CityTrafficSimEnv, self).__init__()
        self.action_space = gym.spaces.MultiDiscrete([2] * len(INTERSECTIONS))
        # Observación: para cada intersección [queue_NS, speed_NS, queue_EW, speed_EW]
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=100, 
            shape=(len(INTERSECTIONS) * 4,), 
            dtype=np.float32
        )

        self.max_steps = 1000
        self.current_step = 0

        # Variables para controlar el estado de los semáforos (0 = NS verde, 1 = EW verde)
        self.current_lights = [0] * len(INTERSECTIONS)

        # Lista para almacenar vehículos. Cada vehículo será un dict con:
        # {"x": float, "y": float, "speed": float, "dir": str, "stopped": bool}
        self.vehicles = []
        self.max_vehicles = max_vehicles  # Número fijo de vehículos en la simulación

        # Definir las zonas de influencia de los semáforos (intersecciones)
        self.intersections = INTERSECTIONS

        # Inicialización de Pygame (se hace la primera vez que se llame render)
        self.screen = None
        self.clock = None

        # Inicializa el entorno
        self.reset()

    def reset(self):
        self.current_step = 0
        self.current_lights = [0] * len(self.intersections)
        self.vehicles = []

        # Inicializamos la cantidad fija de vehículos
        for _ in range(self.max_vehicles):
            direction = random.choice(["N", "S", "E", "W"])
            self.spawn_vehicle(direction=direction, force=True)
            
        # Estado inicial "libre": sin congestión (colas = 0, velocidad máxima simulada)
        self.state = np.array([0, 1.0, 0, 1.0] * len(self.intersections), dtype=np.float32)
        return self.state

    def is_at_traffic_light(self, vehicle):
        """Determina si un vehículo está en la zona de un semáforo y qué intersección"""
        x, y = vehicle["x"], vehicle["y"]
        direction = vehicle["dir"]
        lane = vehicle["lane"]
        stop_distance = 50  # Distancia de detención antes del semáforo
        road_center = LANE_WIDTH // 2
        
        for i, intersection in enumerate(self.intersections):
            int_x = intersection["x"]
            int_y = intersection["y"]
            half_size = intersection["size"] // 2
            
            if direction == "N":
                # El carril depende de si está en el carril derecho (0) o izquierdo (1)
                # Para N, el semáforo está después de la intersección (al sur)
                lane_x = int_x - road_center if lane == 0 else int_x - (LANE_WIDTH + road_center)
                if (abs(x - lane_x) < CAR_WIDTH and 
                    y > int_y + half_size - stop_distance and
                    y < int_y + half_size + 30):  # Añadimos un pequeño margen
                    return i  # Retorna el índice de la intersección
            elif direction == "S":
                # Para S, el semáforo está antes de la intersección (al norte)
                lane_x = int_x + road_center if lane == 0 else int_x + (LANE_WIDTH + road_center)
                if (abs(x - lane_x) < CAR_WIDTH and 
                    y < int_y - half_size + stop_distance and
                    y > int_y - half_size - 30):
                    return i
            elif direction == "E":
                # Para E, el semáforo está antes de la intersección (al oeste)
                lane_y = int_y + road_center if lane == 0 else int_y + (LANE_WIDTH + road_center)
                if (abs(y - lane_y) < CAR_WIDTH and 
                    x < int_x - half_size + stop_distance and
                    x > int_x - half_size - 30):
                    return i
            elif direction == "W":
                # Para W, el semáforo está después de la intersección (al este)
                lane_y = int_y - road_center if lane == 0 else int_y - (LANE_WIDTH + road_center)
                if (abs(y - lane_y) < CAR_WIDTH and 
                    x > int_x + half_size - stop_distance and
                    x < int_x + half_size + 30):
                    return i
        return -1  # No está en ninguna intersección

    def should_stop(self, vehicle):
        """Determina si un vehículo debe detenerse en el semáforo"""
        direction = vehicle["dir"]
        
        # Verificar en qué intersección está (si está en alguna)
        intersection_idx = self.is_at_traffic_light(vehicle)
        
        if intersection_idx >= 0:  # Si está en una intersección
            # Si el semáforo está en rojo para esta dirección
            if (self.current_lights[intersection_idx] == 0 and direction in ["E", "W"]) or \
               (self.current_lights[intersection_idx] == 1 and direction in ["N", "S"]):
                return True
        return False

    def is_in_intersection(self, vehicle):
        """Comprueba si un vehículo está dentro de alguna intersección"""
        x, y = vehicle["x"], vehicle["y"]
        
        for intersection in self.intersections:
            int_x = intersection["x"]
            int_y = intersection["y"]
            half_size = intersection["size"] // 2
            
            if (abs(x - int_x) < half_size and 
                abs(y - int_y) < half_size):
                return True
                
        return False

    def step(self, action):
        self.current_step += 1
        # Actualizamos los semáforos según la acción (array de 0's y 1's)
        if isinstance(action, (list, np.ndarray)):
            self.current_lights = list(action)
        else:
            # Si solo se proporciona un valor, se aplica a todos los semáforos
            self.current_lights = [action] * len(self.intersections)

        # --- (1) Spawnear vehículos al azar en cada dirección (si la cantidad es menor a la deseada) ---
        # Si el número de vehículos es menor que el máximo, se pueden agregar nuevos vehículos.
        while len(self.vehicles) < self.max_vehicles:
            direction = random.choice(["N", "S", "E", "W"])
            self.spawn_vehicle(direction=direction, force=True)

        # --- (2) Actualizar posición de vehículos en función de la luz verde ---
        for v in self.vehicles:
            dir_ = v["dir"]
            
            # Verificar si el vehículo debe detenerse en el semáforo
            if self.should_stop(v):
                v["stopped"] = True
            else:
                # Si el semáforo está en verde o no está en zona de semáforo, puede moverse
                v["stopped"] = False

            # Mover el vehículo si no está detenido
            if not v["stopped"]:
                speed = v["speed"]
                if dir_ == "N":
                    v["y"] -= speed
                elif dir_ == "S":
                    v["y"] += speed
                elif dir_ == "E":
                    v["x"] += speed
                elif dir_ == "W":
                    v["x"] -= speed

            # Si el vehículo sobrepasa los límites, se recicla (wrap-around)
            if v["x"] < 0:
                v["x"] = SCREEN_WIDTH
            elif v["x"] > SCREEN_WIDTH:
                v["x"] = 0
            if v["y"] < 0:
                v["y"] = SCREEN_HEIGHT
            elif v["y"] > SCREEN_HEIGHT:
                v["y"] = 0

        # --- (3) Calcular observaciones y reward para cada intersección ---
        state_values = []
        total_reward = 0
        
        for i, intersection in enumerate(self.intersections):
            int_x = intersection["x"]
            int_y = intersection["y"]
            half_size = intersection["size"]
            
            # Vehículos cercanos a esta intersección
            nearby_vehicles = [v for v in self.vehicles if 
                               abs(v["x"] - int_x) < half_size * 2 and 
                               abs(v["y"] - int_y) < half_size * 2]
            
            queue_NS = sum(1 for v in nearby_vehicles if v["dir"] in ["N", "S"])
            queue_EW = sum(1 for v in nearby_vehicles if v["dir"] in ["E", "W"])
            speed_NS = 1.0 if queue_NS == 0 else max(0.1, 1.0 - queue_NS / 20.0)
            speed_EW = 1.0 if queue_EW == 0 else max(0.1, 1.0 - queue_EW / 20.0)
            
            state_values.extend([queue_NS, speed_NS, queue_EW, speed_EW])
            total_reward -= (queue_NS + queue_EW)  # Penaliza la congestión
        
        self.state = np.array(state_values, dtype=np.float32)
        done = self.current_step >= self.max_steps

        return self.state, total_reward, done, {}

    def render(self, mode="human"):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Simulación de Tráfico Urbano")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill(WHITE)

        # Dibujar carreteras: horizontales y verticales con carriles separados
        road_width = 120  # Ancho para tener espacio para dos carriles
        
        # Carreteras horizontales (tercios de la pantalla)
        pygame.draw.rect(self.screen, GRAY, (0, (SCREEN_HEIGHT // 3 - road_width // 2), SCREEN_WIDTH, road_width))
        pygame.draw.rect(self.screen, GRAY, (0, (SCREEN_HEIGHT * 2 // 3 - road_width // 2), SCREEN_WIDTH, road_width))
        
        # Líneas divisorias de carriles horizontales
        pygame.draw.line(self.screen, WHITE, (0, SCREEN_HEIGHT // 3), (SCREEN_WIDTH, SCREEN_HEIGHT // 3), 2)
        pygame.draw.line(self.screen, WHITE, (0, SCREEN_HEIGHT * 2 // 3), (SCREEN_WIDTH, SCREEN_HEIGHT * 2 // 3), 2)
        
        # Carreteras verticales (tercios de la pantalla)
        pygame.draw.rect(self.screen, GRAY, ((SCREEN_WIDTH // 3 - road_width // 2), 0, road_width, SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, GRAY, ((SCREEN_WIDTH * 2 // 3 - road_width // 2), 0, road_width, SCREEN_HEIGHT))
        
        # Líneas divisorias de carriles verticales
        pygame.draw.line(self.screen, WHITE, (SCREEN_WIDTH // 3, 0), (SCREEN_WIDTH // 3, SCREEN_HEIGHT), 2)
        pygame.draw.line(self.screen, WHITE, (SCREEN_WIDTH * 2 // 3, 0), (SCREEN_WIDTH * 2 // 3, SCREEN_HEIGHT), 2)

        # Dibujar semáforos para cada intersección
        light_size = 20
        offset = 60  # Mayor offset para adaptar a la carretera más ancha
        
        for i, intersection in enumerate(self.intersections):
            int_x = intersection["x"]
            int_y = intersection["y"]
            
            # Colores del semáforo según el estado
            color_NS = GREEN if self.current_lights[i] == 0 else RED
            color_EW = GREEN if self.current_lights[i] == 1 else RED
            
            # Semáforo para dirección N
            pygame.draw.rect(self.screen, color_NS,
                            (int_x - LANE_WIDTH - light_size // 2, 
                             int_y + offset, light_size, light_size))
            
            # Semáforo para dirección S
            pygame.draw.rect(self.screen, color_NS,
                            (int_x + LANE_WIDTH - light_size // 2, 
                             int_y - offset - light_size, light_size, light_size))
            
            # Semáforo para dirección E
            pygame.draw.rect(self.screen, color_EW,
                            (int_x - offset - light_size, 
                             int_y + LANE_WIDTH - light_size // 2, light_size, light_size))
            
            # Semáforo para dirección W
            pygame.draw.rect(self.screen, color_EW,
                            (int_x + offset, 
                             int_y - LANE_WIDTH - light_size // 2, light_size, light_size))

        # Dibujar vehículos
        for v in self.vehicles:
            x, y = v["x"], v["y"]
            dir_ = v["dir"]
            # Para vehículos en direcciones verticales, giramos el rectángulo
            if dir_ in ["N", "S"]:
                width, height = CAR_HEIGHT, CAR_WIDTH
            else:
                width, height = CAR_WIDTH, CAR_HEIGHT

            # Color diferente si el vehículo está detenido
            color = YELLOW if v.get("stopped", False) else BLUE
            pygame.draw.rect(self.screen, color, (x - width/2, y - height/2, width, height))

        # Información adicional
        font = pygame.font.SysFont(None, 24)
        info_text = font.render(f"Vehículos: {len(self.vehicles)}", True, BLACK)
        stopped_text = font.render(f"Vehículos detenidos: {sum(1 for v in self.vehicles if v.get('stopped', False))}", True, BLACK)
        self.screen.blit(info_text, (10, 10))
        self.screen.blit(stopped_text, (10, 40))

        pygame.display.flip()
        self.clock.tick(30)

    def spawn_vehicle(self, direction, force=False):
        """
        Crea (o reubica) un vehículo en una posición inicial según la dirección.
        Si 'force' es True, se agrega independientemente de la posición actual.
        """
        # Solo se agrega un nuevo vehículo si no se supera el número máximo (a menos que force sea True)
        if not force and len(self.vehicles) >= self.max_vehicles:
            return

        # Determinamos el carril (derecho o izquierdo)
        lane = random.choice([0, 1])  # 0 = carril derecho, 1 = carril izquierdo
        
        # Elegir una carretera al azar (primer tercio o segundo tercio)
        road_position = random.choice([1, 2])  # 1 = primer tercio, 2 = segundo tercio
        
        # Posiciones iniciales según la dirección, el carril y la carretera
        road_center = LANE_WIDTH // 2
        if direction == "N":
            # Vehículo que va hacia el norte, aparece en la parte inferior
            if road_position == 1:
                x = (SCREEN_WIDTH // 3) - road_center if lane == 0 else (SCREEN_WIDTH // 3) - (LANE_WIDTH + road_center)
            else:
                x = (SCREEN_WIDTH * 2 // 3) - road_center if lane == 0 else (SCREEN_WIDTH * 2 // 3) - (LANE_WIDTH + road_center)
            y = SCREEN_HEIGHT
        elif direction == "S":
            # Vehículo que va hacia el sur, aparece en la parte superior
            if road_position == 1:
                x = (SCREEN_WIDTH // 3) + road_center if lane == 0 else (SCREEN_WIDTH // 3) + (LANE_WIDTH + road_center)
            else:
                x = (SCREEN_WIDTH * 2 // 3) + road_center if lane == 0 else (SCREEN_WIDTH * 2 // 3) + (LANE_WIDTH + road_center)
            y = 0
        elif direction == "E":
            # Vehículo que va hacia el este, aparece en el lado izquierdo
            x = 0
            if road_position == 1:
                y = (SCREEN_HEIGHT // 3) + road_center if lane == 0 else (SCREEN_HEIGHT // 3) + (LANE_WIDTH + road_center)
            else:
                y = (SCREEN_HEIGHT * 2 // 3) + road_center if lane == 0 else (SCREEN_HEIGHT * 2 // 3) + (LANE_WIDTH + road_center)
        elif direction == "W":
            # Vehículo que va hacia el oeste, aparece en el lado derecho
            x = SCREEN_WIDTH
            if road_position == 1:
                y = (SCREEN_HEIGHT // 3) - road_center if lane == 0 else (SCREEN_HEIGHT // 3) - (LANE_WIDTH + road_center)
            else:
                y = (SCREEN_HEIGHT * 2 // 3) - road_center if lane == 0 else (SCREEN_HEIGHT * 2 // 3) - (LANE_WIDTH + road_center)
        else:
            return

        speed = BASE_SPEED + random.uniform(-0.5, 0.5)
        self.vehicles.append({"x": x, "y": y, "speed": speed, "dir": direction, "stopped": False, "lane": lane})


if __name__ == "__main__":
    env = CityTrafficSimEnv(max_vehicles=50)
    
    # Parsear argumentos de línea de comandos para elegir el modo
    import argparse
    parser = argparse.ArgumentParser(description='Simulador de tráfico con RL')
    parser.add_argument('--mode', type=str, default='rl',
                        choices=['rl_train', 'rl_test', 'fixed'],
                        help='Modo de ejecución: rl_train, rl_test o fixed')
    parser.add_argument('--model', type=str, default=None,
                        help='Ruta al modelo previamente entrenado (para modo rl_test)')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Número de episodios para entrenamiento')
    args = parser.parse_args()
    
    if args.mode == 'fixed':
        # Modo de semáforos con tiempo fijo (como el original), con render en cada paso
        obs = env.reset()
        done = False
        semaphore_duration = 120  # Duración del semáforo en ciclos
        step_counter = 0
        
        current_actions = [0] * len(INTERSECTIONS)
        while not done:
            if step_counter % semaphore_duration == 0:
                for i in range(len(current_actions)):
                    if random.random() > 0.5:
                        current_actions[i] = 1 - current_actions[i]
            
            obs, reward, done, info = env.step(current_actions)
            env.render()     # <-- Aquí renderizas
            step_counter += 1
    
    else:
        # Modo de aprendizaje por refuerzo
        controller = RLTrafficController(env, len(INTERSECTIONS))
        
        if args.mode == 'rl_train':
            # Modo de entrenamiento (headless: SIN render)
            import time
            print("Iniciando entrenamiento del controlador RL...")
            start_time = time.time()
            
            scores = controller.train(episodes=args.episodes, render=False)
            
            end_time = time.time()
            training_duration = end_time - start_time
            hours, remainder = divmod(training_duration, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            print("\n===== RESUMEN DE ENTRENAMIENTO =====")
            print(f"Tiempo total de entrenamiento: {int(hours)}h {int(minutes)}m {int(seconds)}s")
            print(f"Episodios completados: {args.episodes}")
            print(f"Promedio de recompensa (todos): {np.mean(scores):.2f}")
            print(f"Promedio últimos 10 episodios: {np.mean(scores[-10:]):.2f}")
            print(f"Mejor episodio: {max(scores):.2f}")
            print(f"Peor episodio: {min(scores):.2f}")
            print("===================================")
            
            controller.agent.save("traffic_model_final.h5")
            print("Modelo guardado como 'traffic_model_final.h5'")
            
            # Graficar resultados del entrenamiento
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 8))
            
            # Graficar puntuaciones por episodio
            plt.subplot(2, 1, 1)
            plt.plot(scores, 'b-')
            plt.title('Recompensas durante el entrenamiento')
            plt.xlabel('Episodio')
            plt.ylabel('Recompensa total')
            plt.grid(True)
            
            # Graficar promedios móviles (tendencia)
            plt.subplot(2, 1, 2)
            window_size = min(10, len(scores))
            moving_avg = [np.mean(scores[max(0, i-window_size):i+1]) for i in range(len(scores))]
            plt.plot(moving_avg, 'r-')
            plt.title(f'Promedio móvil de recompensas (ventana: {window_size} episodios)')
            plt.xlabel('Episodio')
            plt.ylabel('Recompensa promedio')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('training_results.png')
            print("Gráficos de entrenamiento guardados en 'training_results.png'")
            plt.close()
            
        elif args.mode == 'rl_test':
            # Modo de prueba con modelo entrenado: render sí, pero solo inference
            if args.model:
                controller.agent.load(args.model)
                print(f"Modelo cargado desde: {args.model}")
            else:
                print("Advertencia: No se especificó un modelo. Usando modelo sin entrenar.")
            
            print("Iniciando prueba del controlador RL...")
            obs = env.reset()
            done = False
            while not done:
                action = controller.agent.act(obs)
                obs, reward, done, info = env.step(action)
                env.render()   # <-- Aquí renderizas durante la prueba
            print(f"Promedio de recompensa: {np.mean(controller.scores)}")
    
    pygame.quit()
    sys.exit()
