import gym
import numpy as np
import pygame
import sys
import random
import torch
import argparse
from rl_traffic_controller import RLTrafficController

# Detectar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

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
        self.observation_space = gym.spaces.Box(
            low=0,
            high=100,
            shape=(len(INTERSECTIONS) * 4,),
            dtype=np.float32
        )
        self.max_steps = 1000
        self.current_step = 0
        self.current_lights = [0] * len(INTERSECTIONS)
        self.vehicles = []
        self.max_vehicles = max_vehicles
        self.intersections = INTERSECTIONS
        self.screen = None
        self.clock = None
        self.reset()

    def reset(self):
        self.current_step = 0
        self.current_lights = [0] * len(self.intersections)
        self.vehicles = []
        for _ in range(self.max_vehicles):
            direction = random.choice(["N", "S", "E", "W"])
            self.spawn_vehicle(direction=direction, force=True)
        self.state = np.array([0, 1.0, 0, 1.0] * len(self.intersections), dtype=np.float32)
        return self.state

    def is_at_traffic_light(self, vehicle):
        x, y = vehicle["x"], vehicle["y"]
        direction = vehicle["dir"]
        lane = vehicle["lane"]
        stop_distance = 50
        road_center = LANE_WIDTH // 2
        for i, intersection in enumerate(self.intersections):
            int_x = intersection["x"]
            int_y = intersection["y"]
            half_size = intersection["size"] // 2
            if direction == "N":
                lane_x = int_x - road_center if lane == 0 else int_x - (LANE_WIDTH + road_center)
                if abs(x - lane_x) < CAR_WIDTH and int_y + half_size - stop_distance < y < int_y + half_size + 30:
                    return i
            elif direction == "S":
                lane_x = int_x + road_center if lane == 0 else int_x + (LANE_WIDTH + road_center)
                if abs(x - lane_x) < CAR_WIDTH and int_y - half_size - 30 < y < int_y - half_size + stop_distance:
                    return i
            elif direction == "E":
                lane_y = int_y + road_center if lane == 0 else int_y + (LANE_WIDTH + road_center)
                if abs(y - lane_y) < CAR_WIDTH and int_x - half_size - 30 < x < int_x - half_size + stop_distance:
                    return i
            elif direction == "W":
                lane_y = int_y - road_center if lane == 0 else int_y - (LANE_WIDTH + road_center)
                if abs(y - lane_y) < CAR_WIDTH and int_x + half_size - stop_distance < x < int_x + half_size + 30:
                    return i
        return -1

    def should_stop(self, vehicle):
        idx = self.is_at_traffic_light(vehicle)
        if idx >= 0:
            if (self.current_lights[idx] == 0 and vehicle["dir"] in ["E", "W"]) or \
               (self.current_lights[idx] == 1 and vehicle["dir"] in ["N", "S"]):
                return True
        return False

    def step(self, action):
        self.current_step += 1
        self.current_lights = list(action) if isinstance(action, (list, np.ndarray)) else [action] * len(self.intersections)
        while len(self.vehicles) < self.max_vehicles:
            self.spawn_vehicle(direction=random.choice(["N", "S", "E", "W"]), force=True)
        for v in self.vehicles:
            if self.should_stop(v): v["stopped"] = True
            else: v["stopped"] = False
            if not v["stopped"]:
                if v["dir"] == "N": v["y"] -= v["speed"]
                elif v["dir"] == "S": v["y"] += v["speed"]
                elif v["dir"] == "E": v["x"] += v["speed"]
                elif v["dir"] == "W": v["x"] -= v["speed"]
            if v["x"] < 0: v["x"] = SCREEN_WIDTH
            elif v["x"] > SCREEN_WIDTH: v["x"] = 0
            if v["y"] < 0: v["y"] = SCREEN_HEIGHT
            elif v["y"] > SCREEN_HEIGHT: v["y"] = 0
        state_vals, total_reward = [], 0
        for intersection in self.intersections:
            int_x, int_y = intersection["x"], intersection["y"]
            half = intersection["size"]
            nearby = [v for v in self.vehicles if abs(v["x"]-int_x) < half*2 and abs(v["y"]-int_y) < half*2]
            q_ns = sum(v["dir"] in ["N","S"] for v in nearby)
            q_ew = sum(v["dir"] in ["E","W"] for v in nearby)
            speed_ns = 1.0 if q_ns==0 else max(0.1,1.0-q_ns/20.0)
            speed_ew = 1.0 if q_ew==0 else max(0.1,1.0-q_ew/20.0)
            state_vals += [q_ns, speed_ns, q_ew, speed_ew]
            total_reward -= (q_ns+q_ew)
        self.state = np.array(state_vals, dtype=np.float32)
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
                pygame.quit(); sys.exit()
        self.screen.fill(WHITE)
        rw = 120
        pygame.draw.rect(self.screen, GRAY, (0, SCREEN_HEIGHT//3-rw//2, SCREEN_WIDTH, rw))
        pygame.draw.rect(self.screen, GRAY, (0, SCREEN_HEIGHT*2//3-rw//2, SCREEN_WIDTH, rw))
        pygame.draw.line(self.screen, WHITE, (0,SCREEN_HEIGHT//3),(SCREEN_WIDTH,SCREEN_HEIGHT//3),2)
        pygame.draw.line(self.screen, WHITE, (0,SCREEN_HEIGHT*2//3),(SCREEN_WIDTH,SCREEN_HEIGHT*2//3),2)
        pygame.draw.rect(self.screen, GRAY, (SCREEN_WIDTH//3-rw//2,0,rw,SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, GRAY, (SCREEN_WIDTH*2//3-rw//2,0,rw,SCREEN_HEIGHT))
        pygame.draw.line(self.screen, WHITE,(SCREEN_WIDTH//3,0),(SCREEN_WIDTH//3,SCREEN_HEIGHT),2)
        pygame.draw.line(self.screen, WHITE,(SCREEN_WIDTH*2//3,0),(SCREEN_WIDTH*2//3,SCREEN_HEIGHT),2)
        light, ofs = 20, 60
        for i, inter in enumerate(self.intersections):
            x,y,lights = inter["x"],inter["y"],self.current_lights[i]
            c_ns = GREEN if lights==0 else RED
            c_ew = GREEN if lights==1 else RED
            pygame.draw.rect(self.screen, c_ns,(x-LANE_WIDTH-light//2,y+ofs,light,light))
            pygame.draw.rect(self.screen, c_ns,(x+LANE_WIDTH-light//2,y-ofs-light,light,light))
            pygame.draw.rect(self.screen, c_ew,(x-ofs-light,y+LANE_WIDTH-light//2,light,light))
            pygame.draw.rect(self.screen, c_ew,(x+ofs,y-LANE_WIDTH-light//2,light,light))
        for v in self.vehicles:
            w,h = (CAR_HEIGHT,CAR_WIDTH) if v["dir"] in ["N","S"] else (CAR_WIDTH,CAR_HEIGHT)
            col = YELLOW if v.get("stopped",False) else BLUE
            pygame.draw.rect(self.screen,col,(v["x"]-w/2,v["y"]-h/2,w,h))
        font = pygame.font.SysFont(None,24)
        self.screen.blit(font.render(f"Vehículos: {len(self.vehicles)}",True,BLACK),(10,10))
        self.screen.blit(font.render(f"Detenidos: {sum(v.get('stopped',False) for v in self.vehicles)}",True,BLACK),(10,40))
        pygame.display.flip()
        self.clock.tick(30)

    def spawn_vehicle(self, direction, force=False):
        if not force and len(self.vehicles)>=self.max_vehicles: return
        lane = random.choice([0,1])
        pos = random.choice([1,2])
        rc = LANE_WIDTH//2
        if direction=="N": x = (SCREEN_WIDTH//3-rc) if pos==1 else (SCREEN_WIDTH*2//3-rc); y=SCREEN_HEIGHT
        elif direction=="S": x = (SCREEN_WIDTH//3+rc) if pos==1 else (SCREEN_WIDTH*2//3+rc); y=0
        elif direction=="E": y = (SCREEN_HEIGHT//3+rc) if pos==1 else (SCREEN_HEIGHT*2//3+rc); x=0
        else: y = (SCREEN_HEIGHT//3-rc) if pos==1 else (SCREEN_HEIGHT*2//3-rc); x=SCREEN_WIDTH
        speed = BASE_SPEED + random.uniform(-0.5,0.5)
        self.vehicles.append({"x":x,"y":y,"speed":speed,"dir":direction,"stopped":False,"lane":lane})

if __name__ == "__main__":
    env = CityTrafficSimEnv(max_vehicles=50)
    parser = argparse.ArgumentParser(description='Simulación RL PyTorch')
    parser.add_argument('--mode', choices=['rl_train','rl_test','fixed'], default='rl_train')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--episodes', type=int, default=100)
    args = parser.parse_args()

    if args.mode == 'fixed':
        obs = env.reset(); done=False; step_ctr=0; actions=[0]*len(INTERSECTIONS)
        while not done:
            if step_ctr%120==0:
                actions=[1-a if random.random()>0.5 else a for a in actions]
            obs,_,done,_=env.step(actions); env.render(); step_ctr+=1
    else:
        controller = RLTrafficController(env,len(INTERSECTIONS))
        if args.mode=='rl_train':
            scores=controller.train(episodes=args.episodes,render=False)
            controller.agent.save('traffic_model_final.pt')
        else:
            if args.model: controller.agent.load(args.model)
            controller.test(episodes=args.episodes,render=True)
    pygame.quit(); sys.exit()
