import random
import numpy as np

class Car:
    """Representación de un coche en la simulación"""
    
    def __init__(self, city, id=0):
        self.city = city
        self.id = id
        self.position = self.random_start_position()
        self.destination = self.random_destination()
        self.route = self.plan_route()
        self.speed = 1.0  # Velocidad base en casillas por paso
        self.waiting_time = 0  # Tiempo esperando en semáforos
        self.total_travel_time = 0  # Tiempo total de viaje
        self.reached_destination = False
        
    def random_start_position(self):
        """Genera una posición inicial aleatoria en una calle"""
        valid_positions = []
        for i in range(self.city.size):
            for j in range(self.city.size):
                if self.city.grid[i, j] == 1:  # Si es una calle
                    valid_positions.append((i, j))
        
        return random.choice(valid_positions) if valid_positions else (0, 0)
    
    def random_destination(self):
        """Genera un destino aleatorio diferente a la posición inicial"""
        valid_positions = []
        for i in range(self.city.size):
            for j in range(self.city.size):
                if self.city.grid[i, j] == 1 and (i, j) != self.position:  # Si es una calle y no es la posición actual
                    valid_positions.append((i, j))
        
        return random.choice(valid_positions) if valid_positions else ((self.position[0] + 5) % self.city.size,
                                                                      (self.position[1] + 5) % self.city.size)
    
    def get_valid_next_positions(self, current_pos):
        """Obtiene posiciones válidas adyacentes según la dirección del carril"""
        x, y = int(current_pos[0]), int(current_pos[1])
        valid_next = []
        
        # Comprobar las cuatro direcciones posibles
        directions = [(x-1, y), (x, y+1), (x+1, y), (x, y-1)]  # Norte, Este, Sur, Oeste
        dir_codes = [0, 1, 2, 3]  # Códigos de dirección correspondientes
        
        for (nx, ny), dir_code in zip(directions, dir_codes):
            # Verificar si está dentro de los límites
            if 0 <= nx < self.city.size and 0 <= ny < self.city.size:
                # Verificar si es una calle o intersección
                if self.city.grid[nx, ny] in [1, 3]:  
                    # Si es una calle, comprobar si la dirección es correcta
                    if self.city.grid[nx, ny] == 1:
                        lane_dir = self.city.lane_dir[nx, ny]
                        # Si lane_dir es -1, significa que no hay restricción de dirección
                        if lane_dir == -1 or lane_dir == dir_code:
                            valid_next.append((nx, ny))
                    # Si es una intersección, siempre es válida
                    elif self.city.grid[nx, ny] == 3:
                        valid_next.append((nx, ny))
        
        return valid_next
    
    def plan_route(self):
        """Planifica una ruta desde la posición actual hasta el destino respetando el sentido de los carriles"""
        # Utilizamos el algoritmo A* para encontrar la ruta
        start = (int(self.position[0]), int(self.position[1]))
        goal = (int(self.destination[0]), int(self.destination[1]))
        
        # Si el inicio y el destino son iguales, no hay ruta necesaria
        if start == goal:
            return []
        
        # Inicializar listas abierta y cerrada
        open_list = [start]
        closed_list = set()
        
        # Diccionarios para registrar el costo g y f
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        # Diccionario para registrar de dónde viene cada nodo
        came_from = {}
        
        while open_list:
            # Obtener el nodo con el menor f_score
            current = min(open_list, key=lambda x: f_score.get(x, float('inf')))
            
            # Si hemos llegado al destino, reconstruir y devolver el camino
            if current == goal:
                return self.reconstruct_path(came_from, current)
            
            open_list.remove(current)
            closed_list.add(current)
            
            # Examinar los vecinos válidos
            for neighbor in self.get_valid_next_positions(current):
                if neighbor in closed_list:
                    continue
                
                # Costo tentativo
                tentative_g = g_score.get(current, float('inf')) + 1
                
                if neighbor not in open_list:
                    open_list.append(neighbor)
                elif tentative_g >= g_score.get(neighbor, float('inf')):
                    continue
                
                # Este camino es el mejor hasta ahora
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)
        
        # Si no hay camino, intentar volver a planificar con restricciones más relajadas
        # En este caso simplemente devolvemos una lista vacía indicando que no hay ruta
        return []
    
    def heuristic(self, a, b):
        """Función heurística para A*: distancia de Manhattan"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def reconstruct_path(self, came_from, current):
        """Reconstruye el camino a partir del diccionario de nodos previos"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        
        # El camino está desde el final al principio, así que lo invertimos
        path.reverse()
        # No incluimos la posición inicial
        if path and path[0] == (int(self.position[0]), int(self.position[1])):
            path.pop(0)
        
        return path
    
    def move(self):
        """Mueve el coche según su ruta planeada, respetando semáforos"""
        self.total_travel_time += 1
        
        if not self.route or self.reached_destination:
            self.reached_destination = True
            return
        
        next_position = self.route[0]
        x, y = int(next_position[0]), int(next_position[1])
        
        # Verificar si la próxima posición es una intersección con semáforo
        if self.city.grid[x, y] == 3:
            light_state = self.city.traffic_lights.get((x, y))
            if light_state is None:  # Si no hay semáforo en esta intersección
                pass
            else:
                current_x, current_y = int(self.position[0]), int(self.position[1])
                
                # Determinar dirección del movimiento
                moving_horizontal = current_y != y  # Si cambia Y, estamos moviéndonos horizontalmente (Este-Oeste)
                
                # Si el semáforo está en rojo para su dirección
                if (moving_horizontal and light_state == 1) or (not moving_horizontal and light_state == 0):
                    self.waiting_time += 1
                    return  # No moverse si el semáforo está en rojo
        
        # Moverse a la siguiente posición
        self.position = next_position
        self.route.pop(0)
        
        # Verificar si ha llegado al destino
        if not self.route:
            self.reached_destination = True
            
    def replan_route(self):
        """Replanifica la ruta si es necesario (por ejemplo, si está atascado)"""
        self.route = self.plan_route()