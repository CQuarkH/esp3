import numpy as np

class City:
    """Representación de la ciudad con calles, intersecciones y semáforos"""
    
    def __init__(self, size=10):
        self.size = size
        # Cuadrícula para representar la ciudad: 0=vacío, 1=calle, 2=edificio, 3=intersección
        self.grid = np.zeros((size, size), dtype=int)
        self.traffic_lights = {}  # Diccionario para almacenar los semáforos {(x,y): estado}
        self.cars = []  # Lista para almacenar los objetos Car
        self.traffic_density = np.zeros((size, size))  # Para medir densidad de tráfico
        self.congestion_history = []  # Historial de congestión global
        self.lane_dir = np.full((size, size), fill_value=-1, dtype=int)
        self.setup_city()
        
    def setup_city(self):
        """Configura la estructura básica de la ciudad con calles e intersecciones"""
        # Crear calles principales horizontales y verticales
        for i in range(2, self.size, 3):
            # Calles horizontales
            self.grid[i, :] = 1
            # Calles verticales
            self.grid[:, i] = 1
            self.grid[i, :] = 1
            self.lane_dir[i, :] = (1 if i % 6 == 2 else 3)
            
            self.grid[:, i] = 1
            self.lane_dir[:, i] = (2 if i % 6 == 2 else 0)
            
        # Establecer intersecciones y semáforos
        for row in range(2, self.size, 3):
            for col in range(2, self.size, 3):
                self.grid[row, col] = 3  # Marcar como intersección
                self.lane_dir[row, col] = -1
                # Estado del semáforo: 0=verde horizontal, 1=verde vertical, inicialmente alternados
                self.traffic_lights[(row, col)] = (row + col) % 2
                
        # Colocar edificios (espacios entre calles)
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == 0:  # Si no es calle ni intersección
                    self.grid[i, j] = 2  # Marcar como edificio
    
    def add_car(self, car):
        """Añade un coche a la simulación"""
        self.cars.append(car)
    
    def remove_car(self, car):
        """Elimina un coche de la simulación"""
        if car in self.cars:
            self.cars.remove(car)
            
    def update(self):
        """Actualiza el estado de la ciudad, semáforos y coches"""
        # Actualizar posición de coches
        for car in self.cars:
            car.move()
            
        # Actualizar mapa de densidad de tráfico
        self.update_traffic_density()
        
        # Calcular y registrar congestión total
        congestion = self.calculate_congestion()
        self.congestion_history.append(congestion)
        
    def update_traffic_density(self):
        """Actualiza el mapa de densidad de tráfico basado en posiciones de coches"""
        # Reiniciar matriz de densidad
        self.traffic_density = np.zeros((self.size, self.size))
        
        # Incrementar densidad en posiciones donde hay coches
        for car in self.cars:
            x, y = car.position
            self.traffic_density[int(x), int(y)] += 1
            
        # Normalizar para visualización
        if self.cars:
            self.traffic_density = self.traffic_density / max(1, np.max(self.traffic_density))
    
    def calculate_congestion(self):
        """Calcula el nivel de congestión global como la suma de densidades de tráfico"""
        return np.sum(self.traffic_density)
    
    def get_state_for_intersection(self, intersection):
        """Obtiene el estado del entorno para una intersección específica"""
        x, y = intersection
        state = []
        
        # 1. Estado actual del semáforo
        state.append(self.traffic_lights[intersection])
        
        # 2. Densidad de tráfico en los cuatro segmentos adyacentes
        # Norte
        north_density = sum(self.traffic_density[max(0, x-3):x, y]) if x > 0 else 0
        # Sur
        south_density = sum(self.traffic_density[x+1:min(self.size, x+4), y]) if x < self.size-1 else 0
        # Este
        east_density = sum(self.traffic_density[x, y+1:min(self.size, y+4)]) if y < self.size-1 else 0
        # Oeste
        west_density = sum(self.traffic_density[x, max(0, y-3):y]) if y > 0 else 0
        
        state.extend([north_density, south_density, east_density, west_density])
        
        # 3. Tiempo desde el último cambio de estado
        # Esto requiere un contador adicional que podríamos implementar
        # Por ahora usamos un valor constante como placeholder
        state.append(0.5)
        
        # 4. Congestión global actual
        state.append(self.calculate_congestion())
        
        # 5. Tendencia de congestión (diferencia con respecto al paso anterior)
        if len(self.congestion_history) > 1:
            trend = self.congestion_history[-1] - self.congestion_history[-2]
        else:
            trend = 0
        state.append(trend)
        
        # 6. Hora del día simulada (cíclica entre 0 y 1)
        time_of_day = (len(self.congestion_history) % 24) / 24.0
        state.append(time_of_day)
        
        # 7. Día de la semana simulado (cíclico entre 0 y 1)
        day_of_week = ((len(self.congestion_history) // 24) % 7) / 7.0
        state.append(day_of_week)
        
        return np.array(state, dtype=np.float32)