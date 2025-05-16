from .traffic_simulation import TrafficSimulation

from constants import LIGHT_CYCLE

class FixedCycleTrafficSimulation(TrafficSimulation):
    """Simulación de tráfico con semáforos de ciclo fijo (línea base para comparación)"""
    
    def __init__(self):
        super().__init__()
        # No inicializamos agentes RL
        self.agents = {}
    
    def run_step(self, training=False):
        """Ejecuta un paso de la simulación con lógica de ciclo fijo"""
        self.step_count += 1
        
        # Cambiar estado de semáforos basado en ciclo fijo
        for intersection in self.city.traffic_lights.keys():
            # Cambiar cada LIGHT_CYCLE pasos
            if self.step_count % LIGHT_CYCLE == 0:
                self.city.traffic_lights[intersection] = 1 - self.city.traffic_lights[intersection]
        
        # Actualizar el estado de la ciudad
        self.city.update()
        
        # Recolectar métricas
        self.collect_metrics()
        
        # Cada ciertos pasos, reemplazar coches que llegaron a su destino
        if self.step_count % 20 == 0:
            self.replace_finished_cars()