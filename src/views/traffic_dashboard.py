from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

class TrafficDashboard:
    """Dashboard para visualizar y analizar resultados de la simulación"""
    
    def __init__(self, simulation, results=None):
        self.simulation = simulation
        self.fig = plt.figure(figsize=(18, 12))
        self.gs = self.fig.add_gridspec(3, 3)
        self.results = results
        self.setup_plots()
        
    def setup_plots(self):
        """Configura los diferentes componentes del dashboard"""
        # Mapa de congestión
        self.ax_map = self.fig.add_subplot(self.gs[0:2, 0:2])
        self.ax_map.set_title('Mapa de Congestión')
        
        # Gráfico de congestión global
        self.ax_congestion = self.fig.add_subplot(self.gs[0, 2])
        self.ax_congestion.set_title('Congestión Global')
        self.ax_congestion.set_xlabel('Paso')
        self.ax_congestion.set_ylabel('Nivel')
        
        # Gráfico de tiempo de espera
        self.ax_waiting = self.fig.add_subplot(self.gs[1, 2])
        self.ax_waiting.set_title('Tiempo de Espera Promedio')
        self.ax_waiting.set_xlabel('Paso')
        self.ax_waiting.set_ylabel('Tiempo (pasos)')
        
        # Tabla de métricas
        self.ax_metrics = self.fig.add_subplot(self.gs[2, 0])
        self.ax_metrics.axis('off')
        self.ax_metrics.set_title('Métricas')
        
        # Gráfico de throughput
        self.ax_throughput = self.fig.add_subplot(self.gs[2, 1])
        self.ax_throughput.set_title('Throughput')
        self.ax_throughput.set_xlabel('Paso')
        self.ax_throughput.set_ylabel('Coches completados')
        
        # Comparativa antes/después
        self.ax_compare = self.fig.add_subplot(self.gs[2, 2])
        self.ax_compare.set_title('Comparativa Antes/Después')
        self.ax_compare.axis('off')
        
        plt.tight_layout()
        
    def update(self):
        """Actualiza todos los componentes del dashboard con datos actuales"""
        self._update_congestion_map()
        self._update_congestion_graph()
        self._update_waiting_time_graph()
        self._update_throughput_graph()
        self._update_metrics_table()
        self._update_comparison()
        
        plt.tight_layout()
        plt.draw()
        
    def _update_congestion_map(self):
        """Actualiza el mapa de congestión"""
        self.ax_map.clear()
        self.ax_map.set_title('Mapa de Congestión')
        
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
        self.ax_map.imshow(self.simulation.city.grid, cmap=cmap)
        
        # Mostrar semáforos
        for (x, y), state in self.simulation.city.traffic_lights.items():
            color = 'green' if state == 0 else 'red'
            circle = plt.Circle((y, x), 0.2, color=color)
            self.ax_map.add_patch(circle)
        
        # Mostrar coches
        for car in self.simulation.city.cars:
            x, y = car.position
            rect = plt.Rectangle((y - 0.3, x - 0.3), 0.6, 0.6, 
                               color='yellow' if not car.reached_destination else 'green')
            self.ax_map.add_patch(rect)
        
        # Mostrar densidad de tráfico como mapa de calor transparente
        density_mask = self.simulation.city.traffic_density > 0
        self.ax_map.imshow(np.ma.masked_where(~density_mask, self.simulation.city.traffic_density), 
                        cmap='Reds', alpha=0.5)
    
    def _update_congestion_graph(self):
        """Actualiza el gráfico de congestión global"""
        self.ax_congestion.clear()
        self.ax_congestion.set_title('Congestión Global')
        self.ax_congestion.set_xlabel('Paso')
        self.ax_congestion.set_ylabel('Nivel')
        
        if self.simulation.metrics['congestion']:
            self.ax_congestion.plot(self.simulation.metrics['congestion'])
    
    def _update_waiting_time_graph(self):
        """Actualiza el gráfico de tiempo de espera promedio"""
        self.ax_waiting.clear()
        self.ax_waiting.set_title('Tiempo de Espera Promedio')
        self.ax_waiting.set_xlabel('Paso')
        self.ax_waiting.set_ylabel('Tiempo (pasos)')
        
        if self.simulation.metrics['avg_waiting_time']:
            self.ax_waiting.plot(self.simulation.metrics['avg_waiting_time'])
    
    def _update_throughput_graph(self):
        """Actualiza el gráfico de throughput"""
        self.ax_throughput.clear()
        self.ax_throughput.set_title('Throughput')
        self.ax_throughput.set_xlabel('Paso')
        self.ax_throughput.set_ylabel('Coches completados')
        
        if self.simulation.metrics['throughput']:
            self.ax_throughput.plot(self.simulation.metrics['throughput'])
    
    def _update_metrics_table(self):
        """Actualiza la tabla de métricas"""
        self.ax_metrics.clear()
        self.ax_metrics.axis('off')
        self.ax_metrics.set_title('Métricas')
        
        # Calcular métricas actuales
        metrics = {
            'Congestión promedio': np.mean(self.simulation.metrics['congestion'][-10:]) if self.simulation.metrics['congestion'] else 0,
            'Tiempo de espera promedio': np.mean(self.simulation.metrics['avg_waiting_time'][-10:]) if self.simulation.metrics['avg_waiting_time'] else 0,
            'Coches completados': self.simulation.metrics['throughput'][-1] if self.simulation.metrics['throughput'] else 0,
            'Paso actual': self.simulation.step_count,
            'Coches activos': len([car for car in self.simulation.city.cars if not car.reached_destination])
        }
        
        # Crear tabla
        table_data = [[k, f"{v:.2f}" if isinstance(v, float) else v] for k, v in metrics.items()]
        table = self.ax_metrics.table(cellText=table_data, colLabels=['Métrica', 'Valor'], 
                                    loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
    
    def _update_comparison(self):
        """Actualiza la comparativa antes/después"""
        self.ax_compare.clear()
        self.ax_compare.axis('off')
        self.ax_compare.set_title('Comparativa RL vs. Ciclo Fijo')
        
        # Aquí podrías mostrar una comparación de los resultados con y sin RL
        # Por ahora mostraremos un texto placeholder
        if self.results:
            # Mostrar mejoras porcentuales
            improvements = {
                'Congestión': self.results['congestion_improvement'],
                'Tiempo de espera': self.results['waiting_improvement'],
                'Throughput': self.results['throughput_improvement']
            }
            
            comparison_text = "\n".join([f"{k}: {v:.2f}%" for k, v in improvements.items()])
        else:
            # Placeholder si no hay resultados
            comparison_text = "Resultados de comparación no disponibles."
            
        comparison_text = f"""
        Mejora con Aprendizaje por Refuerzo:
        - Congestión: {improvements['Congestión']:.2f}%
        - Tiempo de espera: {improvements['Tiempo de espera']:.2f}%
        - Throughput: {improvements['Throughput']:.2f}%
        """
        self.ax_compare.text(0.1, 0.5, comparison_text, fontsize=10)
    
    def show(self):
        """Muestra el dashboard"""
        plt.tight_layout()
        plt.show()
    
    def save(self, filename="dashboard.png"):
        """Guarda el dashboard como imagen"""
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')