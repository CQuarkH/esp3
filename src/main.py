import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from views.traffic_dashboard import TrafficDashboard
from sims.traffic_simulation import TrafficSimulation
from sims.fixed_cycle_traffic_simulation import FixedCycleTrafficSimulation

# Configuración de semillas para reproducibilidad
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# Función para comparar resultados entre RL y ciclo fijo
def compare_simulations(steps=200):
    """Compara los resultados entre simulación con RL y con ciclo fijo"""
    # Simulación con RL
    print("Ejecutando simulación con RL...")
    rl_sim = TrafficSimulation()
    
    # Cargar modelos pre-entrenados si existen
    try:
        for intersection, agent in rl_sim.agents.items():
            agent.load(f"agent_intersection_{intersection[0]}_{intersection[1]}_ep10.pth")
        print("Modelos pre-entrenados cargados correctamente.")
    except:
        print("No se encontraron modelos pre-entrenados. Entrenando desde cero...")
        rl_sim.train(episodes=5, steps_per_episode=100)
    
    for _ in tqdm(range(steps)):
        rl_sim.run_step(training=False)
    
    # Simulación con ciclo fijo
    print("Ejecutando simulación con ciclo fijo...")
    fixed_sim = FixedCycleTrafficSimulation()
    for _ in tqdm(range(steps)):
        fixed_sim.run_step()
    
    # Comparar resultados
    plt.figure(figsize=(15, 12))
    
    # Congestión
    plt.subplot(3, 1, 1)
    plt.plot(rl_sim.metrics['congestion'], label='RL')
    plt.plot(fixed_sim.metrics['congestion'], label='Ciclo Fijo')
    plt.title('Comparativa de Congestión')
    plt.xlabel('Paso de simulación')
    plt.ylabel('Nivel de congestión')
    plt.legend()
    
    # Tiempo promedio de espera
    plt.subplot(3, 1, 2)
    plt.plot(rl_sim.metrics['avg_waiting_time'], label='RL')
    plt.plot(fixed_sim.metrics['avg_waiting_time'], label='Ciclo Fijo')
    plt.title('Comparativa de Tiempo Promedio de Espera')
    plt.xlabel('Paso de simulación')
    plt.ylabel('Pasos')
    plt.legend()
    
    # Throughput
    plt.subplot(3, 1, 3)
    plt.plot(rl_sim.metrics['throughput'], label='RL')
    plt.plot(fixed_sim.metrics['throughput'], label='Ciclo Fijo')
    plt.title('Comparativa de Throughput')
    plt.xlabel('Paso de simulación')
    plt.ylabel('Cantidad')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("comparison_results.png")
    plt.show()
    
    # Calcular mejoras porcentuales
    congestion_improvement = ((np.mean(fixed_sim.metrics['congestion']) - np.mean(rl_sim.metrics['congestion'])) 
                             / np.mean(fixed_sim.metrics['congestion'])) * 100
    
    waiting_improvement = ((np.mean(fixed_sim.metrics['avg_waiting_time']) - np.mean(rl_sim.metrics['avg_waiting_time'])) 
                          / np.mean(fixed_sim.metrics['avg_waiting_time'])) * 100
    
    throughput_improvement = ((np.mean(rl_sim.metrics['throughput']) - np.mean(fixed_sim.metrics['throughput'])) 
                             / np.mean(fixed_sim.metrics['throughput'])) * 100
    
    print(f"Mejoras con RL respecto a ciclo fijo:")
    print(f"- Reducción de congestión: {congestion_improvement:.2f}%")
    print(f"- Reducción de tiempo de espera: {waiting_improvement:.2f}%")
    print(f"- Aumento de throughput: {throughput_improvement:.2f}%")
    
    results = {
        'congestion_improvement': congestion_improvement,
        'waiting_improvement': waiting_improvement,
        'throughput_improvement': throughput_improvement
    }
    
    return rl_sim, fixed_sim, results

# Función principal
def main():
    """Función principal para ejecutar la simulación"""
    print("Sistema de Gestión de Tráfico Urbano con Aprendizaje por Refuerzo")
    print("=================================================================")
    print("\n1. Entrenando modelo de ML...")
    
    # Crear y entrenar simulación
    simulation = TrafficSimulation()
    simulation.train(episodes=10, steps_per_episode=200)
    
    print("\n2. Generando visualización de tráfico...")
    simulation.visualize_traffic()
    
    print("\n3. Creando animación de la simulación...")
    animation = simulation.generate_animation(steps=100)
    
    print("\n4. Comparando con semáforos de ciclo fijo...")
    rl_sim, fixed_sim, results = compare_simulations(steps=200)
    
    print("\n5. Mostrando dashboard con resultados...")
    dashboard = TrafficDashboard(rl_sim, results)
    dashboard.update()
    dashboard.save("dashboard_final.png")
    dashboard.show()
    
    print("\nSimulación completada. Resultados guardados en imágenes y archivos de modelos.")

if __name__ == "__main__":
    main()