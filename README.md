# Simulador de Tráfico con Aprendizaje por Refuerzo

Este proyecto implementa un simulador de tráfico urbano con intersecciones controladas por semáforos, que puede funcionar con estrategias de tiempo fijo o mediante un controlador de aprendizaje por refuerzo (RL) que optimiza el flujo de tráfico.

## Características

- Simulación de tráfico en una cuadrícula urbana con 4 intersecciones
- Múltiples vehículos que se desplazan por la ciudad respetando los semáforos
- Dos modos de operación:
  - Semáforos de tiempo fijo (modo tradicional)
  - Controlador inteligente basado en aprendizaje por refuerzo

## Requisitos

- Python 3.7+
- TensorFlow 2.x
- PyGame
- NumPy
- Matplotlib (para visualizar resultados de entrenamiento)

Instalar dependencias:
```
pip install tensorflow pygame numpy matplotlib
```

## Uso

El programa puede ejecutarse en tres modos diferentes:

### 1. Modo de semáforos fijos (tradicional)

```
python main.py --mode fixed
```

### 2. Entrenamiento del modelo de RL

```
python main.py --mode rl_train --episodes 200
```

Donde:
- `--episodes`: Número de episodios de entrenamiento (por defecto: 100)

El modelo entrenado se guardará como `traffic_model_final.h5` y se generará un gráfico de las recompensas durante el entrenamiento.

### 3. Prueba del modelo de RL entrenado

```
python main.py --mode rl_test --model traffic_model_final.h5
```

Donde:
- `--model`: Ruta al modelo previamente entrenado

## Cómo funciona el aprendizaje por refuerzo

El controlador RL utiliza Deep Q-Learning para aprender la política óptima de control de semáforos:

1. **Estado**: Para cada intersección se observa:
   - Cola de vehículos en dirección Norte-Sur
   - Velocidad promedio en dirección Norte-Sur
   - Cola de vehículos en dirección Este-Oeste
   - Velocidad promedio en dirección Este-Oeste

2. **Acciones**: Para cada semáforo:
   - 0: Luz verde para dirección Norte-Sur
   - 1: Luz verde para dirección Este-Oeste

3. **Recompensa**: Penalización basada en la congestión:
   - Menor cantidad de vehículos detenidos = mayor recompensa
   - Mayor velocidad promedio = mayor recompensa

El agente aprende a coordinar los semáforos para minimizar el tiempo de espera y maximizar el flujo de tráfico, adaptándose dinámicamente a las condiciones de tráfico.

## Arquitectura del modelo

- Red neuronal con capas densas (24-24-n_actions)
- Política epsilon-greedy para balance entre exploración y explotación
- Memoria de experiencias para entrenamiento por lotes
- Modelo target para estabilizar el entrenamiento

## Información adicional

La simulación muestra estadísticas en tiempo real:
- Número total de vehículos
- Número de vehículos detenidos
- Representación visual del estado de los semáforos

Los vehículos en amarillo indican que están detenidos en un semáforo.