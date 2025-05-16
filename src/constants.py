
# --------------- GENERAL SETTINGS ----------------------------

GRID_SIZE = 10  # Tamaño de la cuadrícula de la ciudad
MAX_CARS = 20   # Máximo número de coches en la simulación
NUM_INTERSECTIONS = 9  # Número de intersecciones con semáforos
LIGHT_CYCLE = 5  # Ciclo básico de semáforo (en pasos de simulación)

#--------------- FINE TUNING HYPERPARAMETERS ------------------

SIMULATION_STEPS = 1000  # Pasos de simulación total
BATCH_SIZE = 64
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 10000
LEARNING_RATE = 0.001
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 10000
TARGET_UPDATE = 10
STATE_SIZE = 10  # Tamaño del estado para el agente RL
ACTION_SIZE = 4  # Número de acciones posibles para cada semáforo (cambiar duración)