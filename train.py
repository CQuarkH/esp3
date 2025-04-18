# train.py
import os
import argparse
import numpy as np
import tensorflow as tf
import gym
import random
from rl_traffic_controller import RLTrafficController
from main import CityTrafficSimEnv 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    args = parser.parse_args()

    # Inicializa entorno sin pygame/render
    env = CityTrafficSimEnv(max_vehicles=50)

    # Crea controlador y entrena
    controller = RLTrafficController(env, env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0])
    print(f"Entrenando {args.episodes} episodios …")
    scores = controller.train(episodes=args.episodes, render=False)

    # Guarda el modelo en el directorio que SageMaker monta como salida
    model_path = os.path.join(args.model_dir, 'traffic_model_final.h5')
    controller.agent.save(model_path)
    print(f"Modelo guardado en {model_path}")
