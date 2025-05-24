# train_agent.py
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle # Para guardar la Q-table y las recompensas
import json

def train_q_learning_agent(env_name="FrozenLake-v1", params=None, save_results=True):
    """
    Entrena un agente Q-Learning en el entorno especificado.

    Args:
        env_name (str): Nombre del entorno de Gymnasium.
        params (dict): Diccionario de hiperparámetros.
                       Ej: {'num_episodes': 10000, 'alpha': 0.1, 'gamma': 0.99,
                            'epsilon': 1.0, 'epsilon_decay': 0.999, 'min_epsilon': 0.01}
        save_results (bool): Si es True, guarda la Q-table, recompensas y gráfico.

    Returns:
        tuple: Q-table aprendida, lista de recompensas por episodio.
    """
    if params is None:
        params = {
            'num_episodes': 20000,       # Número total de episodios para entrenamiento
            'max_steps_per_episode': 100,# Máximos pasos por episodio
            'alpha': 0.1,               # Tasa de aprendizaje (learning rate)
            'gamma': 0.99,              # Factor de descuento
            'epsilon': 1.0,             # Tasa de exploración inicial
            'epsilon_decay': 0.9995,    # Factor de decaimiento de épsilon (para reducir la exploración)
            'min_epsilon': 0.01         # Mínima tasa de exploración
        }

    # is_slippery=True es el comportamiento por defecto y más desafiante
    env = gym.make(env_name, is_slippery=True)
    
    # Inicializar Q-table con ceros
    # Filas: estados, Columnas: acciones
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    
    rewards_all_episodes = []
    
    print(f"Entrenando con parámetros: {params}")

    for episode in range(params['num_episodes']):
        state, info = env.reset()
        terminated = False
        truncated = False
        current_episode_rewards = 0
        
        for step in range(params['max_steps_per_episode']):
            # Exploración vs Explotación (epsilon-greedy)
            if np.random.uniform(0, 1) < params['epsilon']:
                action = env.action_space.sample() # Explorar: acción aleatoria
            else:
                action = np.argmax(q_table[state, :]) # Explotar: mejor acción conocida
            
            new_state, reward, terminated, truncated, info = env.step(action)
            
            # Actualizar Q-value para la tupla (state, action)
            q_table[state, action] = q_table[state, action] * (1 - params['alpha']) + \
                                     params['alpha'] * (reward + params['gamma'] * np.max(q_table[new_state, :]))
            
            state = new_state
            current_episode_rewards += reward
            
            if terminated or truncated:
                break
        
        # Decaimiento de Epsilon
        params['epsilon'] = max(params['min_epsilon'], params['epsilon'] * params['epsilon_decay'])
        
        rewards_all_episodes.append(current_episode_rewards)

        if (episode + 1) % (params['num_episodes'] // 10) == 0:
            print(f"Episodio {episode + 1}/{params['num_episodes']} - Epsilon: {params['epsilon']:.2f} - Recompensa promedio (últimos 100): {np.mean(rewards_all_episodes[-100:]):.3f}")

    env.close()
    
    print("\nEntrenamiento completado.")
    
    if save_results:
        # Guardar Q-table
        with open('static/data/q_table.pkl', 'wb') as f:
            pickle.dump(q_table, f)
        print("Q-table guardada en static/data/q_table.pkl")

        # Guardar recompensas
        with open('static/data/rewards.json', 'w') as f:
            json.dump(rewards_all_episodes, f)
        print("Recompensas guardadas en static/data/rewards.json")

        # Graficar y guardar recompensas
        plt.figure(figsize=(12, 6))
        # Suavizar la curva de recompensas para mejor visualización
        rewards_smoothed = np.convolve(rewards_all_episodes, np.ones(100)/100, mode='valid')
        plt.plot(rewards_smoothed)
        plt.title('Recompensa Promedio Móvil por Episodio (Ventana de 100 episodios)')
        plt.xlabel('Episodio')
        plt.ylabel('Recompensa Promedio Móvil')
        plt.grid(True)
        plt.savefig('static/images/rewards_plot.png')
        print("Gráfico de recompensas guardado en static/images/rewards_plot.png")
        # plt.show() # Descomentar si quieres ver el gráfico inmediatamente

    return q_table, rewards_all_episodes

def evaluate_agent(q_table, env_name="FrozenLake-v1", num_episodes=100, render=False):
    """
    Evalúa el agente entrenado.
    """
    env_render_mode = "human" if render else None
    eval_env = gym.make(env_name, is_slippery=True, render_mode=env_render_mode)
    
    total_rewards_eval = 0
    successful_episodes = 0
    
    print("\nEvaluando agente entrenado...")
    for episode in range(num_episodes):
        state, info = eval_env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        
        while not terminated and not truncated:
            action = np.argmax(q_table[state, :]) # Tomar la mejor acción según la Q-table
            new_state, reward, terminated, truncated, info = eval_env.step(action)
            state = new_state
            episode_reward += reward
            if render:
                eval_env.render() # Muestra el entorno (si está habilitado)
                # time.sleep(0.1) # Pequeña pausa para ver la simulación

        total_rewards_eval += episode_reward
        if episode_reward > 0: # En FrozenLake, >0 significa éxito
            successful_episodes += 1

    eval_env.close()
    
    avg_reward = total_rewards_eval / num_episodes
    success_rate = successful_episodes / num_episodes
    print(f"Resultados de la evaluación ({num_episodes} episodios):")
    print(f"  - Recompensa promedio: {avg_reward:.3f}")
    print(f"  - Tasa de éxito: {success_rate*100:.2f}%")
    return avg_reward, success_rate

def get_policy_from_q_table(q_table):
    """ Extrae la política óptima (greedy) de la Q-table. """
    return np.argmax(q_table, axis=1)

def run_simulation_steps(q_table, env_name="FrozenLake-v1", max_steps=20):
    """
    Ejecuta una simulación y devuelve los pasos.
    """
    # Usamos rgb_array para poder capturar el estado visual si quisiéramos,
    # aunque aquí solo devolveremos la descripción textual.
    env = gym.make(env_name, is_slippery=True, render_mode="ansi") 
    state, info = env.reset()
    
    simulation_steps = []
    action_map = {0: "Izquierda", 1: "Abajo", 2: "Derecha", 3: "Arriba"}

    for step in range(max_steps):
        action = np.argmax(q_table[state, :])
        
        # Capturar el estado del entorno ANTES de la acción
        # env.render() devuelve la representación textual con render_mode="ansi"
        # Para FrozenLake, no es tan visual como el render "human", pero sirve para el log
        # current_render = env.render() # Esto imprime a consola, no lo capturamos aquí directamente
                                      # pero podríamos si quisiéramos para FrozenLake.
                                      # En otros entornos, 'rgb_array' sería más útil.
        
        new_state, reward, terminated, truncated, info = env.step(action)
        
        step_info = {
            "step": step + 1,
            "state": state,
            "action_code": int(action),
            "action_name": action_map[int(action)],
            "reward": reward,
            "new_state": new_state,
            "terminated": terminated,
            "truncated": truncated,
            "board": env.render() # Captura la representación ANSI del tablero DESPUÉS de la acción
        }
        simulation_steps.append(step_info)
        
        state = new_state
        if terminated or truncated:
            break
            
    env.close()
    return simulation_steps

if __name__ == "__main__":
    # Crear directorios si no existen
    import os
    os.makedirs('static/data', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)

    # --- Entrenamiento y Evaluación ---
    print("--- INICIANDO ENTRENAMIENTO DEL AGENTE ---")
    # Parámetros para una configuración
    params1 = {
        'num_episodes': 20000,
        'max_steps_per_episode': 100,
        'alpha': 0.1,
        'gamma': 0.99,
        'epsilon': 1.0,
        'epsilon_decay': 0.9995, # Decaimiento más lento
        'min_epsilon': 0.01
    }
    q_table_1, rewards_1 = train_q_learning_agent(params=params1, save_results=True) # save_results=True para guardar la primera config
    avg_reward_1, success_rate_1 = evaluate_agent(q_table_1, num_episodes=1000)
    policy_1 = get_policy_from_q_table(q_table_1)
    print("Política aprendida (Configuración 1):")
    print(policy_1.reshape(4,4)) # Asumiendo entorno 4x4

    # --- (Opcional) Comparación con otra configuración ---
    print("\n--- ENTRENANDO CON OTRA CONFIGURACIÓN (SOLO PARA COMPARACIÓN INTERNA) ---")
    params2 = {
        'num_episodes': 20000,
        'max_steps_per_episode': 100,
        'alpha': 0.05, # Tasa de aprendizaje más baja
        'gamma': 0.99,
        'epsilon': 1.0,
        'epsilon_decay': 0.999, # Decaimiento más rápido de epsilon
        'min_epsilon': 0.05     # Epsilon mínimo más alto (más exploración residual)
    }
    # No guardamos estos resultados como los principales, solo para demostración
    q_table_2, rewards_2 = train_q_learning_agent(params=params2, save_results=False)
    avg_reward_2, success_rate_2 = evaluate_agent(q_table_2, num_episodes=1000)
    policy_2 = get_policy_from_q_table(q_table_2)

    print(f"\nComparación de configuraciones (evaluación en 1000 episodios):")
    print(f"Config 1 (alpha={params1['alpha']}, eps_decay={params1['epsilon_decay']}): Avg Reward: {avg_reward_1:.3f}, Success: {success_rate_1*100:.2f}%")
    print(f"Config 2 (alpha={params2['alpha']}, eps_decay={params2['epsilon_decay']}): Avg Reward: {avg_reward_2:.3f}, Success: {success_rate_2*100:.2f}%")

    # Puedes visualizar la política de la primera configuración así:
    # action_map_arrows = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    # policy_arrows_1 = [action_map_arrows[action] for action in policy_1]
    # print("\nPolítica visual (Configuración 1):")
    # print(np.array(policy_arrows_1).reshape(4,4))

    print("\nPara iniciar la aplicación Flask, ejecuta: python app.py")