# app.py
from flask import Flask, render_template, url_for
import numpy as np
import pickle
import json
import os

# Importar funciones del script de entrenamiento
# Asegúrate de que train_agent.py esté en el mismo directorio o en PYTHONPATH
from train_agent import get_policy_from_q_table, run_simulation_steps, evaluate_agent

app = Flask(__name__)

Q_TABLE_PATH = 'static/data/q_table.pkl'
REWARDS_PATH = 'static/data/rewards.json'

def load_q_table():
    if os.path.exists(Q_TABLE_PATH):
        with open(Q_TABLE_PATH, 'rb') as f:
            return pickle.load(f)
    return None

def load_rewards():
    if os.path.exists(REWARDS_PATH):
        with open(REWARDS_PATH, 'r') as f:
            return json.load(f)
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/entorno')
def entorno_page():
    return render_template('entorno.html')

@app.route('/resultados')
def resultados_page():
    q_table = load_q_table()
    rewards_data = load_rewards() # No se usa directamente en la plantilla pero es bueno tenerlo

    if q_table is None:
        return "Error: Q-table no encontrada. Ejecuta train_agent.py primero.", 404

    policy_numeric = get_policy_from_q_table(q_table)
    action_map_arrows = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    policy_visual = [action_map_arrows[action] for action in policy_numeric]
    
    # Para mostrar algunas métricas de evaluación en la página de resultados
    # Esto re-evaluará cada vez que se cargue la página, podría ser costoso para muchas ep.
    # Alternativamente, podrías guardar los resultados de la evaluación en train_agent.py
    # y cargarlos aquí. Por simplicidad, lo re-evaluamos.
    avg_reward, success_rate = evaluate_agent(q_table, num_episodes=100) # Evaluar en menos episodios para rapidez en web
    evaluation_results = {"avg_reward": f"{avg_reward:.3f}", "success_rate": success_rate}

    return render_template('resultados.html', 
                           q_table=q_table.tolist(), # Convertir a lista para Jinja
                           policy=policy_visual,
                           evaluation_results=evaluation_results)

@app.route('/simulacion')
def simulacion_page():
    q_table = load_q_table()
    if q_table is None:
        return "Error: Q-table no encontrada. Ejecuta train_agent.py primero.", 404

    # Ejecutar una simulación para obtener los pasos
    # Usamos el nombre del entorno como en train_agent.py
    simulation_data = run_simulation_steps(q_table, env_name="FrozenLake-v1", max_steps=25)
    
    return render_template('simulacion.html', simulation_steps=simulation_data)

if __name__ == '__main__':
    # Asegurarse de que los directorios para datos estáticos existen
    os.makedirs('static/data', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    
    # Comprobar si los archivos de datos existen, si no, ofrecer entrenar
    if not os.path.exists(Q_TABLE_PATH) or not os.path.exists(REWARDS_PATH):
        print(f"ADVERTENCIA: {Q_TABLE_PATH} o {REWARDS_PATH} no encontrados.")
        print("Por favor, ejecuta 'python train_agent.py' primero para generar los datos del modelo.")
        # Podrías decidir salir aquí si no quieres que la app corra sin datos
        # import sys
        # sys.exit("Saliendo. Entrena al agente primero.")
    
    app.run(debug=True)