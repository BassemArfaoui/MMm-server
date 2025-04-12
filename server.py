import numpy as np
from scipy.stats import expon
import math
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
import matplotlib.pyplot as plt
import io
import base64
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Simulation M/M/m")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.options("/{path:path}")
async def options_handler(request: Request, path: str):
    return Response(status_code=200, headers={
        "Access-Control-Allow-Origin": "http://localhost:3000",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    })

simulation_cache: Dict[str, dict] = {}

class SimulationParams(BaseModel):
    m: int
    lambda_: float
    mu: float
    num_transitions: int = 10000
    last_n: int = 5000

def simulate_mm_queue(m: int, lambda_: float, mu: float, num_transitions: int):
    times = [0]
    states = [0]
    arrival_times = [0]
    arrivals = [0]
    current_state = 0
    current_time = 0
    arrival_count = 0

    for _ in range(num_transitions):
        arrival_rate = lambda_
        service_rate = mu * min(current_state, m) if current_state > 0 else 0
        total_rate = arrival_rate + service_rate

        time_to_event = expon.rvs(scale=1/total_rate)
        current_time += time_to_event

        if current_state == 0:
            p_arrival = 1.0
        else:
            p_arrival = arrival_rate / total_rate
        if np.random.random() < p_arrival:
            current_state += 1
            arrival_count += 1
            arrival_times.append(current_time)
            arrivals.append(arrival_count)
        else:
            current_state -= 1

        times.append(current_time)
        states.append(current_state)

    return times, states, arrival_times, arrivals

def theoretical_pi(m: int, lambda_: float, mu: float, max_k: int = 50):
    a = lambda_ / mu
    pi_0 = 0
    for k in range(m):
        pi_0 += (a ** k) / math.factorial(k)
    pi_0 += (a ** m) / (math.factorial(m) * (1 - a/m))
    pi_0 = 1 / pi_0

    pi = []
    for k in range(max_k + 1):
        if k < m:
            pi_k = ((a ** k) / math.factorial(k)) * pi_0
        else:
            pi_k = ((a ** k) / (m ** (k - m) * math.factorial(m))) * pi_0
        pi.append(pi_k)
    return np.array(pi)

def theoretical_expected_n(m: int, lambda_: float, mu: float):
    a = lambda_ / mu
    pi_0 = 0
    for k in range(m):
        pi_0 += (a ** k) / math.factorial(k)
    pi_0 += (a ** m) / (math.factorial(m) * (1 - a/m))
    pi_0 = 1 / pi_0

    E_N_s = a
    P_q = ((a ** m) / math.factorial(m)) * pi_0 / (1 - a/m)
    E_N_q = P_q * (a/m) / (1 - a/m)
    return E_N_s + E_N_q

def plot_to_base64_step(times: List[float], values: List[int], title: str, xlabel: str, ylabel: str, color: str = 'blue', figsize=(20, 5)):
    plt.figure(figsize=figsize)
    plt.step(times, values, where='post', color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    buf.close()
    return img_base64

def plot_to_base64_histogram(empirical_pi: List[float], theoretical_pi: List[float], title: str, xlabel: str, ylabel: str, figsize=(10, 5)):
    plt.figure(figsize=figsize)
    plt.bar(range(len(empirical_pi)), empirical_pi, alpha=0.5, label='Empirique (Derniers 5000)')
    plt.plot(range(len(theoretical_pi)), theoretical_pi, 'r.-', label='Théorique')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    buf.close()
    return img_base64

def run_simulation(params: SimulationParams):
    cache_key = f"{params.m}_{params.lambda_}_{params.mu}_{params.num_transitions}_{params.last_n}"
    if cache_key in simulation_cache:
        return simulation_cache[cache_key]

    rho = params.lambda_ / (params.m * params.mu)
    if rho >= 1:
        raise ValueError("Le système n'est pas stable : ρ = λ/(mμ) ≥ 1")

    times, states, arrival_times, arrivals = simulate_mm_queue(params.m, params.lambda_, params.mu, params.num_transitions)

    empirical_states = states[-params.last_n:]
    max_state = max(empirical_states)
    bins = np.arange(max_state + 2) - 0.5
    hist, bin_edges = np.histogram(empirical_states, bins=bins, density=True)
    empirical_pi = hist.tolist()

    theoretical_pi_vals = theoretical_pi(params.m, params.lambda_, params.mu, max_k=max_state).tolist()

    empirical_E_N = float(np.mean(empirical_states))
    theoretical_E_N = theoretical_expected_n(params.m, params.lambda_, params.mu)

    full_plot_base64 = plot_to_base64_step(
        times, states,
        title=f'Évolution M/M/{params.m} (Complète, λ={params.lambda_}, μ={params.mu})',
        xlabel='Temps',
        ylabel='Nombre de clients',
        figsize=(20, 5)
    )
    simplified_plot_base64 = plot_to_base64_step(
        times[:501], states[:501],
        title=f'Évolution M/M/{params.m} (Premières 500 transitions, λ={params.lambda_}, μ={params.mu})',
        xlabel='Temps',
        ylabel='Nombre de clients',
        figsize=(20, 5)
    )
    arrivals_plot_base64 = plot_to_base64_step(
        arrival_times, arrivals,
        title=f'Arrivées cumulées au fil du temps (λ={params.lambda_})',
        xlabel='Temps',
        ylabel='Arrivées cumulées',
        color='green',
        figsize=(10, 5)
    )
    histogram_plot_base64 = plot_to_base64_histogram(
        empirical_pi, theoretical_pi_vals,
        title='Distribution stationnaire : Empirique vs Théorique',
        xlabel='Nombre de clients (k)',
        ylabel='Probabilité (π_k)',
        figsize=(10, 5)
    )

    result = {
        "times": times,
        "states": states,
        "arrival_times": arrival_times,
        "arrivals": arrivals,
        "empirical_pi": empirical_pi,
        "theoretical_pi": theoretical_pi_vals,
        "empirical_E_N": empirical_E_N,
        "theoretical_E_N": theoretical_E_N,
        "full_plot_base64": full_plot_base64,
        "simplified_plot_base64": simplified_plot_base64,
        "arrivals_plot_base64": arrivals_plot_base64,
        "histogram_plot_base64": histogram_plot_base64
    }
    simulation_cache[cache_key] = result
    return result

@app.get("/")
async def root():
    return {"message": "Bienvenue dans l'API de Simulation de File d’Attente M/M/m !"}

@app.post("/queue-evolution/full")
async def get_queue_evolution_full(params: SimulationParams):
    try:
        print("Paramètres reçus :", params)
        result = run_simulation(params)
        return {
            "plot_base64": result["full_plot_base64"],
            "empirical_E_N": result["empirical_E_N"],
            "theoretical_E_N": result["theoretical_E_N"],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/queue-evolution/simplified")
async def get_queue_evolution_simplified(params: SimulationParams):
    try:
        result = run_simulation(params)
        return {
            "empirical_E_N": result["empirical_E_N"],
            "theoretical_E_N": result["theoretical_E_N"],
            "plot_base64": result["simplified_plot_base64"]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/arrivals")
async def get_arrivals(params: SimulationParams):
    try:
        result = run_simulation(params)
        return {
            "empirical_E_N": result["empirical_E_N"],
            "theoretical_E_N": result["theoretical_E_N"],
            "plot_base64": result["arrivals_plot_base64"]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/stationary-distribution")
async def get_stationary_distribution(params: SimulationParams):
    try:
        result = run_simulation(params)
        return {
            "empirical_pi": result["empirical_pi"],
            "theoretical_pi": result["theoretical_pi"],
            "empirical_E_N": result["empirical_E_N"],
            "theoretical_E_N": result["theoretical_E_N"],
            "plot_base64": result["histogram_plot_base64"]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/simulation-results")
async def get_simulation_results(params: SimulationParams):
    try:
        result = run_simulation(params)
        return {
            "arrival_times": result["arrival_times"],
            "arrivals": result["arrivals"],
            "empirical_pi": result["empirical_pi"],
            "theoretical_pi": result["theoretical_pi"],
            "empirical_E_N": result["empirical_E_N"],
            "theoretical_E_N": result["theoretical_E_N"],
            "absolute_difference_E_N": abs(result["theoretical_E_N"] - result["empirical_E_N"])
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)