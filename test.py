import numpy as np
from scipy.stats import expon
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters (hardcoded for simplicity; can be made configurable)
m = 1  # Number of servers
lambda_ = 4.0  # Arrival rate (customers per hour)
mu = 5 # Service rate per server (customers per hour)
update_interval = 100  # Milliseconds between updates (10 updates/sec)

# Check stability
rho = lambda_ / (m * mu)
if rho >= 1:
    raise ValueError("System is not stable: ρ = λ/(mμ) ≥ 1")

# Simulation state
times = [0]
states = [0]
current_state = 0
current_time = 0

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6))
line, = ax.step(times, states, where='post', color='blue', label='Number of Customers')
ax.set_xlabel('Time (Scaled Hours)')
ax.set_ylabel('Number of Customers')
ax.set_title(f'M/M/{m} Queue Evolution (Live, λ={lambda_}, μ={mu})')
ax.grid(True)
ax.legend()
ax.set_ylim(0, 10)  # Initial y-axis range, will adjust dynamically
ax.set_xlim(0, 10)  # Initial x-axis range, will adjust dynamically

# Simulate one transition
def simulate_one_transition():
    global current_state, current_time
    
    # Current rates
    arrival_rate = lambda_
    service_rate = mu * min(current_state, m) if current_state > 0 else 0
    total_rate = arrival_rate + service_rate

    # Time to next event (scaled for visibility)
    time_to_event = expon.rvs(scale=1/total_rate) * 0.1  # Scale down time
    current_time += time_to_event

    # Decide event type
    if current_state == 0:
        p_arrival = 1.0
    else:
        p_arrival = arrival_rate / total_rate
    if np.random.random() < p_arrival:
        current_state += 1  # Arrival
    else:
        current_state -= 1  # Departure

    # Append to history
    times.append(current_time)
    states.append(current_state)

    # Limit to last 100 points for performance
    if len(times) > 100:
        times.pop(0)
        states.pop(0)

    return times, states

# Update function for animation
def update(frame):
    times, states = simulate_one_transition()
    
    # Update the step plot
    line.set_data(times, states)
    
    # Dynamically adjust axes
    ax.set_xlim(0, max(times) * 1.1)  # 10% padding
    ax.set_ylim(0, max(max(states) * 1.1, 5))  # Ensure min range of 5, 10% padding
    
    return line,

# Create animation
ani = FuncAnimation(fig, update, frames=None, interval=update_interval, blit=True)

# Show the plot
plt.show()