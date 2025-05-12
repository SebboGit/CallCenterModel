import argparse

import matplotlib.pyplot as plt
import numpy as np

OFF_SEASON_RATE = 1 / 1000  # One call per 1000s
HALF_YEAR_PEAK_RATE = 10
END_YEAR_PEAK_RATE = 20
MEAN_SERVICE_TIME = 100  # Average call duration 100s


class Customer:

    def __init__(self, id: int, arrival_time: float):
        self.id = id
        self.arrival_time = arrival_time
        self.service_start_time = None
        self.departure_time = None

    @property
    def waiting_time(self) -> float:
        """Time customer spent waiting in queue."""
        if self.service_start_time is None:
            return 0
        return self.service_start_time - self.arrival_time

    @property
    def service_time(self) -> float:
        """Time customer spent being served."""
        if self.service_start_time is None or self.departure_time is None:
            return 0
        return self.departure_time - self.service_start_time

    @property
    def system_time(self) -> float:
        """Total time customer spent in the system."""
        if self.departure_time is None:
            return 0
        return self.departure_time - self.arrival_time


def run_simulation(arrival_rate=0.01, num_servers=1, max_time=10000):
    """Run the call center simulation and return collected statistics."""
    # Initialize simulation environment
    sim_env = _initialize_simulation_environment(num_servers)
    sim_env['arrival_rate'] = arrival_rate
    sim_env['max_time'] = max_time

    # Initialize data collection
    stats = _initialize_statistics()

    # Run the simulation loop
    _run_simulation_loop(sim_env, stats)

    return _prepare_final_results(sim_env, stats)


def _initialize_simulation_environment(num_servers):
    """Initialize the simulation environment with required parameters."""
    return {
        'current_time': 0,
        'queue': [],
        'servers': [None] * num_servers,
        'num_servers': num_servers,
        'next_arrival': 0,
        'next_departures': [float('inf')] * num_servers,
        'customer_id': 0,
        'completed_customers': []
    }


def _initialize_statistics():
    """Initialize data structures for collecting simulation statistics."""
    return {
        'time_points': [],
        'queue_lengths': [],
        'system_customers': [],
        'waiting_times': [],
        'system_times': [],
        'server_loads': [],
        'last_update_time': 0,
        'cum_queue_length': 0,
        'cum_system_customers': 0,
        'cum_server_busy': 0
    }


def _run_simulation_loop(sim_env, stats):
    """Main simulation loop handling events until max time is reached."""
    # Schedule first arrival
    sim_env['next_arrival'] = np.random.exponential(1 / sim_env['arrival_rate'])

    while sim_env['current_time'] < sim_env['max_time']:
        _process_next_event(sim_env, stats)

        _record_statistics_if_needed(sim_env, stats)


def _process_next_event(sim_env, stats):
    """Process the next event (arrival or departure) in the simulation."""
    # Determine next event time
    next_event_time = min(sim_env['next_arrival'], min(sim_env['next_departures']))

    _update_cumulative_stats(sim_env, stats, next_event_time)

    sim_env['current_time'] = next_event_time

    # Handle event based on type
    if sim_env['current_time'] == sim_env['next_arrival']:
        _handle_customer_arrival(sim_env)
    else:
        _handle_customer_departure(sim_env)


def _update_cumulative_stats(sim_env, stats, next_event_time):
    """Update time-weighted statistics for the current time period."""
    time_diff = next_event_time - sim_env['current_time']

    # Current state metrics
    current_queue_length = len(sim_env['queue'])
    current_system_customers = current_queue_length + sum(1 for s in sim_env['servers'] if s is not None)
    current_server_busy = sum(1 for s in sim_env['servers'] if s is not None) / sim_env['num_servers']

    # Update cumulative stats
    stats['cum_queue_length'] += current_queue_length * time_diff
    stats['cum_system_customers'] += current_system_customers * time_diff
    stats['cum_server_busy'] += current_server_busy * time_diff


def _handle_customer_arrival(sim_env):
    """Process a customer arrival event."""
    # Create new customer
    customer = Customer(sim_env['customer_id'], sim_env['current_time'])
    sim_env['customer_id'] += 1

    # Find available server if any
    free_server = _find_free_server(sim_env)

    if free_server is not None:
        # Start service immediately
        _assign_customer_to_server(sim_env, customer, free_server)
    else:
        # Add to queue
        sim_env['queue'].append(customer)

    # Schedule next arrival
    sim_env['next_arrival'] = sim_env['current_time'] + np.random.exponential(1 / sim_env['arrival_rate'])


def _find_free_server(sim_env):
    """Find and return index of first available server, or None if all busy."""
    for i, server in enumerate(sim_env['servers']):
        if server is None:
            return i
    return None


def _assign_customer_to_server(sim_env, customer, server_idx):
    """Assign a customer to a server and schedule their departure."""
    customer.service_start_time = sim_env['current_time']
    sim_env['servers'][server_idx] = customer

    # Schedule departure
    service_time = np.random.exponential(MEAN_SERVICE_TIME)
    sim_env['next_departures'][server_idx] = sim_env['current_time'] + service_time


def _handle_customer_departure(sim_env):
    """Process a customer departure event."""
    # Find server with completed service
    server_idx = sim_env['next_departures'].index(sim_env['current_time'])

    # Complete customer service
    customer = sim_env['servers'][server_idx]
    customer.departure_time = sim_env['current_time']
    sim_env['completed_customers'].append(customer)

    # Free the server
    sim_env['servers'][server_idx] = None
    sim_env['next_departures'][server_idx] = float('inf')

    # Check if queue has waiting customers
    if sim_env['queue']:
        next_customer = sim_env['queue'].pop(0)
        _assign_customer_to_server(sim_env, next_customer, server_idx)


def _record_statistics_if_needed(sim_env, stats):
    """Record statistics periodically (every 100 seconds of simulation time)."""
    if sim_env['current_time'] - stats['last_update_time'] >= 100:
        _calculate_and_record_current_stats(sim_env, stats)
        stats['last_update_time'] = sim_env['current_time']


def _calculate_and_record_current_stats(sim_env, stats):
    """Calculate current statistics and add them to the stats collection."""
    current_time = sim_env['current_time']
    completed_customers = sim_env['completed_customers']

    # Calculate time-weighted averages
    mean_queue_length = stats['cum_queue_length'] / current_time if current_time > 0 else 0
    mean_system_customers = stats['cum_system_customers'] / current_time if current_time > 0 else 0
    mean_server_load = stats['cum_server_busy'] / current_time if current_time > 0 else 0

    # Calculate customer averages
    if completed_customers:
        mean_waiting_time = sum(c.waiting_time for c in completed_customers) / len(completed_customers)
        mean_system_time = sum(c.system_time for c in completed_customers) / len(completed_customers)
    else:
        mean_waiting_time = 0
        mean_system_time = 0

    # Record the statistics
    stats['time_points'].append(current_time)
    stats['queue_lengths'].append(mean_queue_length)
    stats['system_customers'].append(mean_system_customers)
    stats['waiting_times'].append(mean_waiting_time)
    stats['system_times'].append(mean_system_time)
    stats['server_loads'].append(mean_server_load)


def _prepare_final_results(sim_env, stats):
    littles_data = _verify_littles_law(sim_env, stats)

    # Return all collected statistics
    return {
        "time_points": stats['time_points'],
        "queue_lengths": stats['queue_lengths'],
        "system_customers": stats['system_customers'],
        "waiting_times": stats['waiting_times'],
        "system_times": stats['system_times'],
        "server_loads": stats['server_loads'],
        "littles_data": littles_data,
        "completed_customers": len(sim_env['completed_customers'])
    }


def _verify_littles_law(sim_env, stats):
    """Verify Little's Law (L = λW) using simulation results."""
    completed_customers = sim_env['completed_customers']

    if not completed_customers or not stats['system_customers']:
        return None

    # Extract required metrics
    L = stats['system_customers'][-1]
    W = stats['system_times'][-1]
    arrival_rate_measured = len(completed_customers) / sim_env['current_time']
    lambda_W = arrival_rate_measured * W

    # Check if L and λW are close enough (within 10%)
    match = abs(L - lambda_W) / max(L, lambda_W, 1e-10) < 0.1

    return {
        "L": L,
        "lambda_W": lambda_W,
        "match": match
    }


def plot_simulation_results(results):
    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Call Center Simulation Results', fontsize=16)

    _configure_plot_layout(fig)

    _plot_queue_metrics(axs, results)
    _plot_time_metrics(axs, results)
    _plot_server_and_littles_law(axs, results)

    _add_simulation_summary(fig, results)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def _configure_plot_layout(fig):
    plt.subplots_adjust(
        left=0.1,
        right=0.9,
        top=0.9,
        bottom=0.05,
        hspace=0.35,
        wspace=0.25
    )


def _plot_queue_metrics(axs, results):
    """Plot queue-related metrics (queue length and system customers)."""
    time_points = results["time_points"]

    # Plot queue length
    _create_subplot(
        ax=axs[0, 0],
        x_data=time_points,
        y_data=results["queue_lengths"],
        color='b',
        title='Mean Queue Length',
        xlabel='Time (s)',
        ylabel='Customers',
        annotation='Shows the average number of\ncustomers waiting in line'
    )

    # Plot system customers
    _create_subplot(
        ax=axs[0, 1],
        x_data=time_points,
        y_data=results["system_customers"],
        color='g',
        title='Mean Number of Customers in System',
        xlabel='Time (s)',
        ylabel='Customers',
        annotation='Shows the average number of customers\nin the system (waiting + being served)'
    )


def _plot_time_metrics(axs, results):
    """Plot time-related metrics (waiting times and system times)."""
    time_points = results["time_points"]

    # Plot waiting times
    _create_subplot(
        ax=axs[1, 0],
        x_data=time_points,
        y_data=results["waiting_times"],
        color='r',
        title='Mean Queue Time',
        xlabel='Time (s)',
        ylabel='Time (s)',
        annotation='Shows the average time customers\nspend waiting before service'
    )

    # Plot system times
    _create_subplot(
        ax=axs[1, 1],
        x_data=time_points,
        y_data=results["system_times"],
        color='m',
        title='Mean Time in System',
        xlabel='Time (s)',
        ylabel='Time (s)',
        annotation='Shows the average total time customers\nspend in the system (waiting + service)'
    )


def _plot_server_and_littles_law(axs, results):
    """Plot server loads and Little's Law verification."""
    time_points = results["time_points"]

    # Plot server loads
    _create_subplot(
        ax=axs[2, 0],
        x_data=time_points,
        y_data=results["server_loads"],
        color='c',
        title='Mean Server Load',
        xlabel='Time (s)',
        ylabel='Utilization (0-1)',
        annotation='Shows the percentage of time\nservers are busy'
    )

    # Little's Law verification
    _plot_littles_law_verification(axs[2, 1], results)


def _create_subplot(ax, x_data, y_data, color, title, xlabel, ylabel, annotation):
    ax.plot(x_data, y_data, f'{color}-', linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.text(
        0.02, 0.95,
        annotation,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top'
    )


def _plot_littles_law_verification(ax, results):
    """Plot Little's Law verification information."""
    ax.axis('off')
    littles_data = results["littles_data"]

    if littles_data:
        match_text = "✓" if littles_data["match"] else "✗"
        littles_text = (
            f"\nLittle's Law Verification:\n\n"
            f"L (mean customers) = {littles_data['L']:.2f}\n"
            f"λW (arrival × time) = {littles_data['lambda_W']:.2f}\n\n"
            f"Match: {match_text}\n\n"
            f"Simulation Parameters:\n\n"
            f"Arrival rate: {results.get('rate', 'N/A')}x base rate\n"
            f"Simulation Time: {results.get('time', 'N/A')} seconds\n"
            f"Number of Servers: {results.get('servers', 'N/A')}"
        )
        ax.text(0.5, 0.5, littles_text, ha='center', va='center', fontsize=12)
    else:
        ax.text(
            0.5, 0.5,
            "Little's Law:\nInsufficient data for verification",
            ha='center', va='center'
        )


def _add_simulation_summary(fig, results):
    plt.figtext(
        0.5, 0.01,
        f"Simulation Summary: {results['completed_customers']} completed customers, "
        f"Final server load: {results['server_loads'][-1]:.2f}",
        ha='center',
        fontsize=10
    )


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Call Center Simulation for Modeling and Simulation Course')
    parser.add_argument('--rate', type=float, default=1,
                        help='Arrival rate multiplier (default: 1x base rate)')
    parser.add_argument('--servers', type=int, default=1,
                        help='Number of servers')
    parser.add_argument('--time', type=int, default=10000,
                        help='Simulation duration in seconds (default: 10000)')
    args = parser.parse_args()

    # Calculate arrival rate
    arrival_rate = OFF_SEASON_RATE * args.rate

    print(f"Running call center simulation with:")
    print(f"  - Arrival rate: {args.rate}x base rate ({arrival_rate:.6f} calls/second)")
    print(f"  - Number of servers: {args.servers}")
    print(f"  - Simulation time: {args.time} seconds")
    print("\nSimulation running, please wait...")

    # Run the simulation
    results = run_simulation(arrival_rate, args.servers, args.time)
    results.update({"rate": args.rate, "time": args.time, "servers": args.servers})

    print(f"Simulation completed with {results['completed_customers']} customers processed.")
    print(f"Final server load: {results['server_loads'][-1]:.2f}")
    print(f"Mean queue length: {results['queue_lengths'][-1]:.2f}")
    print(f"Mean waiting time: {results['waiting_times'][-1]:.2f} seconds")
    print("\nGenerating plots...")

    # Plot the results
    plot_simulation_results(results)


if __name__ == "__main__":
    main()