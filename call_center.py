import argparse

import matplotlib.pyplot as plt
import numpy as np

OFF_SEASON_RATE = 1 / 1000  # One call per 1000s
HALF_YEAR_PEAK_RATE = 10 * OFF_SEASON_RATE  # Tenfold increase
END_YEAR_PEAK_RATE = 20 * OFF_SEASON_RATE  # Twentyfold increase
MEAN_SERVICE_TIME = 100  # Average call duration 100s


class Customer:
    """Customer in the call center system."""

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
    # Initialize data collection lists
    time_points = []
    queue_lengths = []
    system_customers = []
    waiting_times = []
    system_times = []
    server_loads = []

    # Initialize simulation state
    current_time = 0
    queue = []
    servers = [None] * num_servers
    next_arrival = np.random.exponential(1 / arrival_rate)
    next_departures = [float('inf')] * num_servers

    # Statistics tracking
    customer_id = 0
    completed_customers = []

    # Stats for time-weighted averages
    last_update_time = 0
    cum_queue_length = 0
    cum_system_customers = 0
    cum_server_busy = 0

    # Start the simulation
    while current_time < max_time:  # Run for a specified maximum time
        # Determine next event (arrival or departure)
        next_event_time = min(next_arrival, min(next_departures))

        # Update time-weighted statistics
        time_diff = next_event_time - current_time
        current_queue_length = len(queue)
        current_system_customers = len(queue) + sum(1 for s in servers if s is not None)
        current_server_busy = sum(1 for s in servers if s is not None) / num_servers

        cum_queue_length += current_queue_length * time_diff
        cum_system_customers += current_system_customers * time_diff
        cum_server_busy += current_server_busy * time_diff

        # Update current time
        current_time = next_event_time

        # Process the event
        if current_time == next_arrival:
            # Customer arrival
            customer = Customer(customer_id, current_time)
            customer_id += 1

            # Check if a server is available
            free_server = None
            for i, server in enumerate(servers):
                if server is None:
                    free_server = i
                    break

            if free_server is not None:
                # Start service immediately
                customer.service_start_time = current_time
                servers[free_server] = customer

                # Schedule departure
                service_time = np.random.exponential(MEAN_SERVICE_TIME)
                next_departures[free_server] = current_time + service_time
            else:
                # Add to queue
                queue.append(customer)

            # Schedule next arrival
            next_arrival = current_time + np.random.exponential(1 / arrival_rate)

        else:
            # Customer departure from a server
            server_idx = next_departures.index(current_time)

            # Complete service
            customer = servers[server_idx]
            customer.departure_time = current_time
            completed_customers.append(customer)

            # Free the server
            servers[server_idx] = None
            next_departures[server_idx] = float('inf')

            # Check if anyone is waiting
            if queue:
                # Get next customer from queue
                next_customer = queue.pop(0)
                next_customer.service_start_time = current_time

                # Start service
                servers[server_idx] = next_customer

                # Schedule departure
                service_time = np.random.exponential(MEAN_SERVICE_TIME)
                next_departures[server_idx] = current_time + service_time

        # Record statistics every 100 seconds of simulation time
        if current_time - last_update_time >= 100:
            # Calculate time-weighted averages
            mean_queue_length = cum_queue_length / current_time if current_time > 0 else 0
            mean_system_customers = cum_system_customers / current_time if current_time > 0 else 0
            mean_server_load = cum_server_busy / current_time if current_time > 0 else 0

            # Calculate customer averages
            if completed_customers:
                mean_waiting_time = sum(c.waiting_time for c in completed_customers) / len(completed_customers)
                mean_system_time = sum(c.system_time for c in completed_customers) / len(completed_customers)
            else:
                mean_waiting_time = 0
                mean_system_time = 0

            # Record the statistics
            time_points.append(current_time)
            queue_lengths.append(mean_queue_length)
            system_customers.append(mean_system_customers)
            waiting_times.append(mean_waiting_time)
            system_times.append(mean_system_time)
            server_loads.append(mean_server_load)

            last_update_time = current_time

    # Calculate final Little's Law verification
    littles_data = None
    if completed_customers:
        L = system_customers[-1]
        W = system_times[-1]
        arrival_rate_measured = len(completed_customers) / current_time
        lambda_W = arrival_rate_measured * W
        match = abs(L - lambda_W) / max(L, lambda_W, 1e-10) < 0.1

        littles_data = {
            "L": L,
            "lambda_W": lambda_W,
            "match": match
        }

    # Return all collected statistics
    return {
        "time_points": time_points,
        "queue_lengths": queue_lengths,
        "system_customers": system_customers,
        "waiting_times": waiting_times,
        "system_times": system_times,
        "server_loads": server_loads,
        "littles_data": littles_data,
        "completed_customers": len(completed_customers)
    }


def plot_simulation_results(results):
    """Plot the simulation results."""
    # Create the figure and subplots
    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Call Center Simulation Results', fontsize=16)

    # Improve spacing
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.05, hspace=0.35, wspace=0.25)

    # Extract data
    time_points = results["time_points"]
    queue_lengths = results["queue_lengths"]
    system_customers = results["system_customers"]
    waiting_times = results["waiting_times"]
    system_times = results["system_times"]
    server_loads = results["server_loads"]

    # Plot queue length
    axs[0, 0].plot(time_points, queue_lengths, 'b-', linewidth=2)
    axs[0, 0].set_title('Mean Queue Length')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Customers')
    axs[0, 0].grid(True)
    axs[0, 0].text(0.02, 0.95, 'Shows the average number of\ncustomers waiting in line',
                   transform=axs[0, 0].transAxes, fontsize=9, verticalalignment='top')

    # Plot system customers
    axs[0, 1].plot(time_points, system_customers, 'g-', linewidth=2)
    axs[0, 1].set_title('Mean Number of Customers in System')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Customers')
    axs[0, 1].grid(True)
    axs[0, 1].text(0.02, 0.95, 'Shows the average number of customers\nin the system (waiting + being served)',
                   transform=axs[0, 1].transAxes, fontsize=9, verticalalignment='top')

    # Plot waiting times
    axs[1, 0].plot(time_points, waiting_times, 'r-', linewidth=2)
    axs[1, 0].set_title('Mean Queue Time')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Time (s)')
    axs[1, 0].grid(True)
    axs[1, 0].text(0.02, 0.95, 'Shows the average time customers\nspend waiting before service',
                   transform=axs[1, 0].transAxes, fontsize=9, verticalalignment='top')

    # Plot system times
    axs[1, 1].plot(time_points, system_times, 'm-', linewidth=2)
    axs[1, 1].set_title('Mean Time in System')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Time (s)')
    axs[1, 1].grid(True)
    axs[1, 1].text(0.02, 0.95, 'Shows the average total time customers\nspend in the system (waiting + service)',
                   transform=axs[1, 1].transAxes, fontsize=9, verticalalignment='top')

    # Plot server loads
    axs[2, 0].plot(time_points, server_loads, 'c-', linewidth=2)
    axs[2, 0].set_title('Mean Server Load')
    axs[2, 0].set_xlabel('Time (s)')
    axs[2, 0].set_ylabel('Utilization (0-1)')
    axs[2, 0].grid(True)
    axs[2, 0].text(0.02, 0.95, 'Shows the percentage of time\nservers are busy',
                   transform=axs[2, 0].transAxes, fontsize=9, verticalalignment='top')

    # Little's Law verification
    axs[2, 1].axis('off')
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
        axs[2, 1].text(0.5, 0.5, littles_text, ha='center', va='center', fontsize=12)
    else:
        axs[2, 1].text(0.5, 0.5, "Little's Law:\nInsufficient data for verification",
                       ha='center', va='center')

    # Add a text with simulation summary
    plt.figtext(0.5, 0.01,
                f"Simulation Summary: {results['completed_customers']} completed customers, Final server load: {server_loads[-1]:.2f}",
                ha='center', fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main():
    """Main function to run the simulation and plot results."""
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
