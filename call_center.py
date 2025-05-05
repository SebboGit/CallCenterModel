import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import heapq
import time
from enum import Enum
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any, Deque

# Constants
OFF_SEASON_RATE = 1 / 1000  # One call per 1000s
HALF_YEAR_PEAK_RATE = 10 * OFF_SEASON_RATE  # Tenfold increase
END_YEAR_PEAK_RATE = 20 * OFF_SEASON_RATE  # Twentyfold increase
MEAN_SERVICE_TIME = 100  # Average call duration 100s

# Month definitions
HALF_YEAR_PEAK_MONTHS = [6, 7, 8]  # June, July, August
END_YEAR_PEAK_START = (3, 15)  # March 15
END_YEAR_PEAK_END = (3, 31)  # March 31


class EventType(Enum):
    """Event types in the call center simulation."""
    ARRIVAL = 1  # Customer arrival
    DEPARTURE = 2  # Customer departure after service
    STATS_COLLECTION = 3  # For collecting statistics


@dataclass(order=True)
class Event:
    """Event in the simulation's event queue."""
    time: float
    type: EventType = field(compare=False)
    customer_id: Optional[int] = field(default=None, compare=False)


class Customer:
    """Customer in the call center system."""

    def __init__(self, id: int, arrival_time: float):
        self.id = id
        self.arrival_time = arrival_time
        self.service_start_time: Optional[float] = None
        self.departure_time: Optional[float] = None

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


class CallCenterStats:
    """Statistics collector for the call center simulation."""

    def __init__(self, window_size: int = 100):
        # Current state
        self.queue_length = 0
        self.system_customers = 0
        self.server_busy = 0  # 0 or 1 for single server

        # Time tracking
        self.last_update_time = 0

        # Cumulative stats (area under curve)
        self.cum_queue_length = 0
        self.cum_system_customers = 0
        self.cum_server_busy = 0

        # Completed customer stats
        self.completed_customers: List[Customer] = []

        # Time series for plotting
        self.times: Deque[float] = deque(maxlen=window_size)
        self.queue_lengths: Deque[int] = deque(maxlen=window_size)
        self.system_customers_series: Deque[int] = deque(maxlen=window_size)
        self.server_loads: Deque[float] = deque(maxlen=window_size)
        self.mean_waiting_times: Deque[float] = deque(maxlen=window_size)
        self.mean_system_times: Deque[float] = deque(maxlen=window_size)

    def update(self, current_time: float) -> None:
        """Update cumulative statistics based on time passed."""
        time_diff = current_time - self.last_update_time
        self.cum_queue_length += self.queue_length * time_diff
        self.cum_system_customers += self.system_customers * time_diff
        self.cum_server_busy += self.server_busy * time_diff
        self.last_update_time = current_time

    def record_point(self, current_time: float) -> None:
        """Record a data point for plotting."""
        self.update(current_time)

        # Calculate means for plotting
        mean_queue_length = self.mean_queue_length(current_time)
        mean_system_customers = self.mean_system_customers(current_time)
        mean_server_load = self.mean_server_load(current_time)

        # Calculate customer time metrics if we have completed customers
        mean_waiting_time = self.mean_waiting_time()
        mean_system_time = self.mean_system_time()

        # Record data points
        self.times.append(current_time)
        self.queue_lengths.append(mean_queue_length)
        self.system_customers_series.append(mean_system_customers)
        self.server_loads.append(mean_server_load)
        self.mean_waiting_times.append(mean_waiting_time)
        self.mean_system_times.append(mean_system_time)

    def mean_queue_length(self, current_time: float) -> float:
        """Calculate mean queue length up to current time."""
        if current_time == 0:
            return 0
        return self.cum_queue_length / current_time

    def mean_system_customers(self, current_time: float) -> float:
        """Calculate mean number of customers in the system up to current time."""
        if current_time == 0:
            return 0
        return self.cum_system_customers / current_time

    def mean_server_load(self, current_time: float) -> float:
        """Calculate mean server load (utilization) up to current time."""
        if current_time == 0:
            return 0
        return self.cum_server_busy / current_time

    def mean_waiting_time(self) -> float:
        """Calculate mean waiting time for completed customers."""
        if not self.completed_customers:
            return 0
        return sum(c.waiting_time for c in self.completed_customers) / len(self.completed_customers)

    def mean_system_time(self) -> float:
        """Calculate mean time spent in the system for completed customers."""
        if not self.completed_customers:
            return 0
        return sum(c.system_time for c in self.completed_customers) / len(self.completed_customers)

    def verify_littles_law(self) -> Dict[str, float]:
        """Verify Little's Law: L = λW."""
        if not self.completed_customers or not self.times:
            return {"L": 0, "lambda*W": 0, "match": False}

        # Current simulation time
        current_time = self.times[-1]

        # Calculate arrival rate (lambda)
        arrival_rate = len(self.completed_customers) / current_time

        # L = mean number of customers in system
        L = self.mean_system_customers(current_time)

        # W = mean time in system
        W = self.mean_system_time()

        # According to Little's Law: L = lambda * W
        lambda_W = arrival_rate * W

        # Check if Little's Law holds (with some tolerance for numerical issues)
        match = abs(L - lambda_W) / max(L, lambda_W, 1e-10) < 0.1

        return {
            "L": L,
            "lambda*W": lambda_W,
            "match": match
        }


class Simulator:
    """Base class for call center simulators."""

    def __init__(
            self,
            arrival_rate_fn: Callable[[float], float],
            mean_service_time: float,
            service_time_distribution: str = "exponential",
            num_servers: int = 1
    ):
        self.arrival_rate_fn = arrival_rate_fn
        self.mean_service_time = mean_service_time
        self.service_time_distribution = service_time_distribution
        self.num_servers = num_servers

        # State variables
        self.time = 0
        self.customer_id_counter = 0
        self.queue: List[Customer] = []
        self.servers: List[Optional[Customer]] = [None] * num_servers
        self.stats = CallCenterStats()

        # For visualization
        self.paused = False
        self.speed_factor = 1.0

    def generate_arrival_time(self, current_time: float) -> float:
        """Generate time until next arrival using current arrival rate."""
        rate = self.arrival_rate_fn(current_time)
        # Avoid division by zero
        if rate <= 0:
            return float('inf')
        return np.random.exponential(1 / rate)

    def generate_service_time(self) -> float:
        """Generate service time based on distribution."""
        if self.service_time_distribution == "exponential":
            return np.random.exponential(self.mean_service_time)
        elif self.service_time_distribution == "uniform":
            # Uniform distribution with the same mean
            return np.random.uniform(0, 2 * self.mean_service_time)
        elif self.service_time_distribution == "normal":
            # Normal distribution with the same mean and standard deviation
            return max(0, np.random.normal(self.mean_service_time, self.mean_service_time / 2))
        else:
            # Default to exponential
            return np.random.exponential(self.mean_service_time)

    def find_free_server(self) -> Optional[int]:
        """Find a free server and return its index, or None if all are busy."""
        for i, server in enumerate(self.servers):
            if server is None:
                return i
        return None

    def update_stats(self) -> None:
        """Update statistics based on current state."""
        self.stats.queue_length = len(self.queue)
        self.stats.system_customers = len(self.queue) + sum(1 for s in self.servers if s is not None)
        self.stats.server_busy = sum(1 for s in self.servers if s is not None) / self.num_servers
        self.stats.update(self.time)

    def run(self, duration: float, stats_interval: float = 10.0) -> CallCenterStats:
        """Run the simulation for a specified duration."""
        pass


class FixedTimeSimulator(Simulator):
    """Fixed time step simulator for call center."""

    def __init__(
            self,
            arrival_rate_fn: Callable[[float], float],
            mean_service_time: float,
            service_time_distribution: str = "exponential",
            num_servers: int = 1,
            time_step: float = 1.0
    ):
        super().__init__(arrival_rate_fn, mean_service_time, service_time_distribution, num_servers)
        self.time_step = time_step
        self.next_arrival_time = self.time + self.generate_arrival_time(self.time)
        self.server_completion_times = [float('inf')] * num_servers

    def step(self) -> None:
        """Advance the simulation by one time step."""
        old_time = self.time
        self.time += self.time_step

        # Process arrivals that occurred during this time step
        while self.next_arrival_time <= self.time:
            # Create a new customer
            customer = Customer(self.customer_id_counter, self.next_arrival_time)
            self.customer_id_counter += 1

            # Check if a server is available
            server_idx = self.find_free_server()
            if server_idx is not None:
                # Start service immediately
                customer.service_start_time = self.next_arrival_time
                self.servers[server_idx] = customer

                # Schedule service completion
                service_time = self.generate_service_time()
                self.server_completion_times[server_idx] = self.next_arrival_time + service_time
            else:
                # Add to queue
                self.queue.append(customer)

            # Schedule next arrival
            self.next_arrival_time = self.next_arrival_time + self.generate_arrival_time(self.next_arrival_time)

        # Process service completions
        for server_idx in range(self.num_servers):
            if self.servers[server_idx] is not None and self.server_completion_times[server_idx] <= self.time:
                # Complete service for this customer
                customer = self.servers[server_idx]
                customer.departure_time = self.server_completion_times[server_idx]
                self.stats.completed_customers.append(customer)

                # Free the server
                self.servers[server_idx] = None

                # Check if there are customers waiting in the queue
                if self.queue:
                    # Get the next customer from the queue
                    next_customer = self.queue.pop(0)
                    next_customer.service_start_time = self.server_completion_times[server_idx]

                    # Start service
                    self.servers[server_idx] = next_customer

                    # Schedule service completion
                    service_time = self.generate_service_time()
                    self.server_completion_times[server_idx] = next_customer.service_start_time + service_time
                else:
                    # No customers waiting
                    self.server_completion_times[server_idx] = float('inf')

        # Update statistics
        self.update_stats()

    def run(self, duration: float, stats_interval: float = 10.0) -> CallCenterStats:
        """Run the simulation for a specified duration with fixed time steps."""
        next_stats_time = 0

        while self.time < duration:
            if not self.paused:
                self.step()

                # Record statistics at regular intervals
                if self.time >= next_stats_time:
                    self.stats.record_point(self.time)
                    next_stats_time += stats_interval

            # Slow down or speed up the simulation based on speed factor
            real_time_step = self.time_step / self.speed_factor
            if real_time_step > 0:
                time.sleep(real_time_step)

        return self.stats


class EventBasedSimulator(Simulator):
    """Event-based simulator for call center."""

    def __init__(
            self,
            arrival_rate_fn: Callable[[float], float],
            mean_service_time: float,
            service_time_distribution: str = "exponential",
            num_servers: int = 1
    ):
        super().__init__(arrival_rate_fn, mean_service_time, service_time_distribution, num_servers)

        # Event queue (priority queue based on event time)
        self.event_queue: List[Event] = []

        # Schedule first arrival and statistics collection
        self.schedule_arrival()
        heapq.heappush(
            self.event_queue,
            Event(time=0, type=EventType.STATS_COLLECTION)
        )

    def schedule_arrival(self) -> None:
        """Schedule the next customer arrival."""
        arrival_time = self.time + self.generate_arrival_time(self.time)
        heapq.heappush(
            self.event_queue,
            Event(time=arrival_time, type=EventType.ARRIVAL)
        )

    def schedule_departure(self, server_idx: int, customer: Customer) -> None:
        """Schedule a customer departure from a server."""
        service_time = self.generate_service_time()
        departure_time = self.time + service_time
        heapq.heappush(
            self.event_queue,
            Event(time=departure_time, type=EventType.DEPARTURE, customer_id=customer.id)
        )

    def handle_arrival(self) -> None:
        """Process a customer arrival event."""
        # Create a new customer
        customer = Customer(self.customer_id_counter, self.time)
        self.customer_id_counter += 1

        # Check if a server is available
        server_idx = self.find_free_server()
        if server_idx is not None:
            # Start service immediately
            customer.service_start_time = self.time
            self.servers[server_idx] = customer

            # Schedule departure
            self.schedule_departure(server_idx, customer)
        else:
            # Add to queue
            self.queue.append(customer)

        # Schedule next arrival
        self.schedule_arrival()

    def handle_departure(self, event: Event) -> None:
        """Process a customer departure event."""
        # Find the server with this customer
        server_idx = None
        for i, server_customer in enumerate(self.servers):
            if (server_customer is not None and
                    server_customer.id == event.customer_id):
                server_idx = i
                break

        if server_idx is None:
            # This should not happen in a correctly implemented simulation
            print(f"Warning: Customer {event.customer_id} not found in servers at time {self.time}")
            return

        # Complete service for this customer
        customer = self.servers[server_idx]
        customer.departure_time = self.time
        self.stats.completed_customers.append(customer)

        # Free the server
        self.servers[server_idx] = None

        # Check if there are customers waiting in the queue
        if self.queue:
            # Get the next customer from the queue
            next_customer = self.queue.pop(0)
            next_customer.service_start_time = self.time

            # Start service
            self.servers[server_idx] = next_customer

            # Schedule departure
            self.schedule_departure(server_idx, next_customer)

    def handle_stats_collection(self) -> None:
        """Handle statistics collection event."""
        self.stats.record_point(self.time)

        # Schedule next stats collection
        heapq.heappush(
            self.event_queue,
            Event(time=self.time + 10.0, type=EventType.STATS_COLLECTION)
        )

    def process_next_event(self) -> bool:
        """Process the next event in the queue. Returns False if no more events."""
        if not self.event_queue:
            return False

        # Get the next event
        event = heapq.heappop(self.event_queue)

        # Update simulation time
        old_time = self.time
        self.time = event.time

        # Update statistics for the time period
        self.update_stats()

        # Handle event based on type
        if event.type == EventType.ARRIVAL:
            self.handle_arrival()
        elif event.type == EventType.DEPARTURE:
            self.handle_departure(event)
        elif event.type == EventType.STATS_COLLECTION:
            self.handle_stats_collection()

        return True

    def run(self, duration: float, stats_interval: float = 10.0) -> CallCenterStats:
        """Run the event-based simulation for a specified duration."""
        while self.event_queue and self.time < duration:
            if not self.paused:
                # Get the next event
                event = self.event_queue[0]  # Peek without removing

                # Process all events up to duration
                if event.time <= duration:
                    self.process_next_event()
                else:
                    # No more events within the duration
                    break

                # Calculate real time to sleep for visualization
                if len(self.event_queue) > 0:
                    time_to_next_event = self.event_queue[0].time - self.time
                    real_time_sleep = time_to_next_event / self.speed_factor
                    if real_time_sleep > 0:
                        time.sleep(min(real_time_sleep, 0.1))  # Cap at 100ms for responsiveness
            else:
                # When paused, sleep a bit to avoid busy waiting
                time.sleep(0.1)

        return self.stats


def constant_arrival_rate(rate: float) -> Callable[[float], float]:
    """Return a function that gives a constant arrival rate."""
    return lambda t: rate


def seasonal_arrival_rate() -> Callable[[float], float]:
    """
    Return a function that models the seasonal arrival rate as described.
    - Regular rate: 1/1000 calls per second
    - Half-year peak (Jun-Aug): 10x increase
    - End of fiscal year peak (Mar 15-31): 20x increase
    """

    def rate_function(t: float) -> float:
        # Convert simulation time (in seconds) to month and day
        # Assuming simulation starts at beginning of year
        SECONDS_PER_DAY = 24 * 60 * 60
        DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        # Calculate day of year
        days_elapsed = t / SECONDS_PER_DAY

        # Determine month and day
        month = 1
        day = days_elapsed + 1  # Start from day 1

        for m, days in enumerate(DAYS_PER_MONTH, 1):
            if day > days:
                day -= days
                month += 1
            else:
                break

        # Apply seasonal rules
        if month in HALF_YEAR_PEAK_MONTHS:
            # Half-year peak (June-August)
            return HALF_YEAR_PEAK_RATE
        elif month == END_YEAR_PEAK_START[0] and day >= END_YEAR_PEAK_START[1] and day <= END_YEAR_PEAK_END[1]:
            # End of fiscal year peak (March 15-31)
            return END_YEAR_PEAK_RATE
        else:
            # Regular rate
            return OFF_SEASON_RATE

    return rate_function


class SimulationVisualizer:
    """Real-time visualizer for the call center simulation."""

    def __init__(self, simulator):
        self.simulator = simulator
        self.fig, self.axs = plt.subplots(3, 2, figsize=(14, 10))
        self.fig.suptitle('Call Center Simulation', fontsize=16)

        # Initialize subplots
        self.queue_line, = self.axs[0, 0].plot([], [], 'b-', label='Queue Length')
        self.axs[0, 0].set_title('Mean Queue Length')
        self.axs[0, 0].set_xlabel('Time (s)')
        self.axs[0, 0].set_ylabel('Customers')
        self.axs[0, 0].grid(True)

        self.system_line, = self.axs[0, 1].plot([], [], 'g-', label='System Customers')
        self.axs[0, 1].set_title('Mean Number of Customers in System')
        self.axs[0, 1].set_xlabel('Time (s)')
        self.axs[0, 1].set_ylabel('Customers')
        self.axs[0, 1].grid(True)

        self.wait_line, = self.axs[1, 0].plot([], [], 'r-', label='Wait Time')
        self.axs[1, 0].set_title('Mean Queue Time')
        self.axs[1, 0].set_xlabel('Time (s)')
        self.axs[1, 0].set_ylabel('Time (s)')
        self.axs[1, 0].grid(True)

        self.system_time_line, = self.axs[1, 1].plot([], [], 'm-', label='System Time')
        self.axs[1, 1].set_title('Mean Time in System')
        self.axs[1, 1].set_xlabel('Time (s)')
        self.axs[1, 1].set_ylabel('Time (s)')
        self.axs[1, 1].grid(True)

        self.server_line, = self.axs[2, 0].plot([], [], 'c-', label='Server Load')
        self.axs[2, 0].set_title('Mean Server Load')
        self.axs[2, 0].set_xlabel('Time (s)')
        self.axs[2, 0].set_ylabel('Utilization (0-1)')
        self.axs[2, 0].grid(True)

        # Little's Law verification
        self.littles_ax = self.axs[2, 1]
        self.littles_ax.axis('off')
        self.littles_text = self.littles_ax.text(0.5, 0.5, "", fontsize=12,
                                                 ha='center', va='center',
                                                 transform=self.littles_ax.transAxes)
        self.littles_ax.set_title("Little's Law Verification")

        # Controls
        self.pause_button_ax = plt.axes([0.45, 0.01, 0.1, 0.04])
        self.pause_button = plt.Button(self.pause_button_ax, 'Pause')
        self.pause_button.on_clicked(self.toggle_pause)

        self.speed_up_ax = plt.axes([0.6, 0.01, 0.1, 0.04])
        self.speed_up_button = plt.Button(self.speed_up_ax, 'Speed Up')
        self.speed_up_button.on_clicked(self.speed_up)

        self.speed_down_ax = plt.axes([0.3, 0.01, 0.1, 0.04])
        self.speed_down_button = plt.Button(self.speed_down_ax, 'Slow Down')
        self.speed_down_button.on_clicked(self.speed_down)

        # Set tight layout to avoid overlapping
        self.fig.tight_layout(rect=[0, 0.05, 1, 0.95])

        # Set legend
        for ax in self.axs.flatten()[:-1]:
            ax.legend()

    def update(self, frame):
        """Update the plot data."""
        stats = self.simulator.stats

        if stats.times:
            # Update line data
            self.queue_line.set_data(stats.times, stats.queue_lengths)
            self.system_line.set_data(stats.times, stats.system_customers_series)
            self.wait_line.set_data(stats.times, stats.mean_waiting_times)
            self.system_time_line.set_data(stats.times, stats.mean_system_times)
            self.server_line.set_data(stats.times, stats.server_loads)

            # Adjust axes limits
            for ax in self.axs.flatten()[:-1]:
                lines = ax.get_lines()
                if lines:
                    xdata = lines[0].get_xdata()
                    ydata = lines[0].get_ydata()
                    if len(xdata) > 0:
                        ax.set_xlim(min(xdata), max(xdata) + 1)
                        if len(ydata) > 0:
                            ymin = min(ydata)
                            ymax = max(ydata)
                            y_margin = (ymax - ymin) * 0.1 if ymax > ymin else 0.1
                            ax.set_ylim(max(0, ymin - y_margin), ymax + y_margin)

            # Update Little's Law verification
            littles_verification = stats.verify_littles_law()
            match_text = "✓" if littles_verification["match"] else "✗"
            littles_text = (
                f"Little's Law Verification:\n\n"
                f"L (mean customers) = {littles_verification['L']:.2f}\n"
                f"λW (arrival × time) = {littles_verification['lambda*W']:.2f}\n\n"
                f"Match: {match_text}"
            )
            self.littles_text.set_text(littles_text)

        return [self.queue_line, self.system_line, self.wait_line,
                self.system_time_line, self.server_line, self.littles_text]

    def toggle_pause(self, event):
        """Toggle the paused state of the simulator."""
        self.simulator.paused = not self.simulator.paused
        self.pause_button.label.set_text('Resume' if self.simulator.paused else 'Pause')
        plt.draw()

    def speed_up(self, event):
        """Increase the simulation speed."""
        self.simulator.speed_factor *= 2.0
        plt.draw()

    def speed_down(self, event):
        """Decrease the simulation speed."""
        self.simulator.speed_factor /= 2.0
        plt.draw()

    def run(self, interval=100):
        """Run the animation with the given frame interval (ms)."""
        self.animation = FuncAnimation(
            self.fig, self.update, interval=interval, blit=True,
            cache_frame_data=False, save_count=100)
        plt.show()


def run_simulation_with_visualization(
        simulator_type: str,
        arrival_type: str,
        arrival_rate: float = None,
        service_distribution: str = "exponential",
        num_servers: int = 1,
        duration: float = 100000,
):
    """Run the simulation with real-time visualization."""
    # Set up the arrival rate function
    if arrival_type == "constant":
        if arrival_rate is None:
            arrival_rate = OFF_SEASON_RATE
        arrival_rate_fn = constant_arrival_rate(arrival_rate)
    else:  # seasonal
        arrival_rate_fn = seasonal_arrival_rate()

    # Create the simulator
    if simulator_type == "fixed":
        simulator = FixedTimeSimulator(
            arrival_rate_fn=arrival_rate_fn,
            mean_service_time=MEAN_SERVICE_TIME,
            service_time_distribution=service_distribution,
            num_servers=num_servers,
            time_step=1.0
        )
    else:  # event-based
        simulator = EventBasedSimulator(
            arrival_rate_fn=arrival_rate_fn,
            mean_service_time=MEAN_SERVICE_TIME,
            service_time_distribution=service_distribution,
            num_servers=num_servers
        )

    # Start the visualization
    visualizer = SimulationVisualizer(simulator)

    # Start the simulation in a separate thread
    import threading
    sim_thread = threading.Thread(
        target=simulator.run,
        args=(duration, 10.0),
        daemon=True
    )
    sim_thread.start()

    # Run the visualization (this will block until the window is closed)
    visualizer.run()


def compare_arrival_rates():
    """Run simulations with different arrival rates and plot results for comparison."""
    # Set up arrival rates to test (as multiples of the base rate)
    rate_multipliers = [1, 5, 10, 15, 20]

    # Results storage
    results = []

    print("Comparing different arrival rates...")

    for multiplier in rate_multipliers:
        # Calculate arrival rate
        arrival_rate = OFF_SEASON_RATE * multiplier

        # Create simulator
        simulator = EventBasedSimulator(
            arrival_rate_fn=constant_arrival_rate(arrival_rate),
            mean_service_time=MEAN_SERVICE_TIME,
            service_time_distribution="exponential",
            num_servers=1
        )

        # Run simulation silently (high speed, no visualization)
        simulator.speed_factor = 1000.0
        stats = simulator.run(duration=100000, stats_interval=1000.0)

        # Save results
        littles_law = stats.verify_littles_law()
        results.append({
            "multiplier": multiplier,
            "arrival_rate": arrival_rate,
            "queue_length": stats.queue_lengths[-1] if stats.queue_lengths else 0,
            "system_customers": stats.system_customers_series[-1] if stats.system_customers_series else 0,
            "waiting_time": stats.mean_waiting_times[-1] if stats.mean_waiting_times else 0,
            "system_time": stats.mean_system_times[-1] if stats.mean_system_times else 0,
            "server_load": stats.server_loads[-1] if stats.server_loads else 0,
            "littles_law_match": littles_law["match"]
        })

        print(
            f"  Multiplier {multiplier}x: Server load = {results[-1]['server_load']:.2f}, Wait time = {results[-1]['waiting_time']:.2f}s")

    # Plot results
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Extract data for plotting
    multipliers = [r["multiplier"] for r in results]
    queue_lengths = [r["queue_length"] for r in results]
    waiting_times = [r["waiting_time"] for r in results]
    system_times = [r["system_time"] for r in results]
    server_loads = [r["server_load"] for r in results]

    # Plot queue length
    axs[0, 0].plot(multipliers, queue_lengths, 'bo-')
    axs[0, 0].set_title('Mean Queue Length')
    axs[0, 0].set_xlabel('Arrival Rate Multiplier')
    axs[0, 0].set_ylabel('Customers')
    axs[0, 0].grid(True)

    # Plot waiting time
    axs[0, 1].plot(multipliers, waiting_times, 'ro-')
    axs[0, 1].set_title('Mean Waiting Time')
    axs[0, 1].set_xlabel('Arrival Rate Multiplier')
    axs[0, 1].set_ylabel('Time (s)')
    axs[0, 1].grid(True)

    # Plot system time
    axs[1, 0].plot(multipliers, system_times, 'go-')
    axs[1, 0].set_title('Mean System Time')
    axs[1, 0].set_xlabel('Arrival Rate Multiplier')
    axs[1, 0].set_ylabel('Time (s)')
    axs[1, 0].grid(True)

    # Plot server load
    axs[1, 1].plot(multipliers, server_loads, 'mo-')
    axs[1, 1].set_title('Mean Server Load')
    axs[1, 1].set_xlabel('Arrival Rate Multiplier')
    axs[1, 1].set_ylabel('Utilization (0-1)')
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


def compare_service_distributions():
    """Run simulations with different service time distributions and plot results."""
    # Service distributions to test
    distributions = ["exponential", "uniform", "normal"]

    # Results storage
    results = []

    print("Comparing different service time distributions...")

    for dist in distributions:
        # Create simulator
        simulator = EventBasedSimulator(
            arrival_rate_fn=constant_arrival_rate(OFF_SEASON_RATE * 10),  # Use 10x rate for clearer differences
            mean_service_time=MEAN_SERVICE_TIME,
            service_time_distribution=dist,
            num_servers=1
        )

        # Run simulation silently (high speed, no visualization)
        simulator.speed_factor = 1000.0
        stats = simulator.run(duration=100000, stats_interval=1000.0)

        # Save results
        results.append({
            "distribution": dist,
            "queue_length": stats.queue_lengths[-1] if stats.queue_lengths else 0,
            "waiting_time": stats.mean_waiting_times[-1] if stats.mean_waiting_times else 0,
            "system_time": stats.mean_system_times[-1] if stats.mean_system_times else 0,
            "server_load": stats.server_loads[-1] if stats.server_loads else 0,
        })

        print(
            f"  Distribution {dist}: Queue length = {results[-1]['queue_length']:.2f}, Wait time = {results[-1]['waiting_time']:.2f}s")

    # Plot results
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Extract data for plotting
    x = [r["distribution"] for r in results]
    queue_lengths = [r["queue_length"] for r in results]
    waiting_times = [r["waiting_time"] for r in results]
    system_times = [r["system_time"] for r in results]
    server_loads = [r["server_load"] for r in results]

    # Plot queue length
    axs[0, 0].bar(x, queue_lengths)
    axs[0, 0].set_title('Mean Queue Length')
    axs[0, 0].set_xlabel('Service Time Distribution')
    axs[0, 0].set_ylabel('Customers')

    # Plot waiting time
    axs[0, 1].bar(x, waiting_times)
    axs[0, 1].set_title('Mean Waiting Time')
    axs[0, 1].set_xlabel('Service Time Distribution')
    axs[0, 1].set_ylabel('Time (s)')

    # Plot system time
    axs[1, 0].bar(x, system_times)
    axs[1, 0].set_title('Mean System Time')
    axs[1, 0].set_xlabel('Service Time Distribution')
    axs[1, 0].set_ylabel('Time (s)')

    # Plot server load
    axs[1, 1].bar(x, server_loads)
    axs[1, 1].set_title('Mean Server Load')
    axs[1, 1].set_xlabel('Service Time Distribution')
    axs[1, 1].set_ylabel('Utilization (0-1)')

    plt.tight_layout()
    plt.show()


def compare_server_counts():
    """Compare performance with different numbers of servers."""
    # Server counts to test
    server_counts = [1, 2, 3]

    # Results storage
    results = []

    print("Comparing different numbers of servers...")

    for num_servers in server_counts:
        # Create simulator
        simulator = EventBasedSimulator(
            arrival_rate_fn=constant_arrival_rate(OFF_SEASON_RATE * 15),  # Use 15x rate for clearer differences
            mean_service_time=MEAN_SERVICE_TIME,
            service_time_distribution="exponential",
            num_servers=num_servers
        )

        # Run simulation silently (high speed, no visualization)
        simulator.speed_factor = 1000.0
        stats = simulator.run(duration=100000, stats_interval=1000.0)

        # Save results
        results.append({
            "num_servers": num_servers,
            "queue_length": stats.queue_lengths[-1] if stats.queue_lengths else 0,
            "waiting_time": stats.mean_waiting_times[-1] if stats.mean_waiting_times else 0,
            "system_time": stats.mean_system_times[-1] if stats.mean_system_times else 0,
            "server_load": stats.server_loads[-1] if stats.server_loads else 0,
        })

        print(
            f"  Servers: {num_servers}, Queue length = {results[-1]['queue_length']:.2f}, Wait time = {results[-1]['waiting_time']:.2f}s")

    # Plot results
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Extract data for plotting
    x = [r["num_servers"] for r in results]
    queue_lengths = [r["queue_length"] for r in results]
    waiting_times = [r["waiting_time"] for r in results]
    system_times = [r["system_time"] for r in results]
    server_loads = [r["server_load"] for r in results]

    # Plot queue length
    axs[0, 0].bar(x, queue_lengths)
    axs[0, 0].set_title('Mean Queue Length')
    axs[0, 0].set_xlabel('Number of Servers')
    axs[0, 0].set_ylabel('Customers')

    # Plot waiting time
    axs[0, 1].bar(x, waiting_times)
    axs[0, 1].set_title('Mean Waiting Time')
    axs[0, 1].set_xlabel('Number of Servers')
    axs[0, 1].set_ylabel('Time (s)')

    # Plot system time
    axs[1, 0].bar(x, system_times)
    axs[1, 0].set_title('Mean System Time')
    axs[1, 0].set_xlabel('Number of Servers')
    axs[1, 0].set_ylabel('Time (s)')

    # Plot server load
    axs[1, 1].bar(x, server_loads)
    axs[1, 1].set_title('Mean Server Load')
    axs[1, 1].set_xlabel('Number of Servers')
    axs[1, 1].set_ylabel('Utilization (0-1)')

    plt.tight_layout()
    plt.show()


def run_seasonal_analysis():
    """Run simulation with seasonal arrival rates and analyze results."""
    print("Running seasonal arrival rate analysis...")

    # Create simulator with seasonal arrival rates
    simulator = EventBasedSimulator(
        arrival_rate_fn=seasonal_arrival_rate(),
        mean_service_time=MEAN_SERVICE_TIME,
        service_time_distribution="exponential",
        num_servers=1
    )

    # Run simulation for a full year (in seconds)
    SECONDS_IN_YEAR = 365 * 24 * 60 * 60

    # Run simulation silently (high speed, no visualization)
    simulator.speed_factor = 10000.0
    stats = simulator.run(duration=SECONDS_IN_YEAR, stats_interval=SECONDS_IN_YEAR / 1000)

    # Plot results over time
    fig, axs = plt.subplots(3, 1, figsize=(14, 14))

    # Convert time to days for better readability
    days = [t / (24 * 60 * 60) for t in stats.times]

    # Plot queue length
    axs[0].plot(days, stats.queue_lengths, 'b-')
    axs[0].set_title('Mean Queue Length Throughout the Year')
    axs[0].set_xlabel('Day of Year')
    axs[0].set_ylabel('Customers')
    axs[0].grid(True)

    # Plot waiting time
    axs[1].plot(days, stats.mean_waiting_times, 'r-')
    axs[1].set_title('Mean Waiting Time Throughout the Year')
    axs[1].set_xlabel('Day of Year')
    axs[1].set_ylabel('Time (s)')
    axs[1].grid(True)

    # Plot server load
    axs[2].plot(days, stats.server_loads, 'g-')
    axs[2].set_title('Server Load Throughout the Year')
    axs[2].set_xlabel('Day of Year')
    axs[2].set_ylabel('Utilization (0-1)')
    axs[2].grid(True)

    # Add vertical spans for peak seasons
    # Half-year peak (Jun-Aug)
    # Convert month and day to day of year
    DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    half_year_start = sum(DAYS_PER_MONTH[:5]) + 1  # June 1
    half_year_end = sum(DAYS_PER_MONTH[:8])  # August 31

    # End of fiscal year peak (Mar 15-31)
    fiscal_year_start = sum(DAYS_PER_MONTH[:2]) + 15  # March 15
    fiscal_year_end = sum(DAYS_PER_MONTH[:2]) + 31  # March 31

    for ax in axs:
        ax.axvspan(half_year_start, half_year_end, alpha=0.2, color='yellow', label='Half-Year Peak')
        ax.axvspan(fiscal_year_start, fiscal_year_end, alpha=0.2, color='orange', label='Fiscal Year Peak')
        ax.legend()

    plt.tight_layout()
    plt.show()


def analyze_second_server_profitability():
    """Analyze whether adding a second server would be profitable."""
    print("Analyzing profitability of adding a second server...")

    # Define billing and cost parameters
    BILLING_RATE = 0.01  # $ per second of call
    AGENT_COST_PER_HOUR = 20  # $ per hour per agent
    AGENT_COST_PER_SECOND = AGENT_COST_PER_HOUR / 3600

    # Results storage
    results = []

    # Test with different arrival rates
    rate_multipliers = [5, 10, 15, 20]
    server_counts = [1, 2]

    for multiplier in rate_multipliers:
        arrival_rate = OFF_SEASON_RATE * multiplier

        for num_servers in server_counts:
            # Create simulator
            simulator = EventBasedSimulator(
                arrival_rate_fn=constant_arrival_rate(arrival_rate),
                mean_service_time=MEAN_SERVICE_TIME,
                service_time_distribution="exponential",
                num_servers=num_servers
            )

            # Run simulation silently (high speed, no visualization)
            simulator.speed_factor = 1000.0
            stats = simulator.run(duration=1000000, stats_interval=10000.0)

            # Calculate metrics
            calls_per_second = len(stats.completed_customers) / stats.times[-1]
            avg_service_time = stats.mean_system_time()

            # Calculate revenue and costs
            revenue_per_second = calls_per_second * avg_service_time * BILLING_RATE
            cost_per_second = num_servers * AGENT_COST_PER_SECOND
            profit_per_second = revenue_per_second - cost_per_second

            # Calculate customer satisfaction metric (inversely related to wait time)
            customer_satisfaction = 100 / (
                    1 + stats.mean_waiting_times[-1] / 60)  # 100% if no wait, decreases with wait time

            # Save results
            results.append({
                "multiplier": multiplier,
                "arrival_rate": arrival_rate,
                "num_servers": num_servers,
                "waiting_time": stats.mean_waiting_times[-1],
                "system_time": stats.mean_system_times[-1],
                "server_load": stats.server_loads[-1],
                "calls_per_second": calls_per_second,
                "revenue_per_second": revenue_per_second,
                "cost_per_second": cost_per_second,
                "profit_per_second": profit_per_second,
                "customer_satisfaction": customer_satisfaction
            })

            print(
                f"  Rate: {multiplier}x, Servers: {num_servers}, Profit: ${profit_per_second:.4f}/s, Satisfaction: {customer_satisfaction:.1f}%")

    # Plot profitability comparison
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Group results by arrival rate multiplier
    for multiplier in rate_multipliers:
        one_server = next(r for r in results if r["multiplier"] == multiplier and r["num_servers"] == 1)
        two_servers = next(r for r in results if r["multiplier"] == multiplier and r["num_servers"] == 2)

        # Plot data points with connecting lines
        axs[0, 0].plot([1, 2], [one_server["profit_per_second"], two_servers["profit_per_second"]], 'o-',
                       label=f'{multiplier}x')
        axs[0, 1].plot([1, 2], [one_server["waiting_time"], two_servers["waiting_time"]], 'o-', label=f'{multiplier}x')
        axs[1, 0].plot([1, 2], [one_server["server_load"], two_servers["server_load"]], 'o-', label=f'{multiplier}x')
        axs[1, 1].plot([1, 2], [one_server["customer_satisfaction"], two_servers["customer_satisfaction"]], 'o-',
                       label=f'{multiplier}x')

    # Set titles and labels
    axs[0, 0].set_title('Profit per Second')
    axs[0, 0].set_xlabel('Number of Servers')
    axs[0, 0].set_ylabel('Profit ($/s)')
    axs[0, 0].grid(True)

    axs[0, 1].set_title('Mean Waiting Time')
    axs[0, 1].set_xlabel('Number of Servers')
    axs[0, 1].set_ylabel('Time (s)')
    axs[0, 1].grid(True)

    axs[1, 0].set_title('Server Load')
    axs[1, 0].set_xlabel('Number of Servers')
    axs[1, 0].set_ylabel('Utilization (0-1)')
    axs[1, 0].grid(True)

    axs[1, 1].set_title('Customer Satisfaction')
    axs[1, 1].set_xlabel('Number of Servers')
    axs[1, 1].set_ylabel('Satisfaction (%)')
    axs[1, 1].grid(True)

    # Add legends
    for ax in axs.flatten():
        ax.legend(title='Arrival Rate')

    plt.tight_layout()
    plt.show()


def main():
    """Main function to handle command line arguments and run simulations."""
    parser = argparse.ArgumentParser(description='Call Center Simulation')

    parser.add_argument('--mode', choices=['interactive', 'analysis'], default='interactive',
                        help='Interactive mode shows live visualization, analysis mode runs batch analysis')

    parser.add_argument('--simulator', choices=['fixed', 'event'], default='event',
                        help='Type of simulator to use')

    parser.add_argument('--arrival', choices=['constant', 'seasonal'], default='constant',
                        help='Type of arrival rate model')

    parser.add_argument('--rate', type=float, default=None,
                        help='Arrival rate multiplier (for constant arrival rate)')

    parser.add_argument('--service', choices=['exponential', 'uniform', 'normal'], default='exponential',
                        help='Service time distribution')

    parser.add_argument('--servers', type=int, default=1,
                        help='Number of servers')

    parser.add_argument('--duration', type=float, default=100000,
                        help='Simulation duration (in seconds)')

    parser.add_argument('--analysis', choices=['rates', 'distributions', 'servers', 'seasonal', 'profitability', 'all'],
                        default='all', help='Type of analysis to run (in analysis mode)')

    args = parser.parse_args()

    # Interactive mode with visualization
    if args.mode == 'interactive':
        print(f"Running interactive simulation with {args.simulator} simulator...")
        run_simulation_with_visualization(
            simulator_type=args.simulator,
            arrival_type=args.arrival,
            arrival_rate=OFF_SEASON_RATE * args.rate if args.rate else None,
            service_distribution=args.service,
            num_servers=args.servers,
            duration=args.duration
        )

    # Analysis mode (batch runs for comparison)
    else:
        if args.analysis == 'rates' or args.analysis == 'all':
            compare_arrival_rates()

        if args.analysis == 'distributions' or args.analysis == 'all':
            compare_service_distributions()

        if args.analysis == 'servers' or args.analysis == 'all':
            compare_server_counts()

        if args.analysis == 'seasonal' or args.analysis == 'all':
            run_seasonal_analysis()

        if args.analysis == 'profitability' or args.analysis == 'all':
            analyze_second_server_profitability()


if __name__ == "__main__":
    main()
