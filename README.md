# Call Center Simulation
This project simulates a call center with multiple servers (staff members) handling customer calls to analyze performance metrics and staffing requirements.

## Command-line Arguments:
```python3 call_center.py [options]```

| Argument | Description                                            |
|----------|--------------------------------------------------------|
| `--servers N` | Number of servers/staff (default: 1)                   |
| `--time T` | Simulation duration in seconds (default: 10000)        |
| `--rate R` | Manual arrival rate multiplier (default: 1x base rate) |

### Example commands
Basic simulation with constant arrival rate:

```python3 call_center.py --rate 1 --servers 1```

Half year peak with 10-fold increase and two staff members:

```python3 call_center.py --rate 10 --servers 2```

## Output Metrics

The simulation generates graphs showing:
- Mean queue length
- Mean number of customers in system
- Mean queue time
- Mean time in system
- Mean server load (utilization)
- Little's Law verification