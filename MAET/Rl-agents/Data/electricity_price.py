import numpy as np

def generate_load_profile(hours, peak_hours, mid_peak_hours):
    """Generate a simulated electricity load profile for 24 hours."""
    base_load = np.random.normal(0.5, 0.1, hours)  # Baseline load with some noise
    load_profile = np.zeros(hours)
    for hour in range(hours):
        if hour in peak_hours:
            load_profile[hour] = base_load[hour] * 1.5  # 50% higher during peak hours
        elif hour in mid_peak_hours:
            load_profile[hour] = base_load[hour] * 1.2  # 20% higher during mid-peak hours
        else:
            load_profile[hour] = base_load[hour]  # Normal load during off-peak hours
    return load_profile

def calculate_cost(load_profile, peak_hours, mid_peak_hours, peak_rate, mid_peak_rate, off_peak_rate):
    """Calculate the total cost of electricity based on the load profile and pricing."""
    cost = 0
    for hour, load in enumerate(load_profile):
        if hour in peak_hours:
            cost += load * peak_rate
        elif hour in mid_peak_hours:
            cost += load * mid_peak_rate
        else:
            cost += load * off_peak_rate
    return cost

def main():
    hours = 24
    peak_hours = set(range(17, 21))  # 5 PM to 8 PM
    mid_peak_hours = set(range(12, 17))  # Noon to 4 PM
    off_peak_hours = set(range(hours)) - peak_hours - mid_peak_hours

    # Rates in $/kWh
    peak_rate = 0.30
    mid_peak_rate = 0.20
    off_peak_rate = 0.10

    # Generate a load profile for a day
    load_profile = generate_load_profile(hours, peak_hours, mid_peak_hours)

    # Calculate total cost for the day
    total_cost = calculate_cost(load_profile, peak_hours, mid_peak_hours, peak_rate, mid_peak_rate, off_peak_rate)

    print(f"Total electricity cost for the day: ${total_cost:.2f}")

if __name__ == "__main__":
    main()
