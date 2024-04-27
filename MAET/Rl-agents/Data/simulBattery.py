import numpy as np
import matplotlib.pyplot as plt
def fullStairs(tspan, u, *args, **kwargs):
    """
    Draws a stairstep plot of u vs. tspan with the same number of stairs as the length of the input vectors.

    Args:
    tspan : numpy array of length K+1
    u : numpy array of dimension K x M or M x K

    Additional arguments and keyword arguments are passed to plt.step.
    """
    # Check if tspan is a vector
    if not tspan.ndim == 1:
        raise ValueError('tspan must be a one-dimensional array.')

    # Determine the number of input vectors
    nt = len(tspan)
    nRow, nCol = u.shape
    if nt == nRow + 1:
        nu = nRow
        u = np.vstack([u, u[-1, :]])  # Add the last row to the end
    elif nt == nCol + 1:
        nu = nCol
        u = np.column_stack([u, u[:, -1]])  # Add the last column to the end
    else:
        raise ValueError('The number of time steps must equal 1 + the number of input vectors to plot.')


import numpy as np


def generateDrivingPower(t, alpha):
    """
    Generates a time series of chemical power discharged to drive an electric vehicle.

    Args:
    t : numpy array, the K+1 vector time span in hours
    alpha : numpy array, the K vector energy intensity of driving in kWh/km

    Returns:
    pChemDrive : numpy array, a K vector of chemical powers discharged to drive in kW
    """
    K = len(t) - 1
    dt = t[1] - t[0]
    nd = int(K * dt / 24)
    if K * dt % 24 != 0:
        raise ValueError("The time span must contain an integer number of days.")

    # Reshape energy intensity to K/nd x nd (daily blocks)
    alpha = alpha.reshape((K // nd, nd))

    # Initialize driving power array
    pChemDrive = np.zeros((K // nd, nd))
    nt = 3  # number of trips per day

    # Simulate daily trips
    for j in range(nd):
        for i in range(nt):
            # Generate trip start time
            hStart = 6 + 14 * np.random.rand()
            kStart = int(hStart / dt)
            while pChemDrive[kStart, j] > 0:
                hStart = 6 + 14 * np.random.rand()
                kStart = int(hStart / dt)

            # Generate trip distance with a log-normal distribution
            dTrip = min(100, np.random.lognormal(mean=1.8, sigma=1.24))

            # Determine trip speed based on distance
            if dTrip < 15:
                sTrip = 40  # km/h for short trips
            else:
                sTrip = 90  # km/h for long trips

            # Calculate trip duration
            tTrip = dTrip / sTrip

            # Distribute the trip's power demand over its duration
            while tTrip > 0:
                pChemDrive[kStart, j] += alpha[kStart, j] * sTrip * min(dt, tTrip) / dt
                tTrip -= min(dt, tTrip)
                kStart += 1
                if kStart >= pChemDrive.shape[0]:
                    break

    # Flatten the matrix into a vector
    pChemDrive = pChemDrive.flatten()

    return pChemDrive


import numpy as np


def simulatePolicy3(x0, z, pChemDrive, a, tau, etac, etad, pcMax, xMax, xMin, t, hDeadline, xStar):
    """
    Simulates the third electric vehicle charging policy.

    Args:
    x0 : float, initial chemical energy of the battery in kWh
    z : numpy array, indicators that the vehicle is plugged in
    pChemDrive : numpy array, chemical powers discharged to drive in kW
    a : float, discrete-time dynamics parameter
    tau : float, battery's self-dissipation time constant in hours
    etac : float, charging efficiency of the battery
    etad : float, discharging efficiency of the battery
    pcMax : float, maximum charging electrical power in kW
    xMax : float, chemical energy capacity of the battery in kWh
    xMin : float, minimum acceptable chemical energy in kWh
    t : numpy array, simulation time span in hours
    hDeadline : float, charging deadline hour of the day
    xStar : float, desired charge at the deadline in kWh

    Returns:
    x3 : numpy array, stored chemical energies over time in kWh
    p3 : numpy array, electrical charging powers in kW
    """
    K = len(z)  # Number of time steps
    dt = t[1] - t[0]  # Time step duration

    # Initialization
    x3 = np.zeros(K + 1)
    x3[0] = x0
    pChem3 = -pChemDrive
    y3 = np.zeros(K)

    # Simulation
    for k in range(K):
        R_ = k * dt
        if z[k] == 1:
            if x3[k] >= xMax:
                x3[k + 1] = xMax
            elif x3[k] <= xMin:
                y3[k] = 1
                if R_ <= hDeadline:
                    sum_a = np.sum([a ** (hDeadline - i) for i in range(k, hDeadline)])
                    pChem3[k] = min(etac * pcMax, (xStar - a ** (hDeadline - k) * x3[k]) / ((1 - a) * tau * sum_a))
                else:
                    pChem3[k] = min(etac * pcMax, (xMax - (a * x3[k])) / ((1 - a) * tau))
                x3[k + 1] = a * x3[k] + (1 - a) * tau * pChem3[k]
            elif y3[k - 1] == 1:
                y3[k] = 1
                pChem3[k] = min(etac * pcMax, (xMax - (a * x3[k])) / ((1 - a) * tau))
                x3[k + 1] = a * x3[k] + (1 - a) * tau * pChem3[k]
            else:
                x3[k + 1] = a * x3[k]
        else:
            x3[k + 1] = a * x3[k] - (1 - a) * tau * pChemDrive[k] if pChemDrive[k] != 0 else a * x3[k]

    p3 = np.maximum(pChem3 / etac, etad * pChem3)
    p3[z == 0] = 0  # No electrical charging or discharging when unplugged

    return x3, p3

# Example usage
t = np.linspace(0, 10, 6)  # tspan from 0 to 10 with 6 points (0, 2, 4, 6, 8, 10)
u = np.random.rand(5, 1)  # Some random data with 5 time steps

# Data and Parameters
t0 = 0  # initial time, h
nd = 7  # number of days in time span
tf = t0 + 24 * nd  # final time, h
dt = 1 / 60  # time step duration, min
t = np.arange(t0, tf, dt)  # time span, min
K = len(t) - 1  # number of time steps

# EV parameters
tau = 1600  # self-dissipation time constant, h
kkk = -dt / tau
a = np.exp(kkk)  # discrete-time dynamics parameter
etac = 0.95  # charging efficiency
etad = etac  # discharging efficiency
pcMax = 11.5  # charging capacity, kW
pdMax = 0  # discharging capacity, kW
xMax = 80  # energy capacity, kWh
x0 = xMax  # initial energy, kWh
xMin = 0.5 * xMax  # minimum acceptable energy capacity, kWh
alph = 0.3 * np.ones(K)  # energy intensity of driving, kWh/km

# Generate discharge powers for driving
pChemDrive = generateDrivingPower(t, alph)  # this function needs to be defined in Python

# Plugged-in hours
z = np.zeros(K)  # indicator that vehicle is plugged in
z[(t % 24 < 6) | (t % 24 > 20)] = 1  # plug in overnight
z[pChemDrive > 0] = 0  # unplug if vehicle is driving

# Parameters for policy 3
hDeadline = 6  # hour of day of charging deadline, h
xStar = xMax  # charging target, kWh

# Simulation
x3, p3 = simulatePolicy3(x0, z, pChemDrive, a, tau, etac, etad, pcMax, xMax, xMin, t, hDeadline, xStar)

