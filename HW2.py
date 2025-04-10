import numpy as np
import matplotlib.pyplot as plt

#HW brought to you by Robert Mark

R = 2000000             # Orbital radious
T = 7620                # Period of the orbit
PosStd = 500000        # Standard deviation of the position error
VelStd = 2000           # Standard deviaton of the velocity error
f = 0.1                 # Filtering frequency
w = 2 * np.pi / T       # Angular velocity
DataSize = int(T / f)   
Shape = (DataSize, 4)
Shape2 = (DataSize, 2)


def true_state(t, r, p):
    x = r * np.cos(w * t)
    y = r * np.sin(w * t)
    x_dot =-r * w * np.sin(w * t)
    y_dot = r * w * np.cos(w * t)
    return np.array([x, y, x_dot, y_dot]).transpose()


TrueData = true_state(np.linspace(0, T, DataSize), R, T)
Noise = np.zeros(Shape)
Noise[:, 0:2] = np.random.normal(loc=0.0, scale=PosStd, size=Shape2)
Noise[:, 2:4] = np.random.normal(loc=0.0, scale=VelStd, size=Shape2)
Observations = TrueData + Noise

dt = f
Predictions = np.zeros(shape=(DataSize, 4))
Predictions[0] = Observations[0]
F = np.array([[1, 0, dt,  0],                                       # State transition model
              [0, 1,  0, dt],
              [0, 0,  1,  0],
              [0, 0,  0,  1]])
B = np.array([[0.5 * dt**2,           0],                           # Control-input model
              [          0, 0.5 * dt**2],
              [         dt,           0],
              [          0,          dt]])
P = np.array([[PosStd**2,         0,         0,          0],        # Covariance matrix
              [        0, PosStd**2,         0,          0],
              [        0,         0, VelStd**2,          0],
              [        0,         0,         0,  VelStd**2]])       # Covariance of the observation noise
R = np.array([[PosStd**2,         0,         0,          0],
              [        0, PosStd**2,         0,          0],
              [        0,         0, VelStd**2,          0],
              [        0,         0,         0,  VelStd**2]])
H = np.identity(4)                                                  # Observation model, feel free to play with the values here
Q = np.array([[PosStd**2/4000000,         0,         0,          0],
              [        0, PosStd**2/4000000,         0,          0],
              [        0,         0, VelStd**2/4000000,          0],
              [        0,         0,         0,  VelStd**2/4000000]])
for i in np.arange(1, DataSize):
    x = Predictions[i - 1]                                          
    z = Observations[i]                                             
    u = -w**2 * x[0:2]

   
    x_priori = np.dot(F, x) + np.dot(B, u)
    P_priori = np.dot(np.dot(F, P), F.T) + Q
    y = z - np.dot(H, x_priori)
    S = np.dot(np.dot(H,P_priori), H.T) + R
    K = np.dot(np.dot(P_priori, H.T), np.linalg.inv(S))
    x_posteriori = x_priori + np.dot(K,y)
    P_posteriori = np.dot((np.eye(4) - np.dot(K, H)),  P_priori)
    y = z - np.dot(H,x_posteriori)
    Predictions[i] = x_posteriori
    P = P_posteriori

plt.plot(TrueData[:, 0], TrueData[:, 1], label="True position")
plt.plot(Predictions[:, 0], Predictions[:, 1], label="Predictions") #Uncomment when you have the predictions
plt.plot(Observations[:, 0], Observations[:, 1], 'r.', label="Observations",markersize=0.05) #Uncomment to see the input

plt.legend()
plt.show()
