import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp, odeint
w=1

def rhs(t,y):
    return (y[1],-(w**2)*y[0])

sol1=solve_ivp(rhs, (0,100), (1,0) ,method='RK23', t_eval=np.linspace(0,100,1000), vectorized=True)
sol2=solve_ivp(rhs, (0,100), (1,0), t_eval=np.linspace(0,100,1000), vectorized=True)
sol3=solve_ivp(rhs, (0,100), (1,0), method='Radau', t_eval=np.linspace(0,100,1000), vectorized=True)

plt.plot(sol1.t, sol1.y[0,:], color='blue')
plt.plot(sol2.t, sol2.y[0,:], color='red')
plt.plot(sol3.t, sol3.y[0,:], color='green')

plt.show()
