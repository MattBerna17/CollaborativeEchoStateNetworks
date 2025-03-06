import numpy as np
from scipy.integrate import odeint
import pandas as pd

def lorenz(y, t, sigma, rho, beta):
    x1, x2, x3 = y
    dydt = [sigma*(x2-x1) , x1*(rho - x3) - x2 , x1*x2 - beta*x3 ]
    return dydt


if __name__ == "__main__":
    # generate lorenz attractor data
    sigma = 10; rho=28; beta=8/3
    y0 = [5, -8, 10] # initial condition
    endtime = 50 # 2000
    dt_odeint = 0.01
    t = np.linspace(0,endtime, int(endtime/dt_odeint)) # time points
    timewarp = 1
    yt = odeint(lorenz,y0,timewarp*t,args=(sigma,rho,beta)) # solve ODE
    df = pd.DataFrame(pd.DataFrame(pd.concat([pd.DataFrame(t, columns=["t"]), pd.DataFrame(yt, columns=["x", "y", "z"])], axis=1))).reset_index(drop=True)
    df.to_csv("lorenz.csv", )
    