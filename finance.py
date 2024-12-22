import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objs as go
import matplotlib.pyplot as plt

n_assets=5
n_simulation=500

returns= np.random.rand(n_assets,n_simulation)

rand=np.random.rand(n_assets)
weights=rand/sum(rand)
def port_return(returns):
    rets=np.mean(returns, axis=1)
    cov=np.cov(rets.T, aweights=weights, ddof=1)
    portfolio_returns=np.dot(weights, rets.T)
    portfolio_std_dev=np.sqrt(np.dot(weights, np.dot(cov, weights)))
    return portfolio_returns, portfolio_std_dev

portfolio_returns , portfolio_std_dev=port_return(returns)

print(portfolio_returns)
print(portfolio_std_dev)

portfolio=np.array([port_return(np.random.randn(n_assets, i ))
                        for i in range(1,101)])

best_fit=sm.OLS(portfolio[:,1],sm.add_constant(portfolio[:,0]))\
                        .fit().fittedvalues

fig=go.Figure()

fig = go.Figure()
fig.add_trace(go.Scatter(name='Risk-Return Relationship',
            x=portfolio[:, 0],
            y=portfolio[:, 1], mode='markers'))
fig.add_trace(go.Scatter(name='Best Fit Line',
            x=portfolio[:, 0],
            y=best_fit, mode='lines'))
fig.update_layout(xaxis_title = 'Return',
            yaxis_title = 'Standard Deviation',
            width=900, height=470)
fig.show()
#############################este es un nuevo ejercicio#########################
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-darkgrid')

def utility(x):
    return (np.exp(x**gamma))

pl=np.random.uniform(0,1,20)
pl=np.sort(pl)

print(" The highest three probability of loses are {}".format(pl[-3:]))

y = 2
c = 1.5
Q = 5
D = 0.01
gamma = 0.4

def supply(Q):
    return (np.mean(pl[-Q:])* c)

def demand(D):
    return (np.sum(utility(y-D) >pl *utility(y-c)+(1-pl)*utility(y)))

plt.figure()
plt.plot([demand(i) for i in np.arange(0,1.9,0.02)], np.arange(0,1.9, 0.02),"r",label="insurance demand")
plt.plot(range(1,21),[supply(j) for j in range(1,21)],"g",label="insurance supply")
plt.ylabel("Average Cost")
plt.xlabel("Number of People")
plt.legend()
plt.show()
#############################este es un nuevo ejercicio#########################
import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

hola=2