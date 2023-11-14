import numpy as np
import matplotlib.pyplot as plt
import random,math
import sys

def random_walk(n=100,step=1):
    x, y = 0, 0
    timeframe= np.arange(n+1)
    positions=[y]
    directions=["U","D"]

    for i in range(1,n+1):
        choice=random.choice(directions)
        if choice == "U":
            y+=step
            positions.append(y)
        if choice == "D":
            y-=step
            positions.append(y)
    return positions,timeframe


def brownianMotion(mu=0,sigma=1,steps=100,delta_t=0.01):
    t=np.linspace(0,steps*delta_t,steps)
    dB = np.sqrt(delta_t) * np.random.randn(num_steps)
    B = np.cumsum(dB)
    drift = (mu - 0.5 * sigma**2) * t
    return sigma * B + drift


def geometric_brownian_motion(S0=0, mu=0, sigma=1, num_steps=500, delta_t=0.01):
    t = np.linspace(0, num_steps * delta_t, num_steps)
    dB = np.sqrt(delta_t) * np.random.randn(num_steps)
    B = np.cumsum(dB)
    drift = (mu - 0.5 * sigma**2) * t
    return S0 * np.exp(drift + sigma * B)

def simple_bond(A,r):
    """ returns the no arbritage price of a bond that pays $A at the end of one year with interest rate at r percent"""
    cost= A/(1+(r/100))
    return cost

def present_value(c,r):
    """ returns the present value of a contract with payoffs c and interest rate of the corresponding time period"""
    
    t=len(c)
    present_value=0
    for i in range(t):
        present_value+=c[i]/(1+(r/100))**i

    return present_value

def dual_contract(c,r_l,r_b):
    upper_range=present_value(c,r_l)
    lower_range=present_value(c,r_b)
    return [lower_range,upper_range]

def perpetuity(A,r):
    """Fixed amount of A received for all intervals"""
    r=r/100
    return A/r

def annuity(A,r,n):
    """Fixed amount of A received upto n intervals"""
    a=0
    r=r/100
    for i in range(n):
        a+=A/(1+r)**i
    return a


def forward_price(S_0,r,T,dividend='n',c=[],r_d=[]):
    """computes the forward price of a forward contract at time 0 upto a maturity time T
    S_0: current price
    r : rate in decimals
    T: time of delivery
    dividend: non-dividend paying or not
    c: list of dividend payments, empty by default
    r_d: array of discounted rates from time 0"""
    a = S_0*(1+r)**T
    if dividend=='n':
        return a
    elif dividend == 'y':
        cd=0;
        for i in range(len(c)):
            cd+= c[i] * (1+r_d[i])**i
        a+=cd
    return a

## european call option for n periods

def n_period_binomial(S_0,R,u,K,n=1,c=0,mode='call'):
    q=(R-d-c)/(u-d)
    
    ## building the prices array
    rows,cols=(n+1,n+1)
    prices=np.array([[0 for i in range(rows)] for j in range(cols)],dtype=np.float16)
    for i in range(rows):
        for j in range(cols):
            if i<=j:
                prices[i][j]=(u**(j-2*i))*S_0
    print(prices)
    
    p_rows,p_cols=(n+1,n)
    payoffs=np.array([[0 for i in range(p_cols)] for j in range(p_rows)],dtype=np.float16)
    
    #building the payoffs array
    for i in range(p_rows):
        for j in range(p_cols):
            payoffs[i][j]=max(prices[i][j+1]-K,0)
    print(payoffs[:,-1]) 
    
    # building the risk neutral probabilities
    cols=(n+1)
    risk_prob=np.array([0 for j in range(cols)],dtype=np.float16)
    for i in range(cols):
        risk_prob[i]=math.comb(n,i)*(q**(n-i))*((1-q)**i)
    risk_prob.reshape(1,n+1)
    print(risk_prob)
    
    a=np.matmul(risk_prob,payoffs[:,-1])
    return a/(R**n)

def black_scholes(call_put_flag, S, X, T, r, sigma):
    """ calculates the price of the option using the Black-Scholes formula
    parameters:-
    call_put_flag: A string specifying whether the option is a call ('c') or a put ('p')
    S: The current stock price
    X: The option strike price
    T: The time until option expiration, expressed in years
    r: The risk-free interest rate
    sigma: The stock's annualized volatility"""
    d1 = (math.log(S / X) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if call_put_flag == 'c':
        option_price = S * norm_cdf(d1) - X * math.exp(-r * T) * norm_cdf(d2)
    elif call_put_flag == 'p':
        option_price = X * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    else:
        raise ValueError('Call/put flag must be "c" or "p".')
    return option_price

def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

