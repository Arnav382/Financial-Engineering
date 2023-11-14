import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize

########################################################################################################################

def short_rate_lattice(r_i=0.05,u=1.1,d=0.9,n=10):
    """creates a short rate lattice given the initial short rate,up factor, down factor and number of periods
    
    Parameters:
    r_i(float): the initial short rate
    u(float): up-factor
    d(float): down-factor
    n(int): no of periods
    
    Returns:
    np.ndarray: A 2-D numpy array with the short rate lattice"""
    
    rates=np.zeros((n+1,n+1))
    rates[0][0]=r_i
    
    for i in range(1,n+1):
        rates[i][0]=rates[i-1][0]*d
        
        for j in range(1,i+1):
            rates[i][j]=rates[i-1][j-1]*u
            
    return rates

########################################################################################################################


## pricing of zero coupon bonds
### without coupon bearing

def zcb(r_i=0.05,u=1.1,d=0.9,coupon=100.0,del_t=10,q=0.5):
    """creates a zero coupon bond lattice with initial short rate, up factor, down factor, coupon value
    
    Parameters:
    r_i(float): the initial short rate
    u(float): up-factor
    d(float): down-factor
    coupon(float): value upon maturity
    n(int): no of periods
    q(float): probability of up factor
    Returns:
    np.ndarray: A 2-D numpy array with the coupon price lattice
    """
    rates=short_rate_lattice(r_i,u,d,del_t)
    prices=np.zeros((del_t+1,del_t+1))
    for j in range(del_t+1):
        prices[del_t][j]=coupon
    for i in range(del_t-1,-1,-1):
        for j in range(i+1):
            prices[i][j]=(q*prices[i+1][j]+(1-q)*prices[i+1][j+1])/(1+rates[i][j])
    return prices.round(1)


########################################################################################################################


## with coupon bearing
def coupon_bearing_bond(r_i=0.05,u=1.1,d=0.9,coupon=100,n=10,q=0.5,del_t=4,per=0.1,print_matrice=False):
    
    """returns the forward price of the coupon bearing bond with initial short rate, up factor,
       down factor, coupon value, delivery time, divident percent return and the boolean expression of
       printing forward prices matrice
    
    Parameters:
    r_i(float): the initial short rate
    u(float): up-factor
    d(float): down-factor
    coupon(float): value upon maturity
    n(int): no of periods
    q(float): probability of up factor
    del_t(int): delivery time of bond dividends
    per(float): dividend percent return
    print_matrice(boolean): prints the coupon bearing bond lattice if set to true
    int: forward price of the coupon bearing bond
    Returns:
    np.ndarray: A 2-D numpy array with the coupon price lattice"""
    
    rates=short_rate_lattice(r_i,u,d,n)
    prices=np.zeros((n+1,n+1))
    for j in range(n+1):
        prices[n][j]=(1+per)*coupon
        for i in range(n-1,-1,-1):
            for j in range(i+1):
                prices[i][j]=(q*prices[i+1][j]+(1-q)*prices[i+1][j+1])/(1+rates[i][j])
                if i==del_t+1: prices[i][j]+=coupon*per
    if print_matrice:
        return prices.round(2)
    
    a=zcb(r_i,u,d,coupon,del_t,q)[0][0]
    
    return (coupon*prices[0][0]/a).round(2)


########################################################################################################################


### pricing of European options on zero coupon bonds(ZCB)

def european_op(zcb=a,exp_t=2,K=84.,sr_lat=sr,q=0.5,type_="c"):
    """
    computes european option price of a zcb with no coupon bearing with expiration time and strike price
    
    Parameters:
    zcb(2-D numpy array): zero coupon bond pricing lattice
    exp_t(int): expiration time
    K(float): strike price
    q(float): probability of up factor
    type_(char): type of option, default is call(c), other option is put(p)
    Returns:
    np.ndarray: A 2-D numpy array with the coupon price lattice
    """
    
    zcb=zcb[:exp_t+1]
    n=zcb.shape[0]
    new=np.zeros((n,n))
    payoff=zcb[-1:][0]
    
    if type_=="c":
        for i in range(n):
            payoff[i]=max(payoff[i]-K,0)
            new[n-1][i]=payoff[i]
        
        for i in range(exp_t-1,-1,-1):
            for j in range(i+1):
                new[i][j]=(q*new[i+1][j]+(1-q)*new[i+1][j+1])/(1+sr[i][j])
    
    elif type_=="p":
        for i in range(n):
            payoff[i]=max(K-payoff[i],0)
            new[n-1][i]=payoff[i]
        
        for i in range(exp_t-1,-1,-1):
            for j in range(i+1):
                new[i][j]=(q*new[i+1][j]+(1-q)*new[i+1][j+1])/(1+sr[i][j])
    else:
        print("Invalid type, returning empty matrice")
    return new


########################################################################################################################


### pricing of European options on zero coupon bonds(ZCB)

def american_op(zcb=a,exp_t=6,K=80.,sr_lat=sr,q=0.5,type_="c"):
    """computes American put option price of a zcb with no coupon bearing with expiration time and strike price
    
    Parameters:
    zcb(2-D numpy array): zero coupon bond pricing lattice
    exp_t(int): expiration time
    K(float): strike price
    q(float): probability of up factor
    type_(char): type of option, default is put(p), other option is call(c)"""
    zcb=zcb[:exp_t+1]
    n=zcb.shape[0]
    new=np.zeros((n,n))
    payoff=zcb[-1:][0]
    if type_=="p":
        for i in range(n):
            payoff[i]=max(K-payoff[i],0)
            new[n-1][i]=payoff[i]
    
        for i in range(exp_t-1,-1,-1):
            for j in range(i+1):
                new[i][j]=max(K-zcb[i][j],(q*new[i+1][j]+(1-q)*new[i+1][j+1])/(1+sr[i][j]))
                
    elif type_=="c":
        for i in range(n):
            payoff[i]=max(payoff[i]-K,0)
            new[n-1][i]=payoff[i]
    
        for i in range(exp_t-1,-1,-1):
            for j in range(i+1):
                new[i][j]=max(zcb[i][j]-K,(q*new[i+1][j]+(1-q)*new[i+1][j+1])/(1+sr[i][j]))
    else:
        print("Invalid type, returning empty matrice")
    return new


########################################################################################################################


### futures contract of bonds pricing

def future_cb(cbb=a,exp_t=4,q=.5,print_matrice=False):
    """
    Returns np.darray if print_matrice is set to true, else returns the bond futures price at initial time"""
    ca = cbb[exp_t]
    prices=np.zeros((exp_t+1,exp_t+1))
    for j in range(exp_t+1):
        prices[exp_t][j]=ca[j]

    for i in range(exp_t-1,-1,-1):
        for j in range(i+1):
            prices[i][j]=(q*prices[i+1][j]+(1-q)*prices[i+1][j+1])
            
    if print_matrice:
        return prices
    
    return prices[0][0]

########################################################################################################################
### caplet pricing

def caplet(K=0.02,t=6,sr_lattice=q_i,p=.5,print_matrice=True):
    """
    """
    q=sr_lattice[-1:][0]
    arr=np.zeros((t,t))
    
    for j in range(t):
        arr[t-1][j]= max(0,q[j]-K)
        arr[t-1][j]=arr[t-1][j]/(1+q_i[t-1][j])
    
    for i in range(t-2,-1,-1):
        for j in range(i+1):
            arr[i][j]=(p*arr[i+1][j]+(1-p)*arr[i+1][j+1])/(1+q_i[i][j])
    if print_matrice:
        return arr.round(4)
    return arr[0][0]

########################################################################################################################

### pricing swaps

def swaps(K=0.05,t=6,sr_lattice=q_i,p=.5,print_matrice=False):
    """
    """
    q=q_i[-1:][0]
    arr=np.zeros((t,t))
    
    for j in range(t):
        arr[t-1][j]= q[j]-K
        arr[t-1][j]=arr[t-1][j]/(1+q_i[t-1][j])
    for i in range(t-2,-1,-1):
        for j in range(i+1):
            arr[i][j]=((q_i[i][j]-K) + p*arr[i+1][j]+(1-p)*arr[i+1][j+1])/(1+q_i[i][j])
    if print_matrice:
        return arr.round(5)
    return arr[0][0]

### swaption pricing

def swaption(K=0.05,exp_t=3,sr_lattice=q_i,p=.5,print_matrice=False):
    """ pricing of european call swaptions
    """
    t=q_i.shape[0]
    q=q_i[-1:][0]
    prices=swaps(K,t,sr_lattice,p,print_matrice=True)
    arr=prices
    for j in range(t):
        arr[exp_t][j]=max(0,arr[exp_t,j])
        
    for i in range(exp_t-1,-1,-1):
        for j in range(i+1):
            arr[i][j]=(p*arr[i+1][j]+(1-p)*arr[i+1][j+1])/(1+q_i[i][j])
    
    if print_matrice:
        return arr.round(5)
    return arr[0][0]

########################################################################################################################

### Forward equations

def forward_eqn_lattice(short_rate_lattice=q_i,for_spot_rate=False):
    """computes and returns forward equation lattice"""
    n=short_rate_lattice.shape[0]
    fwd=np.zeros((n+1,n+1))
    fwd[0,0]=1.
    fwd[1,0]=1/(2*(1+short_rate_lattice[0,0]))
    fwd[1,1]=1/(2*(1+short_rate_lattice[0,0]))
    
    for i in range(2,n):
        fwd[i][0]= 0.5*fwd[i-1,0]/(1+short_rate_lattice[i-1,0])
        for j in range(1,i+1):
            fwd[i][j]= ((fwd[i-1,j]/(2*(1+short_rate_lattice[i-1,j]))) + (fwd[i-1,j-1]/(2*(1+short_rate_lattice[i-1,j-1]))))
    
    fwd[n][0]=0.5*fwd[n-1,0]/(1+short_rate_lattice[n-1,0])
    fwd[n][n]=0.5*fwd[n-1,n-1]/(1+short_rate_lattice[n-1,n-1])
    
    for j in range(1,n):
        fwd[n][j]=((fwd[n-1,j]/(2*(1+short_rate_lattice[n-1,j]))) + (fwd[n-1,j-1]/(2*(1+short_rate_lattice[n-1,j-1]))))
        
    if for_spot_rate==False:
        fwd=np.delete(fwd,n,0)
        fwd=np.delete(fwd,n,1)
    return fwd.round(8)
 
########################################################################################################################

### Black Derman Toy (BDT) model

def BDT(n=14,a=14*[5.],b=0.005):
    """"""
    lattice=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            lattice[i,j]=a[i]*np.exp(b*j)
    return lattice.round(2)/100

### building forward rate lattice using BDT lattice

def forward_price_lattice(short_rate_lattice=BDT()):
    zcb_bdt=forward_eqn_lattice(short_rate_lattice,for_spot_rate=True).round(3)
    zcb_bdt_p=np.zeros(zcb_bdt.shape[0]-1)
    for i in range(zcb_bdt.shape[0]-1):
        zcb_bdt_p[i]=zcb_bdt[i+1].sum()
    return zcb_bdt_p.round(3)

#### calculating the spot rates

def spot_rate(price_arr=zcb_bdt_p):
    n=len(price_arr)
    arr=np.zeros(n)
    for i in range(n):
        arr[i]=(1/price_arr[i])**(1/(i+1))
    return arr-1

########################################################################################################################
########################################################################################################################
## calibration of BDT

def mse(actual,predicted):
    return ((actual-predicted)**2).sum()
########################################################################################################################
########################################################################################################################


## defaultable bond pricing

def defaultable_bond(F=100,n=4,c=0.02,interest_rate=0.025,R=0.25,hazard_rate=0.02,return_h_rate=False):
    """returns the price of a defaultable bond 
    Parameters:
    F: Face value of the bond (int)
    n: time period of the bond(time to maturity) (int)
    interest_rate: interest rate of the market at initial period (float)
    R: recovery rate (float)
    return_h_rate: returns the hazard rate array if set to True, Used for calibration of model (boolean)
    Returns:
    price of the bond (int)"""
    hazard_rates=n*[hazard_rate]
    
    cpf=n*[c*F]
    cpf[n-1]+=F

    prob_s=np.zeros(n+1)
    prob_d=np.zeros(n+1)
    discount_rate=np.zeros(n+1)
    prob_s[0]=1
    prob_d[0]=0
    discount_rate[0]=1
    for i in range(1,n+1):
        prob_s[i]=prob_s[i-1]*(1-hazard_rates[i-1])
        prob_d[i]=prob_s[i-1]*hazard_rates[i-1]
        discount_rate[i]=1/(1 + interest_rate)**i
    
    dr=discount_rate[1:]
    p_s=prob_s[1:]
    p_d=prob_d[1:]
    a=dr*(p_s*cpf+p_d*F*R)
    
    if return_h_rate:
        return hazard_rates, a.sum()
    
    return a.sum()

########################################################################################################################
