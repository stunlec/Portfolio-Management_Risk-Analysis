import numpy as np
import pandas as pd
from math import sqrt
import scipy.stats as stats
from scipy.optimize import minimize

def skew(ser):
    mean = np.mean(ser)
    sdev = np.std(ser)
    sumdev = 0
    for i in range(len(ser)):
        diff = ser[i]-mean
        sumdev = sumdev + (diff**3)
    num = sumdev/(sdev**3)
    skew = num/(len(ser)-1)
    return skew

def kurt(ser):
    mean = np.mean(ser)
    sdev = np.std(ser)
    sumdev = 0
    for i in range(len(ser)):
        diff = ser[i]-mean
        sumdev = sumdev + (diff**4)
    num = sumdev/(sdev**4)
    kurt = num/(len(ser)-1)
    return kurt

def ret(ser):
    n=len(ser)
    ret_arr=np.zeros((1,n))
    for i in range(n):
        if i==0:
            ret_arr[0][0]=0
        else:
            ret_arr[0][i]=(ser[i]-ser[i-1])/ser[i-1]
    return ret_arr
        

def ann_ret(ser,N):
    n=len(ser)
    ret=1
    for i in range(n):
        ret *= (1+ser[i])
    ann_ret = pow(ret,1/N)-1
    return ann_ret*100

def ann_vol(ser):
    sdev = np.std(ser)
    ann_vol = sdev*sqrt(12)
    return ann_vol

def Sharpe_Ratio(ser):
    mean = np.mean(ser)
    sd = np.std(ser)

    SR=((mean/100)-(0.03))/(sd*sqrt(12))
    return SR

def Jarque_bera(ser):
    kurt = kurt(ser)
    skew = skew(ser)
    n = float(len(ser))
    
    JB = (n/6)*((skew**2) + (1/4)*(kurt-3.0)**2)
    
    if JB>=0.0 and JB<=0.05:
        return True
    else:
        return False

def drawdown(ini_inv,ser):
    wealth_index = ini_inv*(1 + ser).cumprod()
    prev_peaks = wealth_index.cummax()
    drawdown = (wealth_index-prev_peaks)/prev_peaks
    wealth_index.plot()
    prev_peaks.plot()
    drawdown.plot()
    
    return pd.DataFrame({"Wealth": wealth_index,
                        "Previous Peaks": prev_peaks,
                        "Drawdown": drawdown})

def semi_dev(ser):
    mean = np.mean(ser)
    total = 0
    for i in range(len(ser)):
        if (ser[i]<mean):
            diff = mean-ser[i]
            total+=(diff**2)
    var = total/len(ser)
    return sqrt(var)

def hist_var(ser,t_period,alpha):
    ser = np.array(ser)
    ser = np.sort(ser)
    index = int((t_period*(alpha/100))-1)
    return ser[index]

def hist_cvar(ser,t_period,alpha):
    ser = np.array(ser)
    ser = np.sort(ser)
    index = int((t_period*(alpha/100))-1)
    neg_dev = ser[:index]
    return np.mean(neg_dev)

def gauss_var(ser,alpha=5):
    mean = np.mean(ser)
    sdev = np.std(ser)
    z = -1.65
    var = -(mean+(z*sdev))
    return var

def gauss_cvar(ser,alpha=5):
    ser = np.array(ser)
    var = gauss_var(ser)
    sum_ret = 0
    for i in range(len(ser)):
        if(ser[i]<var):
            sum_ret+=ser[i]
    cvar = sum_ret/len(ser)
    return cvar

def cf_var(ser,alpha=5):
    S = skew(ser)
    K = kurt(ser)
    z = -1.65
    term1 = (1/6)*((z**2)-1)*S
    term2 = (1/24)*((z**3)-(3*z))*(K-3)
    term3 = (1/36)*(2*(z**3)-(5*z))*(S**2)
    mean = np.mean(ser)
    sdev = np.std(ser)
    Z = z+term1+term2-term3
    var = -(mean+Z*sdev)/(1-0.05)
    return var

def portfolio_vol(weights,cov_matrix):
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    return portfolio_volatility

def portfolio_return(weights,er):
    ret = np.dot(weights, er)
    avg_ret = np.mean(ret)
    return ret

def minimize_vol(target_return, er, cov):
    
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights,er)
    }
    weights = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)
    return weights.x

def optimal_weights(n_points, er, cov):
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights

def plot_ef(n_points, er, cov):
    
    weights = optimal_weights(n_points, er, cov) 
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    return ef.plot.line(x="Volatility", y="Returns", style='.-')

def msr(riskfree_rate, er, cov):
    
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    weights = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x
