import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from pandas import DataFrame
import math
import scipy.optimize as opt

def n_avg(list, i, j):
    sum = 0
    div = 0
    for offset_i in [-1,0, 1]:
        for offset_j in [-1,0, 1]:
            new_i = i + offset_i
            new_j = j + offset_j
            if (new_i >= 0 and new_j >= 0 and new_i < len(list) and new_j < len(list)):
                sum += list[new_i][new_j]
                div += 1
    avg = (sum-list[i][j]) / (div-1)
    return avg

def smoothing(list):
    for x in range(0,len(list)):
        for y in range(0,len(list)):
            if list[x][y] > 1.25*n_avg(list,x,y):
                list[x][y] = n_avg(list,x,y)
    return(list)

def smoothingFunction(list,iterations):
    runs=0
    while runs < iterations:
        for i in range(0,len(list)):
            smoothing(list[i])
        runs += 1

def data_fit(p0, func, xvar, yvar, err, tmi=0):
    try:
        fit = optimize.least_squares(residual, p0, args=(func,xvar, yvar, err), verbose=tmi)
    except Exception as error:
        print("Something has gone wrong:",error)
        return p0, np.zeros_like(p0), -1, -1
    pf = fit['x']

    print()

    try:
        cov = np.linalg.inv(fit['jac'].T.dot(fit['jac']))          
    except:
        print('Fit did not converge')
        print('Result is likely a local minimum')
        print('Try changing initial values')
        print('Status code:', fit['status'])
        print(fit['message'])
        return pf, np.zeros_like(pf), -1, -1

    chisq = sum(residual(pf, func, xvar, yvar, err) **2)
    dof = len(xvar) - len(pf)
    red_chisq = chisq/dof
    pferr = np.sqrt(np.diagonal(cov)) 
    pfcov = cov[0,1]
    print('Converged with chi-squared {:.2f}'.format(chisq))
    print('Number of degrees of freedom, dof = {:.2f}'.format(dof))
    print('Reduced chi-squared {:.2f}'.format(red_chisq))
    print()
    Columns = ["Parameter #","Initial guess values:", "Best fit values:", "Uncertainties in the best fit values:"]
    print('{:<11}'.format(Columns[0]),'|','{:<24}'.format(Columns[1]),"|",'{:<24}'.format(Columns[2]),"|",'{:<24}'.format(Columns[3]))
    for num in range(len(pf)):
        print('{:<11}'.format(num),'|','{:<24.3e}'.format(p0[num]),'|','{:<24.3e}'.format(pf[num]),'|','{:<24.3e}'.format(pferr[num]))
    return pf, pferr, chisq, dof

def residual(p,func, xvar, yvar, err):
    return (func(p, xvar) - yvar)/err

def gaussianfunc_bg(p,x):
    return p[0]/(p[2]*np.sqrt(2*np.pi))*np.exp(-(x-p[1])**2/(2*p[2]**2))+p[3]


def featureIdentification(norm_array,x,x_coord,y_coord,x_bound,y_bound):
    yx=norm_array[y_coord]
    dyx=np.sqrt(yx)
    x_guess=np.array([4*yx[x_coord],x_coord/2,2,0.1*yx[x_coord]])

    yy=norm_array[:,x_coord]
    dyy=np.sqrt(yy)
    y_guess=np.array([4*yy[y_coord],y_coord/2,2,0.1*yy[y_coord]])

    xmin_value,xmax_value=x_coord-x_bound[0], x_coord+x_bound[1]
    ymin_value,ymax_value=y_coord-y_bound[0], y_coord+y_bound[1]

    x_pf, x_pferr, x_chisq, x_dof = data_fit(x_guess, gaussianfunc_bg, x[xmin_value:xmax_value], yx[xmin_value:xmax_value], dyx[xmin_value:xmax_value])
    y_pf, y_pferr, y_chisq, y_dof = data_fit(y_guess, gaussianfunc_bg, x[ymin_value:ymax_value], yy[ymin_value:ymax_value], dyy[ymin_value:ymax_value])

    print(' ')
    print(ymin_value,ymax_value)
    print('===============================================')
    print('Feature identified at ({:.2f} +- {:.2f},{:.2f} +- {:.2f}) cm'.format(x_pf[1],x_pferr[1],y_pf[1],y_pferr[1]) )