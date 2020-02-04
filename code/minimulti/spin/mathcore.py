#!/usr/bin/env python
import numpy as np


def Euler_integration(dfunc, f0, step, check_step=False,
                      allowed_max_delta=1e-2):
    """
    f(x+delta t)= f(x)+ df(x)/dt * delta t
    """
    df0 = dfunc(f0)
    if check_step:
        mdelta = np.max(np.abs(df0))
        if mdelta * step > allowed_max_delta:
            print(
                "Warning: delta > max_delta. Please check if the step is too large. Suggested step: ",
                0.01 * 1e12 / mdelta)
    f1 = f0 + df0 * step
    return f1


def Heun_integration(dfunc, f0, step, check_step=False,
                     allowed_max_delta=1e-2):
    """
    f(x+delta t)= f(x)+ df(x)/dt * delta t
    but with a corrected df(x)
    """
    df0 = copy.deepcopy(dfunc(f0))
    f1 = f0 + df0 * step
    #print("df0:", df0)
    if check_step:
        mdelta = np.max(np.abs(df0))
        if mdelta * step > allowed_max_delta:
            print(
                "Warning: delta > max_delta. Please check if the step is too large. Suggested step: ",
                0.005 * 1e12 / mdelta)
    df1 = dfunc(f1)
    f2 = f0 + (0.5 * step)* (df0 + df1) 
    return f2
