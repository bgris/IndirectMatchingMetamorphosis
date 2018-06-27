#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 19:03:32 2018

@author: barbara
"""


import copy


def GradientDescent(niter, epsV, epsZ, functional, X_init):
    X = copy.deepcopy(X_init)
    grad_op = functional.gradient
    energy=functional(X)
    print(" initial energy : {}".format(energy))

    for k in range(niter):
        grad=grad_op(X)
        X_temp0=X.copy()
        X_temp0[0]= (X[0]- epsV *grad[0]).copy()
        X_temp0[1]= (X[1]- epsZ *grad[1]).copy()
        energy_temp0=functional(X_temp0)
        if energy_temp0<energy:
            X=X_temp0.copy()
            energy=energy_temp0
            epsV*=1.1
            epsZ*=1.1
            print(" iter : {}  , energy : {}".format(k,energy))
        else:
            X_temp1=X.copy()
            X_temp1[0]= (X[0]- epsV *grad[0]).copy()
            X_temp1[1]= (X[1]- 0.5*epsZ *grad[1]).copy()
            energy_temp1=functional(X_temp1)
    
            X_temp2=X.copy()
            X_temp2[0]= (X[0]- 0.5*epsV *grad[0]).copy()
            X_temp2[1]= (X[1]- epsZ *grad[1]).copy()
            energy_temp2=functional(X_temp2)
    
            X_temp3=X.copy()
            X_temp3[0]= (X[0]- 0.5*epsV *grad[0]).copy()
            X_temp3[1]= (X[1]- 0.5*epsZ *grad[1]).copy()
            energy_temp3=functional(X_temp3)
    
            if (energy_temp3<=energy_temp1 and energy_temp3<=energy_temp2):
                X_temp0=X_temp3.copy()
                energy_temp0=energy_temp3
                epsZ*=0.5
                epsV*=0.5
            else:
                if (energy_temp1<=energy_temp2 and energy_temp1<=energy_temp3):
                    X_temp0=X_temp1.copy()
                    energy_temp0=energy_temp1
                    epsZ*=0.5
                else:
                    X_temp0=X_temp2.copy()
                    energy_temp0=energy_temp2
                    epsV*=0.5
    
            if energy_temp0<energy:
                X=X_temp0.copy()
                energy=energy_temp0
                epsV *= 1.1
                epsZ *= 1.1
                print(" iter : {}  , energy : {}".format(k,energy,epsV, epsZ))
            else:
                print("iter : {}, reducing steps".format(k))

    return X