#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""

Examples for SpinEq

Created by Marc-André on 2012-11-10.
Copyright (c) 2012 IGBMC. All rights reserved.
"""
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from SpinEq import Eq, NMR, SpinSys

Debug = False
def Dprint(*arg):
    """prints only if Debug is true"""
    if Debug:
        print(*arg)

##################################################
##################################################
##################################################
# a few examples
## First test Mass Action Law ####################
#
def EL_eq(Keq=1/45E-6, Ltot=1E-3, Etot=20E-6, verbose=False):
    """
    This one simulate a one site binding :
    L + E   <->  EL
    3 species : L  E  EL
                0  1  2
    """
    ##### describe the scene

    eq1 = Eq(Species=["E","L","EL"])   # we define a 3 species system
    L=0; E=1; EL=2      # we give them symbolic names
    
    eq1.set_K3(Keq, L, E, EL)       # set a reversible equilibirum, with Keq
    eq1.set_massconserv(Ltot, [1,0,1])  # tell what is transformed in what, in L
    eq1.set_massconserv(Etot, [0,1,1])  # and in E
    eq1.set_concentration(Ltot, L)               # set the total concentration of E and L
    eq1.set_concentration(Etot, E)
    #####
    if verbose:
        print ("Initial system:")
        print ("===============")
        eq1.report()            # set-up finished, print a report
    eq1.solve()        # solve the equilibrium, by default equilibrium time is 1E-3 second
    if verbose:
        print ("\n final concentrations  :")      # and report
        print (  " ====================")
        eq1.showconc()
    cf = eq1.get_concentration_array()
    if verbose:
        print ("[complex] / [Etotal] = %.2f %%"%(100*cf[EL]/(cf[E]+cf[EL])) )
    if Debug:
        print ("[Etot] ",cf[E]+cf[EL], Etot)
        print ("[Ltot] ",cf[L]+cf[EL], Ltot)
    return cf
##################################################
def EL_titr():
    "test EL_eq with varying value for Keq"
    Eoccu = []
    Lfree = []
    Kv = pow(10.0,np.linspace(-7,0,20))
    for K in Kv:
        cf = EL_eq(Keq=1/K)
        Eoccu.append(100*cf[2]/(cf[1]+cf[2]))
        Lfree.append(100*cf[0]/(cf[0]+cf[2]))
    plt.semilogx(Kv, Eoccu, label="% E occupied")
    plt.semilogx(Kv, Lfree, label="% L free")
    plt.legend()
    plt.xlabel("Keq for cste [L] and [E]")
    plt.show()
##################################################
def BSA(Ltot=2E-3, Etot=20e-6, verbose=False):
    """
    a protein with two affinity sites
    L + E   <-> E1L     high aff
    L + E   <-> E2L     low aff
    L + E1L <-> EL2
    L + E2L <-> EL2
    5 species : L  E  E1L  E2L  EL2
                0  1  2    3    4
    """
    # create object
    eq1 = Eq(Species=["E", "L", "E1L", "E2L", "EL2"])
    # it is convenient to define species
    L=0; E=1; E1L=2; E2L=3; EL2=4

    # set-up parameters - here Two Kd separated by a factor 20
    Khigh = 1/100E-6     # 100 uM
    Klow = 1/5E-3      # 5 mM
    eq1.set_K3(Khigh, L, E, E1L)
    eq1.set_K3(Klow, L, E, E2L)
    eq1.set_K3(Khigh, L, E2L, EL2)
    eq1.set_K3(Klow, L, E1L, EL2)
    eq1.set_concentration(Ltot, L)
    eq1.set_concentration(Etot, E)
    eq1.set_massconserv(Ltot, [1,0,1,1,2])  # in L
    eq1.set_massconserv(Etot, [0,1,1,1,1])  # in E
    if verbose:
        print ("Initial system:")
        print ("===============")
        eq1.report()
    eq1.solve()    # maxtime=1E-4, step=100,verbose=True)
    cf = eq1.get_concentration_array()
    if verbose:
        print ("\n final concentrations:")      # and report
        print (  " ====================")
        eq1.showconc()
        print ("[complex] / [Etotal] = %f %%"%(100*(cf[E1L]+cf[E2L]+cf[EL2])/(Etot)))
    return cf
##################################################
def BSA_titre():
    "titration of the BSA example with varying ligand concentration"
    # it is convenient to define species
    L=0; E=1; E1L=2; E2L=3; EL2=4
    Etot = 20E-6
    Ltot = pow(10,np.linspace(-6,0,100))
#    Ltot = [100E-6, 0.25E-3, 0.5E-3, 1E-3, 2E-3]
#    Ltot = [2E-3]
    res =  [BSA(L,Etot) for L in Ltot]
    H = [(r[E1L]+r[EL2])/Etot for r in res]
    L = [(r[E2L]+r[EL2])/Etot for r in res]

    # plt.semilogx(Ltot,H,label="high")
    # plt.plot(Ltot,L,label="low")
    plt.semilogx(Ltot,[r[E]/Etot for r in res], label="Efree")
    plt.plot(Ltot,[r[E1L]/Etot for r in res], label="E1L")
    plt.plot(Ltot,[r[E2L]/Etot for r in res], label="E2L")
    plt.plot(Ltot,[r[EL2]/Etot for r in res], label="EL2")
    plt.legend()
    plt.xlabel("[Ltot]")
    plt.show()
##################################################
def Robertson():
    """    
    simulate complex a kinetics equation
    interconvertion of A to C with the transient species B
    
    A   -k1-> B     0.04
    2B  -k2-> B+C   3e7
    B+C -k3-> A+C   1e4

    see :
    http://www.radford.edu/~thompson/vodef90web/problems/demosnodislin/Single/DemoRobertson/demorobertson.pdf
        H.H. Robertson. The solution of a set of reaction rate equations, pages 178–182. Academ Press, 1966.
    
    """
    Atot = 1.0
    eq1 = Eq(["A", "B", "C"])     # we define a 3 species system
    A=0; B=1; C=2
    eq1.set_K11cin(0.04, A, B )
    eq1.set_K12cin(3E7, B, B, C, stoechios=(2,1,1))
    eq1.set_K22cin(1E4, B, C, A, C)
    eq1.set_massconserv(Atot, [1,1,1])  # tell what is transformed in what, in A
    eq1.set_concentration(Atot, A)               # set the total concentration of A
    #####
    eq1.report()            # set-up finished, print a report
    t = pow(10.0,np.linspace(-6,6,1000))    # span a large range of time 1E-6 to 1E6 sec
    tt, res = eq1.solve(time=t, trajectory=True)        # solve the kinetics
    print ("\n final concentrations  :")      # and report
    eq1.showconc()
    res[:,B] *= 10000   # multiply B species to see it in plot
    for i in range(3):
        plt.semilogx(t, res[:,i], label=eq1.Spnames[i])
    plt.xlabel('time (sec)')
    plt.legend()
    plt.title('interconvertion of A to C with the transient species B (x 10000)')
    plt.show()
    
##################################################
### Then check spin dynamics #################
##################################################
def S2():
    """
    NOE in a 2 coupled spins system
    """
    # set-up equilibrium
    eq1 = NMR(1,2)
    #  add spin systems
    R1E = 1.0
    sigma = -0.2
    Relax = np.matrix([[R1E,sigma],[sigma,R1E]])
    eq1.set_spin(0, SpinSys(2,Relax))
    eq1.saturate(0, 0)
    eq1.report()
    print()
    cf = eq1.solve(maxtime=1)    # maxtime=1E-4, step=100,verbose=True)
    print ("saturate one spin, look at transfer after 1sec")
    print ("sigma/rho: %.2f  noe: %.2f"%(sigma/R1E, cf[2]-1.0))
##################################################
def T1():
    """
    compares selective and global T1 in 2 coupled spins system
    """
    # set-up equilibrium
    eq1 = NMR(1, 2)
    #  add spin systems
    R1E = 1.0       # non selectif T1 : spin-lattice relaxation
    sigma = -0.5    # strong sigma, corresponding to a large prot
    Relax = np.matrix([[R1E,sigma],[sigma,R1E]])
    eq1.set_spin(0, SpinSys(2,Relax))
    eq1.set_magnetization(0.0, 0, 0)   # pulse on one spin
    eq1.report()
    st = 10./1e-2
    (t,res) = eq1.solve(maxtime=10.0, step=st, trajectory=True) #,verbose=True)
    plt.plot(t, [r[1] for r in res ], label="selective pulse")
    # second exp
    eq1.set_magnetization(0.0, 0, 0)   # pulse on both spin
    eq1.set_magnetization(0.0, 0, 1)   # pulse on both spin
    eq1.report()
    st = 10./1e-2
    (t,res) = eq1.solve(maxtime=10.0, step=st, trajectory=True) #,verbose=True)
    plt.plot(t, [r[1] for r in res ], label="global excitation")
    plt.title("magnetization recovery\nin a two coupled spin system")
    plt.legend()
    plt.show()
##################################################
def T1n(n=5):
    """
    compares selective and global T1 in n coupled spins system
    """
    # set-up equilibrium
    eq1 = NMR(1, n)
    #  add spin systems
    R1E = 0.5*(n+1)       # non selectif T1 : spin-lattice relaxation
    sigma = -0.5    # strong sigma, corresponding to a large prot
    Relax = np.matrix(sigma*np.ones((n,n)))     # set sigma everywhere
    for i in range(n):
        Relax[i,i] = R1E    # set diagonal to T1
    eq1.set_spin(0, SpinSys(n,Relax))
    eq1.set_magnetization(0.0, 0, 0)   # pulse on one spin
    eq1.report()
    st = 10./1e-2
    (t,res) = eq1.solve(maxtime=10.0, step=st, trajectory=True) #,verbose=True)
    plt.plot(t, [r[1] for r in res ], label="selective pulse")
    # second exp
    for i in range(n):
        eq1.set_magnetization(0.0, 0, i)   # pulse on all spin
    eq1.report()
    st = 10./1e-2
    (t,res) = eq1.solve(maxtime=10.0, step=st, trajectory=True) #,verbose=True)
    plt.plot(t, [r[1] for r in res ], label="global excitation")
    plt.title("magnetization recovery\nin a %d coupled spin system"%n)
    plt.legend()
    plt.show()
##################################################
def Inv2():
    """
    spin inversion in a 2 coupled spins system
    """
    # set-up equilibrium
    eq1 = NMR(1, 2)
    #  add spin systems
    R1E = 1.0
    #        sigma = 0.2
    for sigma in (-0.5, ):#0.25, 0.0, -0.25, -0.5, -0.8, -0.9, -0.95):
        Relax = np.matrix([[R1E,sigma],[sigma,R1E]])
        eq1.set_spin(0, SpinSys(2,Relax))
        eq1.set_magnetization(-1.0, 0, 0)   # inverse one spin
        eq1.report()
        st = 10./1e-2
        (t,res) = eq1.solve(maxtime=30.0, step=st, trajectory=True) #,verbose=True)
        plt.plot(t, [r[1] for r in res ], label="$\sigma = %f$"%sigma)
        plt.plot(t, [r[2] for r in res ])
    plt.legend()
    plt.show()
#############################################
def Invn(n=5, sigma=-0.3, mode="inverse"):
    """
    spin inversion in a n coupled spins system in linear geometry
    sigma is the interspin cross relaxation, can be positive or negative
    mode can be "inverse" or "saturate"
    """ 
    # set-up equilibrium
    eq1 = NMR(1, n)
    #  add spin systems
    R1E = 1.0
    #        sigma = 0.2
#        Relax = np.matrix([[R1E,sigma],[sigma,R1E]])
    Relax = R1E*np.eye(n) + sigma*(np.diag(np.ones(n-1),1)+np.diag(np.ones(n-1),-1))
    Relax = np.matrix(Relax)
    print (Relax)
    eq1.set_spin(0, SpinSys(n, Relax))
    if mode == "inverse":
        eq1.set_magnetization(-1.0, 0, 0)   # inverse first spin
    else:
        eq1.saturate(0, 0)  # or saturate it
#        eq1.report()
    dt = 1e-2   # one step every 10msec.
    maxtime = 10.    # for 10 sec
    (t,res) = eq1.solve(maxtime=maxtime, step=maxtime/dt, trajectory=True) #,verbose=True)
    for i in range(1,n+1):
        plt.plot(t, [r[i] for r in res ], label="spin %d"%i)
    plt.title("$\sigma = %f$"%sigma)
    plt.legend()
    plt.show()
##################################################
###### Then mix both spins and equilibrium
##################################################
def CG():
    "fig 1 of Clore et Gronenborn. J Magn Reson. (1982) vol. 48 (3) pp. 402-417"
    pass  # !!!
def EL_STD(Khigh = 1/3E-4, Ltot=1E-3, Etot=1E-4, time=1.0, step=10, verbose=False, trajectory=False):
    """
    Saturation Difference Difference experience on the equilibrium
    E + L <-> EL 
    L (1 spin) + E (1 spin saturated) <--> EL (two spins - E spin saturated)
    
    """
    # set-up equilibrium
    Dprint ("K=", Khigh)
    eq1 = NMR(("E","L","EL"),4)
    L = 0; E = 1; EL = 2
    eq1.set_K3(Khigh, L, E, EL)
    eq1.set_concentration(Ltot,L)
    eq1.set_concentration(Etot,E)
    eq1.set_massconserv(Ltot,  [1,0,1])  # in L
    eq1.set_massconserv(Etot,  [0,1,1])  # in E
    # then add spin systems
    R1L = 1.0
    R1E = 10.0
    sigma = -0.5*R1E
    eq1.set_spin(L, SpinSys(1,R1L))         #L
    eq1.set_spin(E, SpinSys(1,R1E))         #E
    Relax = np.matrix([[R1E,sigma],[sigma,R1E]])
    eq1.set_spin(EL, SpinSys(2,Relax))      #EL
    eq1.set_spinflux(E,0,EL,1)      # which is which
    eq1.set_spinflux(L,0,EL,0)
    if verbose:
        print (r"\Starting\n")
        eq1.report()
    # compute equilibrium
    cf = eq1.solve(maxtime=1E-3, step=10)
#    eq1.Integrator = "odeint"
    # then apply saturation
    eq1.saturate(E, 0)                      # we saturate the protein
    eq1.saturate(EL, 1)                     # in both binding states !

    if verbose:
        print ("\nsaturation\n")
        eq1.report()
    res = eq1.solve(maxtime=time , step=step, trajectory=trajectory, verbose=verbose)
    finale = eq1.get_concentration_array()
    if verbose:
        print ("\n final concentrations:")
        eq1.showconc()
        print ("\n final magnetizations:")
        eq1.showmagn()
        print ("fraction occupied %f %%"%(100*finale[EL]/(finale[E]+finale[EL])))
    if not trajectory:
        print ("EL_STD: Kd %.2g     STD : %.2f %%"%(1/Khigh, 100*(1-res[3])))
    return res
##################################################
def EL_evol():
    "evolution of EL_STD upon variation of contact time and Kd"
    tmax = 1.0
    result = []
    for Kd in pow(10,np.linspace(-7,2,15)):
        (t, res) = EL_STD(Khigh=1/Kd, time=tmax, step=100, verbose=False, trajectory=True)
        plt.plot(t, [r[3] for r in res], label="Kd = %.2g"%Kd)
        print ("Kd : %.2g     STD : %.2f %%"%(Kd, 100*(1-res[len(t)-1,3])))
        result.append([Kd, 100*(1-res[len(t)-1,3])])
    plt.title("evolution of STD upon variation of contact time,\nfor varying $K_d$")
    plt.xlabel('time (sec)')
    plt.ylabel('NMR signal')
    plt.legend()
    plt.figure()
    result = np.array(result)
    plt.semilogx(result[:,0],result[:,1])
    plt.xlabel('$K_d$')
    plt.ylabel('% STD')
    plt.title("maximum STD as a function of $K_d$")
    plt.show()
##################################################
def EL_STD_titr(K = 1/45E-6):
    """
    evolution of EL_STD for varying Etot
    """
    Etotl = pow(10.0, np.linspace(-7,-2,20))
    for K in (1E-1, 1E-2, 1E-3, 1E-4,1E-5):
        i = 0
        Magn = np.zeros((4,len(Etotl)))
        for Etot in Etotl:
            cf = EL_STD(Khigh = 1/K, Etot=Etot, time=1)
            #cf = cff[-1]
            #print (cf)
            ss = (1.0-cf[3]) #*cf[0] + (1.0-cf[5])*cf[2])/1E-3
            #print ("% STD = ", 100*ss)
            Magn[:,i] = cf[3:]
            i += 1
    #        STD.append(cf[3])
            #print ("% L fixee ",(100*cf[2]/(cf[0]+cf[2])))
#        for i in range(4):
        plt.semilogx(Etotl,Magn[0,:],label="K=%f"%K)
    plt.legend()
    plt.show()
##################################################
def EL_STD_3s(Khigh = 1/45E-6, Ltot=1E-3, Etot=1E-4, time=5.0, verbose=False, trajectory=False):
    """
    STD exp on
    E + L <-> EL equilibrium
    L (1 spin) + E (1 spin saturated) <--> EL (two spins)
    Same as EL_STD, but here a three spin model is used for the protein-ligand complex :
    EL : (L - E1 - E2) where only E2 is saturated
    """
    # set-up equilibrium
    print ("K=", Khigh)
    eq1 = NMR(("E","L","EL"), 6)
    L = 0; E = 1; EL = 2
    eq1.set_K3(Khigh, L, E, EL)
    eq1.set_concentration(Ltot,L)
    eq1.set_concentration(Etot,E)
    eq1.set_massconserv(Ltot,  [1,0,1])  # in L
    eq1.set_massconserv(Etot,  [0,1,1])  # in E
    # then add spin systems
    R1L = R1E = 1.0
    sigma = -0.99
    RelaxE = np.matrix([[R1E, sigma],
                        [sigma, R1E]])
    RelaxEL = np.matrix([[R1E,  sigma, 0.0],
                        [sigma, R1E,   sigma],
                        [0.0,   sigma, R1E]])
    eq1.set_spin(L, SpinSys(1,R1L))         #L
    eq1.set_spin(E, SpinSys(2,RelaxE))         #E
    eq1.set_spin(EL, SpinSys(3,RelaxEL))      #EL
    eq1.set_spinflux(E,0,EL,1)      # which is which
    eq1.set_spinflux(E,1,EL,2)      # which is which
    eq1.set_spinflux(L,0,EL,0)
    if verbose:
        print ("\nDepart\n")
        eq1.report()
        print ("state_dic", eq1.state_dic)
        print ("dic_state", eq1.dic_state)
        print ("species2spins")
        for i in range(eq1.Nspecies):
            print (i, list(eq1.species2spins(i)))
        print ("spin2species")
        for i in range(eq1.Nspins):
            print (i, eq1.spin2species(i))
    # compute equilibrium
    cf = eq1.solve(maxtime=1E-2, step=100)
    # then apply saturation
    eq1.conc = cf[0:3]
    eq1.saturate(E, 1)                      # we saturate the protein
    eq1.saturate(EL, 2)
    if verbose:
        print ("\nsaturation\n")
        eq1.report()
    st = max(1,time/1E-3) # computes at 1ms resolution
    res = eq1.solve(maxtime=time, step=st, trajectory=trajectory, verbose=verbose)
    if trajectory:
        (t, cf3) = res
        plt.plot(t,cf3[:,3])
        (t, cf) = EL_STD(trajectory=True,time=time)
        plt.plot(t,cf[:,3])
        plt.show()
    return res
##################################################
def EL_3s_evol():
    "evolution of EL_STD_3s upon variation of contact time"
    tmax = 10.0
    for K in pow(10,np.linspace(-7,-3,5)):
        (t, res) = EL_STD_3s(Khigh=1/K, time=tmax, trajectory=True)
        plt.plot(t, [r[3] for r in res],label="K = %f"%K)
    plt.legend()
    plt.show()

##################################################
if __name__ == '__main__':
    # S2()
    Invn(sigma=-0.3)
    #Invn(n=8, sigma=-0.5, mode="saturate")
    #T1n(5)
    #EL_STD()
    #EL_evol()
    #BSA(verbose=True)
    #BSA_titre()
    #EL_eq()
    #Robertson()
    #EL_STD_3s(trajectory=True)
    #EL_3s_evol()