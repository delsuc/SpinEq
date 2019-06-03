#!/usr/bin/env python
# encoding: utf-8
"""
Spineq.py

Created by Marc-Andre' on 2011-04-12.
Copyright (c) 2011 IGBMC. All rights reserved.

This program computes chemical equilibria along with NMR magnetization evolution.

Eq() is for chemical reactions
    simulate equlibria and kinetics
NMR() is for NMR experiement in an chemical reaction
    simulate Noe
    STD, TrNOE, etc...

all concentration are in M
all times are in sec

look at examples.py for tutorials

version dec-2013 - cleaned for stoechiometry and kinetics. examples in separated file.
version may-2014 - finalized, added licence and greating.
version apr-2017 - ported to python 3

parameters :
Debug = False
KinMax = 1E8        # the fastest kinetic speed possible
Precision = 1e-8    # precision used in ode integration
"""

from __future__ import print_function, division

name = 'Spineq'
license = "Cecill 2.0"
authors = 'Marc-Andre'' Delsuc <madelsuc@unistra.fr> and Bruno Kieffer <kieffer@igbmc.fr>'
version = "4 Apr 2017"

import numpy as np
np.set_printoptions(precision=5, threshold=None, edgeitems=None, linewidth=120, suppress=None, nanstr=None, infstr=None)
from scipy import integrate

Debug = False
KinMax = 1E8        # the fastest kinetic speed possible
Precision = 1e-8    # precision used in ode integration

print("""
    Welcome to the {0} program,  version {1}

    Licence : {2}
    Authors : {3}
    
    check examples.py for help
    """.format(name, version, license, authors))
    
def Dprint(*arg):
    """prints only if Debug is true"""
    if Debug:
        print(arg)
#######################################################
class Flux(object):
    """
    describe an elementary kinetic flux for the Eq class
    used by Eq()
    """
    def __init__(self, k):
        """each flux is defined as
        self.kin : the true kinetic constant
        self.destroyed : a list of destroyed species given as pairs (index,stoechio)
        self.created : a list of created species given as pairs     (index,stoechio)
        all values are in indexes in the Species list of the Eq object
        
        when called, a self.keff corresponding to the flux speed is computed;
        self.keff = $ k * \prod_{S \in destroyed} [S]$ where [S] is the concentration of S
        """
        if k>KinMax:
            raise Exception("error in kinetic parameter - value %f is too fast"%k)
        self.k = k
        self.destroyed = []
        self.created = []
        self.keff = 0.0     # this one keeps the current flux speed, computed at each steps
    def add_destroyed(self, index, stoechio=1):
        self.destroyed.append((index, stoechio))
    def add_created(self, index, stoechio=1):
        self.created.append((index, stoechio))
    def report(self):
        """prints a report"""
        print(self.k, self.destroyed, "-->", self.created, self.keff)
        
#######################################################
class Eq(object):
    """
    The main class for chemical equilibirum
    Creators are
    Eq(3)   : 3 anonymous species
    Eq(["A", "B", "C"]) : 3 species calles A, B, C
    Eq(["A", "B", "C"], varcreate=True) :  same as above, but the variable A=0 , B=1 and C=3 are created in the caller namespace
        usefull, but dangerous in large projects
    """
    def __init__(self, Nspecies=None, Species=None, varcreate=False ):
        if (not Nspecies) and (not Species):
            raise Exception("You should define either Nspecies (number of Species) or Species (liste of species names)")
        elif Nspecies and Species:
            if len(Species) != Nspecies:
                raise Exception("Nspecies and Species list mismatch ")
        elif Species and not Nspecies:
            Nspecies = len(Species)
        elif not Species and Nspecies:
            if type(Nspecies) != int:       # this happends when calling Eq(["A", "B", "C"]) 
                Species = Nspecies
                Nspecies = len(Species)
            else:
                Species = ["esp%d"%i for i in range(Nspecies)]
        self.Nspecies = Nspecies
        self.Spnames = Species        
        if varcreate:
            from inspect import currentframe
            frame = currentframe().f_back
            try:
                for i,v in enumerate(self.Spnames):
                    print(i,v)
                    frame.f_globals[v] = i
            finally:
                del frame  # break cyclic dependencies as stated in inspect docs    print locals()
        if Nspecies == 1:     # initialize concentrations at zeros unless there is only one species (NMR case)
            self.conc = np.ones(Nspecies)
        else:
            self.conc = np.zeros(Nspecies)
        self.kinlist = []       # initialize kinetic flux list
        self.massconserv = []   # initialize mass conservation matrix list
        self.diffusion = KinMax    # used for diffusion limited eq
        # Integrator determines which integrator is used
        #  ode  / odeint
        # odeint is probably to be prefered
        self.Integrator = "odeint"   # "ode"
    def showFlux(self, K):
        "display a kinetic in human readable format"
        def stname(f):
            " compute a string from flux f with molecule with stoechiometry "
            if f[1] == 1:
                st = self.Spnames[f[0]]
            else:
                st = "%d %s"%(f[1], self.Spnames[f[0]])
            return st
        left = " + ".join( [ stname(i)  for i in K.destroyed ] )
        right = " + ".join( [ stname(i) for i in K.created ] )
        print(left, "\t  - %.2e ->\t  "%K.k, right)

    def report(self):
        """prints a report"""
        print("""
Kinetics
========""")
        for K in self.kinlist:
            self.showFlux(K)
        print("\nmass conservation rules")
        for m in self.massconserv:
            print(m)
        print("\nconcentrations")
        self.showconc()
    def showconc(self):
        "show concentrations, "
        for i in range(self.Nspecies):
            if self.conc[i]==0.0:
                print("%s : \t 0.0"%(self.Spnames[i]))
            elif self.conc[i]>=1.0:
                print("%s : \t %.3f M"%(self.Spnames[i], self.conc[i]))
            elif self.conc[i]>1e-3:
                print("%s : \t %.3f mM"%(self.Spnames[i], 1E3*self.conc[i]))
            elif self.conc[i]>1e-6:
                print("%s : \t %.3f uM"%(self.Spnames[i], 1E6*self.conc[i]))
            else:
                print("%s : \t %.3f nM"%(self.Spnames[i], 1E9*self.conc[i]))
    def set_concentration(self, conc, i):
        """set initial (current) concentration of species i - in M"""
        self.conc[i] = conc
    def set_concentration_array(self, conc_array):
        """set all initial (current) concentrations - in M"""
        self.conc[:] = conc_array[:]
    def get_concentration(self, i):
        """get current concentration of species i - in M"""
        return self.conc[i]
    def get_concentration_array(self):
        """get an array of all concentration """
        return self.conc[:]
    def set_massconserv(self, tot_conc, array):
        """
        defines a mass conservation rule as a total concentration and an array
        used to give total concentrations

        eg in a Eq(3);  S1 + 2 S2 = S3    gives 2 rules : 
                [S1] + [S3] = A
            and [S2] + 2[S3] = B
            given as 
                eq.set_massconserv(A, [1,0,1])
                eq.set_massconserv(B, [0,1,2])
        """
        self.massconserv.append((tot_conc, np.array(array)))
    def set_K11cin(self, kcin, i, j, stoechios=(1,1)):
        """
        defines a kinetics flux Si -> Sj at speed kcin
        """
        K = Flux(kcin)
        K.add_destroyed(i, stoechios[0])
        K.add_created(j, stoechios[1])
        self.kinlist.append( K )
    def set_K21cin(self, kcin, i, j, k, stoechios=(1,1,1)):
        """
        defines a kinetics flux Si + Sj -> Sk at speed kcin
        """
        K = Flux(kcin)
        K.add_destroyed(i, stoechios[0])
        K.add_destroyed(j, stoechios[1])
        K.add_created(k, stoechios[2])
        self.kinlist.append( K )
    def set_K12cin(self, kcin, i, j, k, stoechios=(1,1,1)):
        """
        defines a kinetics flux Si -> Sj + Sk at speed kcin
        """
        K = Flux(kcin)
        K.add_destroyed(i, stoechios[0])
        K.add_created(j, stoechios[1])
        K.add_created(k, stoechios[2])
        self.kinlist.append( K )
    def set_K22cin(self, kcin, i, j, k, l, stoechios=(1,1,1,1)):
        """
        defines a kinetics flux Si + Sj -> Sk + Sl at speed kcin
        """
        K = Flux(kcin)
        K.add_destroyed(i, stoechios[0])
        K.add_destroyed(j, stoechios[1])
        K.add_created(k, stoechios[2])
        K.add_created(l, stoechios[3])
        self.kinlist.append( K )
    def set_K2(self, Keq, i, j, kon=None, stoechios=(1,1)):
        """defines the Si <-> Sj equilibrium constant"""
        if kon is None:
            kon = self.diffusion
        koff = kon/Keq
        self.set_K11cin(kon, i, j, stoechios=stoechios )
        self.set_K11cin(koff, j, i, stoechios=tuple(reversed(stoechios)))
    def set_K3(self, Keq, i, j, k, kon=None, stoechios=(1,1,1)):
        """
        defines the Si + Sj <-> Sk equilibrium constant
        
        """
        if Keq >=1:  # tight binding - limited in diffusion on kon
            if kon is None:
                kon = self.diffusion
            koff = kon/Keq
        else:   # weak binding, limited on diffusion on koff
            koff = self.diffusion
            kon = Keq*koff
        self.set_K21cin(kon, i, j, k, stoechios=stoechios)
        self.set_K12cin(koff, k, i, j, stoechios=(stoechios[2], stoechios[0], stoechios[1]))
    def _evol(self, t, conc):
        "used by ode"
        return self.evol(conc, t)
    def evol(self, conc, t=0):
        """
        compute derivative of evolution
        
        on each flux, di Di -k-> ci Cj  (d,c stoechio  D C concentrations )
        speed is K = k \prod_i [ conc(D_i)^{d_i} ]       // stored as flux.keff
        dDi/dt = -di K     dCj/dt = +ci K
        """
        dconc = np.zeros_like(conc)
        conc = np.maximum(conc, 0.0)  # remove negative concentrations - if ever

        # first, cure concentrations - happends due to unavoidable slow drift
#         for (c0, arr) in self.massconserv:
#             ctot = np.sum(arr*conc)    # current conc
#             error = ctot/c0    # negative if concentration is too high
#             print "error ", error
#             conc = conc + arr*conc*(error-1)

        # compute concentration variations
        for flux in self.kinlist:
            k = flux.k
            for (sp, st) in flux.destroyed:     # compute conc dependence
                k *= conc[sp]**st
            flux.keff = k   # store for later use
            for (sp, st) in flux.destroyed:     # compute derivatives
                dconc[sp] -= st*k
            for (sp, st) in flux.created:
                dconc[sp] += st*k
        #add mass conservation
        for (c0, arr) in self.massconserv:
            dct = np.sum(arr*dconc)    # variation of total conc - should be zero
            n = arr.sum()
            dconc -= dct/n
        # finally, cure concentrations - could happend due to unavoidable slow drift
        # for (c0, arr) in self.massconserv:
        #     ctot = np.sum(arr*conc)    # current conc
        #     error = (c0-ctot)/c0    # negative if concentration is too high
        #     dconc = 1E3*error*arr

        Dprint( "dconc:",dconc)
        return dconc
    def jacobian(self, t, conc):
        """
        compute jacobian of evolution   - not used - not debugged -
        
        on each flux, Di -k-> Cj
        speed is K = k \prod_i [ conc(D_i) ]       // stored as flux.keff
        dK/dDi = k \prod_{l!=i} [ conc(D_l) ]
        dK/dCJ = 0
        dXi^2/dt/dDj = -/+ dK/dDj   with Xi in Di, Cj
        
        """
        jac = np.zeros((self.Nspecies, self.Nspecies))  # jac(i,j) contains dDi^2/dt/dDj
        for flux in self.kinlist:
            for (sgn,chemlist) in ((-1,flux.destroyed), (1,flux.created)):  # join them, depends only on sign
                for (spi, sti) in chemlist:
                    for (spj, stj) in flux.destroyed:     # compute dDi^2/dt/dDj
                        k = flux.k
                        if spi != spj:
                            k *= conc[spi]          # first order only so far !
                        jac[spi,spj] = sgn*k
        print("jacobian:\n",jac)
        return jac
    def solve(self, maxtime=1E-3, step=2, trajectory=False, time=None, verbose=False):
        """
        compute evolution of equilibrium system
        maxtime : the duration of integration
        step : the number of steps
        time : if given, used for integration instead of maxtime and step - only if initialized with odeint
        trajectory : if true, returns all step, otherwise return the last point
        """
        if self.Integrator == "ode":        # True means integrate.ode False means integrate.odeint
            t = []
            y = []
            r = integrate.ode(self._evol).set_integrator('vode', method='bdf', with_jacobian=False)
            r.set_initial_value(self.conc, 0.0)
            dt = maxtime/step
            while r.successful() and r.t < maxtime:
                r.integrate(r.t+dt)
                if trajectory:
                    t.append(r.t)
                y.append(r.y)
        elif self.Integrator == "odeint":
            if time is not None:
                t = time
            else:
                if step>1:
                    t = np.linspace(0, maxtime, step)
                else:
                    t = [maxtime]
            y = integrate.odeint(self.evol, self.conc, t, rtol=Precision, atol=Precision )#, Dfun=self.jacobian)
            self.conc[:] = y[-1,:]
        else:
            raise Exception("wrong integrator")
        if verbose: print(t); print(y)
        if trajectory:
            return (t, y)
        else:
            return y[len(y)-1]
#######################################################
class SpinSys(object):
    """define a spin systems for the NMR class"""
    def __init__(self, Nspins, Relax, Mo=None, Mc=None):
        """
        defines the spin system
        Relax is the relaxation matrix
        Mo is the stationary state - set to ones() if not defined
        Mc is the current magnetisation - set to Mo if not defined
        """
        self.Nspins = Nspins
        if Mo is None:
            Mo = np.ones(Nspins)
        if Mc is None:
            Mc = Mo.copy()
        self.Relax = np.matrix(Relax)
        if Nspins>1:
            r_diagless = Relax - np.diag(np.diag(Relax))    # remove diagonal
            s = np.ravel( abs(r_diagless.sum(axis=0)) )      # and compute sum of off-diagonal values
            s = abs(s)
            for i in range(len(self.Relax)):
                self.Relax[i,i] = max(s[i],self.Relax[i,i])     # and force diagonal to be larger
        self.Mo = np.matrix(Mo)
        self.Mc = Mc
        if self.Relax.shape != ((Nspins, Nspins)) or abs(self.Relax.T - self.Relax).sum() != 0.0 :
            if Nspins >1:
                raise Exception("Relax matrix should be a symetric %d x %d matrix"%(Nspins, Nspins))
            else:
                raise Exception("Relax should be a scalar")
    def deriv(self, M):
        """computes time derivative
        dM/dt = -R(M-Mo)
        """
        dM = -(M-self.Mo)*self.Relax
#        Dprint M, M-self.Mo, dM
        return dM
        
    def report(self):
        """prints a report"""
        print("%d spin(s)  - Relax matrix : \n%s"%(self.Nspins, str(self.Relax)))
        print("equilibrium magnetizations :", self.Mo)
        print("initial magnetizations :", self.Mc)
        
#######################################################
class NMR(Eq):
    """
    The main class for NMR.
    Extends Eq class to handle longitudinal relaxation
    Handles T1 and NOE relaxation - this allows to compute selective and non-selective T1, TR-NOE, STD, etc...
    """
    def __init__(self, EqArgs, Nspins):
        """
        NMR object is initialized as a Eq() with an associated number of spins
        EqArgs is either an int (number of species) or a list of species
        """
        if type(EqArgs) == int:
            Nspecies = EqArgs
            super(NMR, self).__init__(Nspecies=EqArgs)
        else:
            super(NMR, self).__init__(Species=EqArgs)
            Nspecies = len(EqArgs)
        self.spinSysList = [None]*Nspecies             # will contain SpinSys objects
        self.spinflux = np.zeros((Nspins,Nspins))   # keeps track of which spin goes where
        self.Nspins = Nspins                        # This counts how many spins are defined
        self.state_dic = [None]*Nspecies            # keeps track of which states are in which species
                                                    # states of Species i starts at self.state_dic[i]
        self.dic_state = [None]*Nspins              # reciprocal of state_dic; state i belongs to species self.dic_state[i]
        self.saturated = []                         # This will keep a list of saturated spin-states
    def set_spin(self, i, S):
        """
        Defines ith spinsystem in spinSysList as S
        S should be a SpinSys instance
        """
        self.spinSysList[i] = S
        if i == 0:
            self.state_dic[i] = 0
        else:
            self.state_dic[i] = self.state_dic[i-1] + self.spinSysList[i-1].Nspins
        for j in range(S.Nspins):
            self.dic_state[j+self.state_dic[i]] = i
    def set_magnetization_array(self, magn_array):
        """
        set all magnetizations, along species, from magn_array
        """
        k = 0
        for i in range(self.Nspecies):
            for m in range(self.spinSysList[i].Nspins):
                self.set_magnetization(magn_array[k], i, m)
                k=k+1
    def set_magnetization(self, M, species, spin):
        """
        set the spin in a system to M - used to set-up initial magnetisation before evolution
        """
        self.spinSysList[species].Mc[spin] = M
    def get_magnetization(self, species, spin):
        """
        get the magnetization of the spin in a system
        """
        return self.spinSysList[species].Mc[spin]
    def get_magnetization_array(self):
        """
        get all magnetizations, along species, from magn_array
        """
        state =  np.zeros(self.Nspins)
        k = 0
        for i in range(self.Nspecies):             # set state
            S = self.spinSysList[i]
            state[k:k+S.Nspins] = S.Mc[:]
            k += S.Nspins
        return state
    def species2spins(self, i):
        """
        given species i, return list of index of associated spin states
        """
        # starts at self.state_dic[i] and contains self.spinSysList[i].Nspins spins
        return (self.state_dic[i]+j for j in range(self.spinSysList[i].Nspins) )
    def spin2species(self, i):
        """
        given spin i, returns index of associated species
        """
        return self.dic_state[i]
    def spinconc(self, i):
        """computes concentration of species bearing spin i"""
        return self.conc[self.spin2species(i)]
    def set_spinflux(self, Ma, Sa, Mb, Sb):
        """
        setup up spin flux
        tells that spin Sa of species Ma, goes into spin Sb of molecule Mb during a chemical equilibrium
        info is stored in the SpinFlux list
        allways symetric (ma,sa,mb,sb) implies (mb,sb,ma,sa)
        """
        ia = self.state_dic[Ma]+Sa
        ib = self.state_dic[Mb]+Sb
        self.spinflux[ia,ib] = 1.0
        self.spinflux[ib,ia] = 1.0
    def saturate(self, Sys, spin, initialized=True):
        """
        tells that spin spin of spin system Sys is saturated
        if initialized==True, magnetisation is set to 0.0 at once, otherwise it is untouched
        """
        # is stored as an index in developped spinstate list
        self.saturated.append(self.state_dic[Sys]+spin)
        if initialized:
            self.set_magnetization(0.0, Sys, spin)
    def report(self):
        """prints a report"""
        super(NMR, self).report()
        print("""
Spin Systems
============""")
        for s in self.spinSysList:
            s.report()
        print("Total of %d spins"%self.Nspins)
        if self.saturated: print("the following spins are saturated :", self.saturated)
        print("spin fluxes:\n", self.spinflux)
    def showmagn(self):
        "show magnetisation"
        for i in range(self.Nspecies):
            l =  "%s :"%self.Spnames[i]
            for m in range(self.spinSysList[i].Nspins):
                l = l + "\t%f"%self.spinSysList[i].Mc[m]
            print(l)
       
    def _evol(self, t, conc):
        "wrapper around evol"
        return self.evol(conc, t)
    def evol(self, state, t=0):
        """
        compute evolution derivative
        state is a developped list of [species concentration] + [spin states]

        uses Clore et Gronenborn. Theory and applications of the transferred nuclear Overhauser effect to the study of the conformations of small ligands bound to proteins. Journal of Magnetic Resonance (1969) (1982) vol. 48 (3) pp. 402-417
        equation 5-7 of this paper are taken with the following conventions:
            M = conc x magn  = c x m
            where conc is species concentration and
                  magn is a scalar ranging from 0 to Mo
        """
        # dMi/dt =  -r (Mi-Mio) + s (M'i-M'io) + k'-1 Mj - k1 Mi  (Clore eqs 5-8)
        # 
        # let on kinetic :   destroyed -k-> created   :    di -k-> cj
        # let k1 = k \prod_{i in destroyed} [di]
        # ( assume no relaxation for the moment (r = s = 0) )
        # 
        # if l in destroyed
        #     dMl/dt =  - k1 ml
        # if l in created
        #     dMl/dt =  + k1 mj  / where j is the spins which experience the exchange
        # 
        # with dM/dt = c dm/dt + m dc/dt
        # and assuming dc/dt = 0      dM/dt = c dm/dt
        # 
        # then it becomes
        # if l in destroyed
        #     dml/dt =  - k1 ml / [dl]
        # if l in created
        #     dml/dt =  + k1 mj / [cl]
        # 
        #      ci dmi/dt =  -r ci (mi-mio) + s ci (m'i-m'io) + k-1cjmj - k1cimi
        #      dmi/dt =  -r (mi-mio) + s (m'i-m'io) + k-1(cj/ci) mj - k1 mi
        #      
        # AEQS - QED
        #      same as dm/dt = 1/c ( dM/dt - m dc/dt)  with dc/dt = 0
        

        Dprint( "state : ", state)
        ############ collect values
        conc = state[:self.Nspecies]
        magn = state[self.Nspecies:]
        ############# concentrations evolution
        dconc = super(NMR, self).evol(conc)

        ############# magnetisation - 2 steps
        dmagn = np.zeros(self.Nspins)
        ############# this one is simply the Solomon equation, using relax matrices 
        for i in range(self.Nspecies):
            S = self.spinSysList[i]
            to = self.state_dic[i]
            tosz = self.spinSysList[i].Nspins
            mg = np.matrix(magn[to:to+tosz])     # magnetisation vector of spin system i
#            dmg = -(mg-S.Mo)*S.Relax
            dmg = S.deriv(mg)
            dmagn[to:to+tosz] = dmg[:]
        Dprint ("dmagn R", dmagn)
        ############# the second one is chemical exchange
        dmagnK = np.zeros(self.Nspins)
        for flux in self.kinlist:
# eg of one flux
#   E/0 + L/0 -k-> EL/01            destroyed = (E,L)   created=(EL)
# another one
#   EL/01     -k-> L/0 + E/0        destroyed = (EL)   created=(E, L)
#
            k = flux.keff   # should already be computed by Eq.evol() 
            Dprint( k )
            for (sp, st) in flux.destroyed:     # compute derivatives
                for l in self.species2spins(sp):
                    rate = k*magn[l] #/conc[sp]
                    dmagnK[l] -= rate      # all state are created/destroyed in proportion of their own intensity
                    Dprint( "%d -dest->       %f  %f %f"%(l, -k, magn[l], -rate ))
            for (sp, st) in flux.created:
                cto = conc[sp]   # concentration of the created species
                to = self.state_dic[sp]
                tosz = self.spinSysList[sp].Nspins
#                    frm = self.state_dic[ds]
#                    frmsz = self.spinSysList[ds].Nspins
                frm = 0
                frmsz = self.Nspins
                submatrix = self.spinflux[ frm:frm+frmsz, to:to+tosz  ]  # where spin fluxes for both species are
                Dprint( submatrix )
                for (j, i) in np.argwhere(submatrix!=0):
                    cj = conc[self.spin2species(frm+j)]
                    coef = 1 #/cj #cj/(cj+cto);   # print coef,
#                    coef = cj/cto;    print coef,
                    rate = k*coef*magn[frm+j]
                    dmagnK[to+i] += rate
                    Dprint ( "%d --creat-> %d     %f %f %f "%(frm+j, to+i, k, coef, rate ))
        ############# apply saturations
        Dprint ("dmagn K", dmagnK)
        for k in self.saturated:
#            dmagnK[k] += -1000*magn[k]       # down to zero if saturated
            dmagn[k] = 0
#            magn[k] = 0
        ############ collect and report
        Dprint ("dmagn satu", dmagnK)
        dstate = np.zeros_like(state)
        dstate[:self.Nspecies] = dconc[:]
        dstate[self.Nspecies:] = dmagnK[:] + dmagn[:]
        if Debug:
            print("dstate", dstate)
        return dstate

    def solve(self, maxtime=1E-3, step=2, trajectory=False, time=None, verbose=False):
        """
        compute evolution of equilibrium system
        maxtime : the duration of integration
        step : the number of steps
        trajectory : if true, returns all step, otherwise return the last point
        """
        state =  np.zeros(self.Nspecies + self.Nspins)      # concatenate conc and state
        state[:self.Nspecies] = self.get_concentration_array()        # set conc
        state[self.Nspecies:] = self.get_magnetization_array()        # set magn
        Dprint( "initial state",state)

        if self.Integrator == "ode":        # True means integrate.ode False means integrate.odeint
            t = []
            y = []
            r = integrate.ode(self._evol)
            r.set_integrator('dop853', nsteps=100, first_step=1E-8)   #method='Adams'
            r.set_initial_value(state, 0.0)
            dt = maxtime/step
            while r.t < maxtime:
                r.integrate(r.t+dt)
                ttarget = r.t+dt
                while not r.successful() and r.t < ttarget: 
                    print("redoing at %f / %f"%(r.t, ttarget))
                    self.conc = r.y[0:self.Nspecies]
                    if verbose: self.showmagn()
                    self.set_magnetization_array(r.y[self.Nspecies:])
                    r.integrate(r.t+dt)
                Dprint( "ok", r.t)
                if trajectory:
                    t.append(r.t)
                y.append(r.y)
            print("\nt final",r.t, r.successful())
        elif self.Integrator == "odeint":
            if time is not None:
                t = time
            else:
                if step>1:
                    t = np.linspace(0, maxtime, step)
                else:
                    t = [maxtime]
            y = integrate.odeint(self.evol, state, t, rtol=Precision, atol=Precision)
        else:
            raise Exception("wrong integrator")
        self.set_concentration_array(y[-1,:self.Nspecies])        # set conc
        self.set_magnetization_array(y[-1,self.Nspecies:])        # set magn
        
        if verbose: print(t); print(y)
        if trajectory:
            return (t, y)
        else:
            return y[len(y)-1]
            
#######################################################
hbar = 1.054*1e-34
mu0 = 1E-7 # mu0/4*Pi
gamma_H = 2.675222005E8 # Rad.s-1.T-1
gamma_N = -2.71261804E7 # Rad.s-1.T-1
gamma_C = 6.728284E7 # Rad.s-1.T-1
gamma_P = 10.8394E7 # Rad.s-1.T-1
class Rates():
    """ 
    A class that computes relaxation rates 
    """
    def __init__(self, field = 11.4, tauc = 1e-9, dist = 3.0E-10, teta = 0):
        """
        field   Bo in Tesla
        tauc    Correlation time in s
        dist    distance in meter
        teta    Angle between the B1 field and B0 during ROESY mixing time (0 for NOESY) - in radian
        """
        self.field = field
        self.tauc = tauc
        self.dist = dist
        self.teta = teta

    def J(self,w,tc=None):
        """
        returns J(w tc) for w and tc
        w or tc can be an array, 
        """
        if tc is None : tc=self.tauc
        # if type(tc) == 'list':
        #     tc = np.array(tc)
        return tc/(1+w*w*tc*tc)

    def calc_sigma(self,r=None,tc=None,teta=None,g1=None):
        """
        Computes the homonuclear cross-relaxation rate in a tilted frame (ROESY or NOESY)
        Return a tuple containing cross_relaxation and auto-relaxation rates
        none parameters defaults to internal
        """
        if r is None : r = self.dist
        if tc is None : tc = self.tauc
        if teta is None : teta = self.teta
        if g1 is None : g1 = gamma_H

        J=self.J

        K = mu0*hbar*g1*g1*(r**(-3)) # Dipolar constant
        w1 = g1*self.field
        K2 = K**2
        sigma = 0.1*K2*(6*J(2*w1,tc)-tc)
        mu = 0.1*K2*(2*tc+3*J(w1,tc)) # Check this expression from M. Goldmann 1988
        rho = 0.1*K2*(tc+3*J(w1,tc)+6*J(2*w1,tc))
        lam = 0.05*K2*(5*tc+9*J(w1,tc)+6*J(2*w1,tc)) # Longitudinal relaxation in B1 field
        c2 = np.cos(teta)**2
        s2 = np.sin(teta)**2
        sig = c2*sigma+s2*mu
        r1 = c2*rho+s2*lam
        return (sig,r1)

    def daragan(self, nres=100, temp=298):
       """ Calculate a tauc value from the number of residues of a protein and the temperature 
       according an interpolate formula proposed by Daragan 
       temperature should be in Kelvin
       returns a tauc in ns """
       return 1E-9*np.exp(2416/temp)*nres**(0.93)*0.00918/temp    
##################################################
# tests
##################################################
import unittest
class Tests(unittest.TestCase):
    """ tests """
    def setUp(self):
        self.verbose = 1    # verbose >0 switches on messages
    def announce(self):
        if self.verbose >0:
            print(self.shortDescription())
    def test_eq(self):
        """ Test association equilibrium
        This one simulate a one site binding :
        L + E   <->  EL
        3 species : L  E  EL
                    0  1  2
        """
        self.announce()
        Keq=1/45E-6     # Ka is inverse of Kd
        Ltot=1E-3
        Etot=100E-6
        ##### describe the scene
        eq1 = Eq(3)     # we define a 3 species system
        # it is convenient to define species
        L=0; E=1; EL=2      # we give them symbolic names
        eq1.Spnames[E] = "E"    # and names in the system
        eq1.Spnames[L] = "L"
        eq1.Spnames[EL] = "EL"
        eq1.set_K3(Keq, L, E, EL)       # set a reversible equilibirum, with Keq
        eq1.set_massconserv(Ltot, [1,0,1])  # tell what is transformed in what, in L
        eq1.set_massconserv(Etot, [0,1,1])  # and in E
        eq1.set_concentration(Ltot, L)               # set the total concentration of E and L
        eq1.set_concentration(Etot, E)
        #####
        eq1.solve(step=10)        # solve the equilibrium, by default equilibrium time is 1E-3 second
        cf = eq1.get_concentration_array()
        #       eq1.showconc()
        #       print "[complex] / [Etotal] = %f %%"%(100*cf[EL]/(cf[E]+cf[EL]))
        self.assertAlmostEqual(cf[E]+cf[EL], Etot)
        self.assertAlmostEqual(cf[L]+cf[EL], Ltot)
        self.assertAlmostEqual( cf[EL]/(cf[E]+cf[EL]), 0.9526, 4)
    def test_noe(self):
        """ testing NOE
        NOE in a 2 coupled spins system
        one spin is saturated
        """
        self.announce()
        # set-up equilibrium
        eq1 = NMR(1,2)
        #  add spin systems
        R1E = 1.0
        sigma = -0.2
        Relax = np.matrix([[R1E,sigma],[sigma,R1E]])
        eq1.set_spin(0, SpinSys(2,Relax))
        eq1.saturate(0, 0)
#        eq1.report()
        Dprint('')
        eq1.solve(maxtime=10)    # maxtime=1E-4, step=100,verbose=True)
        transfer = eq1.get_magnetization(0,1)
        Dprint( "sigma/rho:", sigma/R1E, "noe : ", transfer-1.0)
        #sigma/R1E is maximum possible NOE
        #cf[2] is final magnetisation 
        self.assertAlmostEqual(1+sigma/R1E, transfer, 4 ) # 4 digit accuracy

##################################################
if __name__ == '__main__':
    unittest.main()
