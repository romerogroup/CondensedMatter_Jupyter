#!/usr/bin/env python
"""
mover
"""
import os
import numpy as np
from minimulti.constants import mu_B, gyromagnetic_ratio, bohr_mag
from minimulti.spin.hist_file import SpinHistFile, read_hist
from minimulti.constants import Boltzmann
from math import sin, cos
import numba


# JIT function to accelerate python code
@numba.njit('float64[:](float64[:],float64[:])', fastmath=True)
def cross(a, b):
    ret = np.empty(shape=3, dtype=numba.float64)
    ret[0] = a[1] * b[2] - a[2] * b[1]
    ret[1] = a[2] * b[0] - a[0] * b[2]
    ret[2] = a[0] * b[1] - a[1] * b[0]
    return ret


@numba.njit(fastmath=True, parallel=False)
def get_ds(nspin, Heff, gamma_L, gilbert_damping, s, dt, ds):
    for i in numba.prange(nspin):
        Ri = cross(s[i, :], Heff[i, :])
        ds[i, :] = -gamma_L[i] * (
            Ri + gilbert_damping[i] * cross(s[i, :], Ri)) * dt


@numba.njit(fastmath=True, parallel=False)
def get_ds_DM(nspin, Heff, gamma_L, gilbert_damping, s, dt, ds):
    for i in numba.prange(nspin):
        Hr = gamma_L[i] * (
            Heff[i, :] + gilbert_damping[i] * cross(s[i, :], Heff[i, :]))
        #Bnorm=np.linalg.norm(Hr)
        Bnorm = np.sqrt(Hr[0] * Hr[0] + Hr[1] * Hr[1] + Hr[2] * Hr[2])
        axis = Hr / Bnorm  # axis
        half_angle = Bnorm * dt * 0.5  # half angle
        r = axis * sin(half_angle)
        w = cos(half_angle)
        si = s[i, :]
        ds[i, :] = 2.0 * cross(r, (cross(r, si) + w * si))


@numba.njit(fastmath=True)
def normalize(S, nspin):
    for i in range(nspin):
        S[i, :] /= np.sqrt(np.sum(S[i, :] * S[i, :]))


class SpinMover(object):
    def __init__(self,
                 hamiltonian,
                 s=None,
                 hist_fname='Spinhist.nc',
                 write_hist=True):
        self.hamiltonian = hamiltonian
        self.nspin = self.hamiltonian.nspin
        if s is None:
            self.s = np.random.random((self.nspin, 3)) - 0.5
            self.normalize_s()
        else:
            self.s = np.array(s, dtype=float)
        self.ms = self.hamiltonian.ms
        self.ds = np.zeros_like(self.s)

        self.tol_dsdt = 1e-5
        self.time_list = []
        self.Hext_list = []
        self.current_time = 0.0

        self.gyro_ratio = hamiltonian.gyro_ratio
        self.gilbert_damping = hamiltonian.gilbert_damping
        self.write_hist = write_hist
        self.hist_fname = hist_fname
        if self.write_hist:
            if os.path.exists(hist_fname):
                os.remove(hist_fname)
            self.histfile = SpinHistFile(fname=hist_fname, nspin=self.nspin)
            self.histfile.write_cell(
                Ms=self.ms,
                spin_positions=self.hamiltonian.pos,
                Rlist=self.hamiltonian.Rlist,
                iprim=self.hamiltonian.iprim)
        # temporaries
        self._ds0 = np.zeros_like(self.s)
        self.H_lang = np.zeros_like(self.s)
        self.Heff = np.zeros_like(self.s)
        self.Heff1 = np.zeros_like(self.s)
        self.Heff2 = np.zeros_like(self.s)

    def set(self,
            time_step=0.1,
            temperature=0,
            total_time=100,
            save_all_spin=False):
        """
        set parameters for simulation:
        args:
        ====================
        timestep: in ps
        temperature: in K, default 0K.
        total_time: total simulation time
        """
        self.dt = time_step * 1e-12  # default in ps
        self.temperature = temperature
        self.total_time = total_time * 1e-12
        self.time_list = np.arange(0.0, self.total_time, self.dt)
        self.save_all_spin = save_all_spin
        self._langevin_tmp = np.sqrt(
            2.0 * self.gilbert_damping * Boltzmann * self.temperature /
            (self.gyro_ratio * self.dt) / self.ms)
        self.gamma_l = self.gyro_ratio / (1.0 + self.gilbert_damping**2)

    def normalize_s(self):
        """
        normalize so the norm of self.S[i,:] is 1
        """
        #snorm = np.linalg.norm(self.s, axis=1)
        #self.s /= np.expand_dims(snorm, axis=1)
        normalize(self.s, self.nspin)

    def calc_langevin_heff(self):
        """
        A effective magnetic field from Langevin term.
        """
        if self.temperature < 1e-07:
            return np.zeros([len(self.ms), 3])
        else:
            self.H_lang = np.random.randn(self.nspin, 3) * np.expand_dims(
                self._langevin_tmp, axis=1)

    #@profile
    def get_ds_HeunP(self, Heff, ds):
        """
        calculate Delta S for a time step delta_t
        """
        get_ds(
            self.nspin,
            Heff=self.Heff,
            gamma_L=self.gamma_l,
            gilbert_damping=self.gilbert_damping,
            s=self.s,
            dt=self.dt,
            ds=ds)

    def get_ds_DM(self, Heff, ds):
        get_ds_DM(
            self.nspin,
            Heff=self.Heff,
            gamma_L=self.gamma_l,
            gilbert_damping=self.gilbert_damping,
            s=self.s,
            dt=self.dt,
            ds=ds)

    def Euler_integration(self):
        # get dsdt
        self.calc_langevin_heff()
        # predict
        self.hamiltonian.get_effective_field(self.s, self.Heff)
        self.Heff /= self.ms[:, None]
        self.Heff += self.H_lang
        self.get_ds_HeunP(self.Heff, self._ds0)
        self.s = self.s + self._ds0
        normalize(self.s, self.nspin)

    def Heun_integration(self):
        self.calc_langevin_heff()
        # predict
        self.hamiltonian.get_effective_field(self.s, self.Heff1)
        self.Heff1 /= self.ms[:, None]
        self.Heff = self.Heff1 + self.H_lang
        self.get_ds_HeunP(self.Heff, self._ds0)
        self._ds0 += self.s
        normalize(self._ds0, self.nspin)

        # correction
        self.hamiltonian.get_effective_field(self._ds0, self.Heff2)
        self.Heff2 /= self.ms[:, None]
        self.Heff = (self.Heff1 + self.Heff2) * 0.5 + self.H_lang
        self.get_ds_DM(self.Heff, self.ds)
        self.s += self.ds
        normalize(self.s, self.nspin)

    def DM_integration(self):
        self.calc_langevin_heff()
        # predict
        self.hamiltonian.get_effective_field(self.s, self.Heff1)
        self.Heff1 /= self.ms[:, None]
        self.Heff = self.Heff1 + self.H_lang
        self.get_ds_DM(self.Heff, self._ds0)
        stmp = self.s + self._ds0

        # correction
        self.hamiltonian.get_effective_field(stmp, self.Heff2)
        self.Heff2 /= self.ms[:, None]
        self.Heff = (self.Heff1 + self.Heff2) * 0.5 + self.H_lang
        self.get_ds_DM(self.Heff, self.ds)
        self.s += self.ds

    def Hybrid_integration(self):
        """
        Use HeunP method as predict and rotation as correction.
        """
        self.calc_langevin_heff()
        # predict
        self.hamiltonian.get_effective_field(self.s, self.Heff1)
        self.Heff1 /= self.ms[:, None]
        self.Heff = self.Heff1 + self.H_lang
        self.get_ds_HeunP(self.Heff, self._ds0)
        self._ds0 += self.s + self._ds0
        normalize(self._ds0, self.nspin)

        # correction
        self.hamiltonian.get_effective_field(self._ds0, self.Heff2)
        self.Heff2 /= self.ms[:, None]
        self.Heff = (self.Heff1 + self.Heff2) * 0.5 + self.H_lang
        self.get_ds_DM(self.Heff, self.ds)
        self.s += self.ds

    def run_one_step(self,
                     method='DM',
                     write_hist=True,
                     time=None,
                     itime=None):
        """
        one time step
        """
        if method == 'HeunP':
            self.Heun_integration()
        elif method == 'DM':
            self.DM_integration()
        elif method == 'Euler':
            self.Euler_integration()
        elif method == 'Hybrid':
            self.Hybrid_integration()
        else:
            raise ValueError("method should be  HeunP, DM or Euler")
        self.current_time = self.current_time + self.dt
        self.current_s = self.s
        if write_hist:
            self.histfile.write_S(S=self.s, time=self.current_time, itime=itime)

    def run(self, write_step=1, method='HeunP'):
        """
        all time step.
        """
        if self.write_hist:
            self.histfile.write_S(S=self.s, time=0.0, itime=0)
        for i, time in enumerate(self.time_list[1:]):
            self.hamiltonian.set_Hext_at_time(time)

            if self.write_hist and i % write_step == 0:
                self.run_one_step(
                    method=method, write_hist=True, time=time, itime=i)
            else:
                self.run_one_step(
                    method=method, write_hist=False, time=time, itime=i)


    def set_MH_signal(self, Hext_list):
        """
        set external H for MH simmulation.
        Hext_list: a list of external Hfield
        """
        self.Hext_list = Hext_list

    def run_MH(self, Hext_list=None, print_S=True):
        """
        Run MH simulatioin
        """
        if Hext_list is not None:
            self.Hext_list = Hext_list
        S_list = []
        for Hext in self.Hext_list:
            self.hamiltonian.set_external_hfield(Hext)
            self.run()
            S = self.current_s
            S_list.append(S)
            if print_S:
                print("H Field: ", Hext)
                print("Current Spin orientation: \n", S)
                print('\n')
        return S_list

    def get_total_spin(self, abs_val=False):
        self.total_spin = np.sum(self.s * self.ms[:, None] / bohr_mag, axis=0)
        return self.total_spin

    def get_total_spin_abs(self, abs_val=False):
        self.total_spin_abs = np.sum(
            np.abs(self.s) * self.ms[:, None] / bohr_mag, axis=0)
        return self.total_spin_abs

    def get_hist(self):
        if not self.write_hist:
            raise IOError(
                "Hist not saved to histfile. Use write_hist=True to save it.")
        return read_hist(self.hist_fname)

    def close_hist(self):
        self.histfile.close()

    def __del__(self):
        if self.write_hist:
            self.histfile.close()
