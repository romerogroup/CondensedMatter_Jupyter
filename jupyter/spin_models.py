#!/usr/bin/env python

from ase.atoms import Atoms
from minimulti.spin.hamiltonian import SpinHamiltonian
from minimulti.spin.mover import SpinMover
from minimulti.spin.qsolver import QSolver

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
from minimulti.constants import mu_B, eV
meV = eV * 1e-3


def plot_3d_vector(positions, vectors, length=0.1):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax=fig.gca()
    n = positions.shape[1]
    x, y, z = positions.T
    u, v, w = vectors.T
    ax.quiver(
        x,
        y,
        z,
        u,
        v,
        w,
        length=length,
        normalize=False,
        pivot='middle',
        cmap='seismic')
    plt.show()


def plot_2d_vector(positions, vectors, show_z=True, length=0.1, ylimit=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n = positions.shape[1]
    x, y, z = positions.T
    u, v, w = vectors.T
    #ax.streamplot(x, y, u, v,  linewidth=1, cmap=plt.cm.inferno,
    #    density=2, arrowstyle='->', arrowsize=1.5)
    ax.scatter(x, y, s=50, color='r')
    if show_z:
        ax.quiver(
            x,
            y,
            u,
            v,
            w,
            length=length,
            units='width',
            pivot='middle',
            cmap='seismic',
        )
    else:
        ax.quiver(x, y, u, v, units='width', pivot='middle', cmap='seismic')
    if ylimit is not None:
        ax.set_ylim(ylimit[0], ylimit[1])
    #plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def plot_supercell(ham,
                   supercell_matrix=np.diag([30, 1, 1]),
                   plot_type='2d',
                   length=0.1,
                   ylimit=None):
    sc_ham = ham.make_supercell(supercell_matrix)
    sc_ham.s = np.random.rand(*sc_ham.s.shape) - 0.5
    mover = SpinMover(hamiltonian=sc_ham)
    mover.set(time_step=3e-4, temperature=0, total_time=6, save_all_spin=False)

    mover.run(write_step=20)
    pos = np.dot(sc_ham.pos, sc_ham.cell)
    if plot_type == '2d':
        plot_2d_vector(
            pos, mover.s, show_z=False, length=length, ylimit=ylimit)
    elif plot_type == '3d':
        plot_3d_vector(pos, mover.s, length=length)


#exchange_1d()


def plot_spinwave(ham,
                  qnames=['$\Gamma$', 'X'],
                  qvectors=[(0, 0, 0), (0.5, 0, 0)]):
    fig = plt.figure()
    from ase.build import bulk
    from ase.dft.kpoints import get_special_points
    from ase.dft.kpoints import bandpath
    kpts, x, X = bandpath(qvectors, ham.cell, 300)
    qsolver = QSolver(hamiltonian=ham)
    evals, evecs = qsolver.solve_all(kpts, eigen_vectors=True)
    nbands = evals.shape[1]
    for i in range(nbands):
        plt.plot(x, evals[:, i] / 1.6e-21)
    plt.xlabel('Q-point (2$\pi$)')
    plt.ylabel('Energy (meV)')
    plt.xlim(x[0], x[-1])
    plt.xticks(X, qnames)
    for x in X:
        plt.axvline(x, linewidth=0.6, color='gray')
    plt.show()


def plot_M_vs_time(ham, supercell_matrix=np.eye(3), temperature=0):
    sc_ham = ham.make_supercell(supercell_matrix)
    mover = SpinMover(hamiltonian=sc_ham)
    mover.set(
        time_step=1e-5,
        temperature=temperature,
        total_time=1,
        save_all_spin=True)

    mover.run(write_step=10)

    hist = mover.get_hist()
    hspin = np.array(hist['spin'])
    time = np.array(hist['time'])
    tspin = np.array(hist['total_spin'])

    Ms = np.linalg.det(supercell_matrix)
    plt.figure()
    plt.plot(
        time, np.linalg.norm(tspin, axis=1) / Ms, label='total', color='black')
    plt.plot(time, tspin[:, 0] / Ms, label='x')
    plt.plot(time, tspin[:, 1] / Ms, label='y')
    plt.plot(time, tspin[:, 2] / Ms, label='z')
    plt.xlabel('time (s)')
    plt.ylabel('magnetic moment ($\mu_B$)')
    plt.legend()
    #plt.show()
    #avg_total_m = np.average((np.linalg.norm(tspin, axis=1)/Ms)[:])
    plt.show()


def plot_M_vs_T(ham, supercell_matrix=np.eye(3), Tlist=np.arange(0.0, 110,
                                                                 20)):
    Mlist = []
    for temperature in Tlist:
        sc_ham = ham.make_supercell(supercell_matrix)
        mover = SpinMover(hamiltonian=sc_ham)
        mover.set(
            time_step=2e-4,
            #damping_factor=0.1,
            temperature=temperature,
            total_time=1,
            save_all_spin=True)

        mover.run(write_step=10)

        hist = mover.get_hist()
        hspin = np.array(hist['spin'])
        time = np.array(hist['time'])
        tspin = np.array(hist['total_spin'])

        Ms = np.linalg.det(supercell_matrix)
        avg_total_m = np.average((np.linalg.norm(tspin, axis=1) / Ms)[300:])
        print("T: %s   M: %s" % (temperature, avg_total_m))
        Mlist.append(avg_total_m)

    plt.plot(Tlist, Mlist)
    plt.ylim(-0.01, 1.01)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Average magnetization (Ms)')
    plt.show()


def exchange_1d_hamiltonian(J1=3e-21,
                            J2=0e-21,
                            DMI=[0, 0, 0e-21],
                            k1=np.array([-0 * mu_B]),
                            k1dir=np.array([[0.0, 0.0, 1.0]]),
                            plot_type='2d'):
    # make model
    atoms = Atoms(symbols="H", positions=[[0, 0, 0]], cell=[1, 1, 1])
    spin = np.array([[0, 1, 0]])

    ham = SpinHamiltonian(
        cell=atoms.cell,
        pos=atoms.get_scaled_positions(),
        spinat=spin,
        zion=atoms.get_atomic_numbers())
    ham.gilbert_damping = [0.8]
    #ham.gyro_ratio=[1.0]

    J = {
        (0, 0, (1, 0, 0)): J1,
        (0, 0, (-1, 0, 0)): J1,
        (0, 0, (2, 0, 0)): J2,
        (0, 0, (-2, 0, 0)): J2,
    }
    ham.set_exchange_ijR(exchange_Jdict=J)

    k1 = k1
    k1dir = k1dir
    ham.set_uniaxial_mca(k1, k1dir)

    DMIval = np.array(DMI)
    DMI = {
        #(0, 0, (0, 0, 1)): DMIval,
        #(0, 0, (0, 0, -1)): -DMIval,
        #(0, 0, (0, 1, 0)): DMIval,
        #(0, 0, (0, -1, 0)): -DMIval,
        (0, 0, (-1, 0, 0)):
        -DMIval,
        (0, 0, (1, 0, 0)):
        DMIval,
    }
    ham.set_dmi_ijR(dmi_ddict=DMI)

    return ham


def canting_1d_hamiltonian(
        J1=3e-21,
        #J2=0e-21,
        DMI1=[0, 0, 0e-21],
        DMI2=[0, 0, -0e-21],
        k1=np.array([-0 * mu_B]),
        k1dir=np.array([[0.0, 0.0, 1.0]]),
        plot_type='2d'):
    # make model
    atoms = Atoms(symbols="H", positions=[[0, 0, 0]], cell=[1, 1, 1])
    spin = np.array([[0, 1, 0]])

    ham = SpinHamiltonian(
        cell=atoms.cell,
        pos=atoms.get_scaled_positions(),
        spinat=spin,
        zion=atoms.get_atomic_numbers())
    ham.gilbert_damping = [0.8]
    #ham.gyro_ratio=[1.0]

    J = {
        (0, 0, (1, 0, 0)): J1,
        (0, 0, (-1, 0, 0)): J1,
        #(0, 0, (2, 0, 0)): J2,
        #(0, 0, (-2, 0, 0)): J2,
    }
    ham.set_exchange_ijR(exchange_Jdict=J)

    k1 = k1
    k1dir = k1dir
    ham.set_uniaxial_mca(k1, k1dir)
    sc_ham = ham.make_supercell(np.diag([2, 1, 1]))

    DMI1val = np.array(DMI1)
    DMI2val = np.array(DMI2)
    DMI = {
        (0, 1, (0, 0, 0)): DMI1val,
        (1, 0, (0, 0, 0)): -DMI1val,
        (0, 1, (-1, 0, 0)): -DMI2val,
        (1, 0, (1, 0, 0)): DMI2val,
    }
    sc_ham.set_dmi_ijR(dmi_ddict=DMI)

    return sc_ham


def square_2d_hamiltonian(Jx=-5e-21,
                          Jy=-5e-21,
                          k1=np.array([-000 * mu_B]),
                          k1dir=np.array([[0.0, 0, 1.0]])):
    atoms = Atoms(
        symbols="H",
        positions=[[0, 0, 0]],
        cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    spin = np.array([[0, 1, 0]])

    ham = SpinHamiltonian(
        cell=atoms.cell,
        pos=atoms.get_scaled_positions(),
        spinat=spin,
        zion=atoms.get_atomic_numbers())
    ham.gilbert_damping = [1.0]
    J = {
        (0, 0, (1, 0, 0)): Jx,
        (0, 0, (-1, 0, 0)): Jx,
        (0, 0, (0, 1, 0)): Jy,
        (0, 0, (0, -1, 0)): Jy,
    }
    ham.set_exchange_ijR(exchange_Jdict=J)
    ham.set_uniaxial_mca(k1, k1dir)
    return ham


def traingular_2d_hamiltonian(J1=-0e-21,
                              k1=np.array([-000 * mu_B]),
                              k1dir=np.array([[0.0, 0, 1.0]])):
    """
    Isolated spin in an applied field. field:10T, time step: 1fs.
    Total time: 100 ps, No Langevin term.
    """
    # make model
    atoms = Atoms(
        symbols="H",
        positions=[[0, 0, 0]],
        cell=[[1, 0, 0], [-1.0 / 2, np.sqrt(3) / 2, 0], [0, 0, 1]])
    spin = np.array([[0, 1, 0]])

    ham = SpinHamiltonian(
        cell=atoms.cell,
        pos=atoms.get_scaled_positions(),
        spinat=spin,
        zion=atoms.get_atomic_numbers())
    ham.gilbert_damping = [1.0]
    J = {
        (0, 0, (1, 0, 0)): J1,
        (0, 0, (0, 1, 0)): J1,
        (0, 0, (1, 1, 0)): J1,
        (0, 0, (-1, 0, 0)): J1,
        (0, 0, (0, -1, 0)): J1,
        (0, 0, (-1, -1, 0)): J1,
    }
    ham.set_exchange_ijR(exchange_Jdict=J)
    ham.set_uniaxial_mca(k1, k1dir)
    return ham


def cubic_3d_hamiltonian(Jx=-0e-21,
                         Jy=-0e-21,
                         Jz=0e-21,
                         DMI=[0, 0, 0e-21],
                         k1=np.array([-0 * mu_B]),
                         k1dir=np.array([[0.0, 0, 1.0]])):
    atoms = Atoms(
        symbols="H",
        positions=[[0, 0, 0]],
        cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    spin = np.array([[0, 1, 0]])

    ham = SpinHamiltonian(
        cell=atoms.cell,
        pos=atoms.get_scaled_positions(),
        spinat=spin,
        zion=atoms.get_atomic_numbers())
    ham.gilbert_damping = [1.0]
    J = {
        (0, 0, (1, 0, 0)): Jx,
        (0, 0, (-1, 0, 0)): Jx,
        (0, 0, (0, 1, 0)): Jy,
        (0, 0, (0, -1, 0)): Jy,
        (0, 0, (0, 0, 1)): Jz,
        (0, 0, (0, 0, -1)): Jz,
    }
    ham.set_exchange_ijR(exchange_Jdict=J)
    ham.set_uniaxial_mca(k1, k1dir)
    DMIval = np.array(DMI)
    DMI = {
        (0, 0, (0, 0, 1)): DMIval,
        (0, 0, (0, 0, -1)): -DMIval,
        #(0, 0, (0, 1, 0)): DMIval,
        #(0, 0, (0, -1, 0)): -DMIval,
        #(0, 0, (-1, 0, 0)): -DMIval,
        #(0, 0, (1, 0, 0)): DMIval,
    }
    ham.set_dmi_ijR(dmi_ddict=DMI)
    return ham


def cubic_3d_2site_hamiltonian(Jx=-0e-21,
                               Jy=-0e-21,
                               Jz=0e-21,
                               DMI=[0, 0, 0e-21],
                               k1=np.array([-0 * mu_B]),
                               k1dir=np.array([[0.0, 0, 1.0]])):
    atoms = Atoms(
        symbols="H",
        positions=[[0, 0, 0]],
        cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    spin = np.array([[0, 1, 0]])

    ham = SpinHamiltonian(
        cell=atoms.cell,
        pos=atoms.get_scaled_positions(),
        spinat=spin,
        zion=atoms.get_atomic_numbers())
    ham.gilbert_damping = [0.8]
    J = {
        (0, 0, (1, 0, 0)): Jx,
        (0, 0, (-1, 0, 0)): Jx,
        (0, 0, (0, 1, 0)): Jy,
        (0, 0, (0, -1, 0)): Jy,
        (0, 0, (0, 0, 1)): Jz,
        (0, 0, (0, 0, -1)): Jz,
    }
    ham.set_exchange_ijR(exchange_Jdict=J)
    ham.set_uniaxial_mca(k1, k1dir)
    sc_ham = ham.make_supercell(np.diag([2, 1, 1]))
    DMIval = np.array(DMI)
    DMI = {
        (0, 1, (0, 0, 0)): DMIval,
        #(1, 0, (0, 0, 0)): -DMIval,
        (0, 1, (-1, 0, 0)): DMIval,
        #(1, 0, (1, 0, 0)): -DMIval,
        #(0, 0, (0, 1, 0)): DMIval,
        #(0, 0, (0, -1, 0)): -DMIval,
        #(0, 0, (-1, 0, 0)): -DMIval,
        #(0, 0, (1, 0, 0)): DMIval,
    }
    sc_ham.set_dmi_ijR(dmi_ddict=DMI)
    return sc_ham
