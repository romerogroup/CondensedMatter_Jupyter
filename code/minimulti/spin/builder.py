from ase.atoms import Atoms
import numpy as np
from minimulti.spin.hamiltonian import SpinHamiltonian
from minimulti.constants import mu_B, eV


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
                          dmi=5e-21,
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
    DMI = {
        (0, 0, (1, 0, 0)): [0, -dmi, 0],
        (0, 0, (-1, 0, 0)): [0,dmi, 0],
        (0, 0, (0, 1, 0)): [dmi,0, 0 ],
        (0, 0, (0, -1, 0)): [-dmi, 0, 0],
    }
    ham.set_dmi_ijR(dmi_ddict=DMI)
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
