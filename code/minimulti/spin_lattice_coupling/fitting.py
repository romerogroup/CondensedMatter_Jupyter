"""
Fit coefficients.
"""

from minimulti.abstract.datatype import IFClike, Rijv


def fit_cubic_J(E_FM, E_G, E_A, E_C):
    E_0 = sum(E_FM, E_G, E_A, E_C) / 4.0
    J = (E_G - E_FM) / 24.0 + (E_A - E_C) / 8.0

    J_dict = {
        (0, 0, (0, 0, -1)): J,
        (0, 0, (0, 0, 1)): J,
        (0, 0, (0, -1, 0)): J,
        (0, 0, (0, 1, 0)): J,
        (0, 0, (-1, 0, 0)): J,
        (0, 0, (1, 0, 0)): J,
    }
    return E_0, J, J_dict


def fit_cubic_T(IFC_FM, IFC_G, IFC_A, IFC_C):
    IFC_0 = (IFC_FM + IFC_G + IFC_A + IFC_C) / 4.0
    J2 = (IFC_G - IFC_FM) / 24.0 + (IFC_A - IFC_C) / 8.0

    J2_dict = Rijv(shape=J2.shape, dtype=J2.dtype)
    J2_dict.update({
        (0, 0, (0, 0, -1)): J2,
        (0, 0, (0, 0, 1)): J2,
        (0, 0, (0, -1, 0)): J2,
        (0, 0, (0, 1, 0)): J2,
        (0, 0, (-1, 0, 0)): J2,
        (0, 0, (1, 0, 0)): J2,
    })
    return IFC_0, J2, J2_dict*(-0.5)
