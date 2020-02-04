#!/usr/bin/env python
"""
functions to modify tight binding models
"""

def shift_onsite(model, value, func):
    """
    shift_onsite
    func: A function to
    """
    onsite = model.get_onsite()
    bset = model.bset

    for i, b in enumerate(bset):
        if func(b, bset):
            onsite[i] += value

    model.set_onsite(onsite)
    return model


def shift_species_onsite(model, value, symbol):
    """
    shift_species_onsite
    """
    onsite = model.get_onsite()
    bset = model.bset
    symbols = bset.atoms.get_chemical_symbols()
    print(symbols)
    for i, b in enumerate(bset):
        if symbols[b.site] == symbol:
            onsite[i] += value
    model.set_onsite(onsite, mode='reset')
    return model

def shift_species_spin_split(model, value, symbol):
    """
    shift_species_spin splitting
    """
    onsite = model.get_onsite()
    bset = model.bset
    symbols = bset.atoms.get_chemical_symbols()
    for i, b in enumerate(bset):
        if symbols[b.site] == symbol:
            if b.spin==0:
                onsite[i] += value
            elif b.spin ==1:
                onsite[i] += value
    model.set_onsite(onsite, mode='reset')
    return model

