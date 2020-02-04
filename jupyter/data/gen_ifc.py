#!/usr/bin/env python
from abipy.abilab import abiopen

def gen_ifc(mag):
    ddbfile="%s.DDB"%mag
    ddb=abiopen(ddbfile)
    ddb.anaget_ifc(workdir='%s_IFC'%mag, symdynmat=0)

for mag in [
    'FM',
    'G',
    'A',
    'C']:
    print gen_ifc(mag)
