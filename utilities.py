from zse.collections import framework
from zse.rings import get_osites
from zse.rings import get_tsites
import numpy as np

def label_osites(atoms, code):

    z = framework(code)
    osites,omult,first = get_osites(code)
    oinds = [atom.index for atom in atoms if atom.symbol=='O']

    zcell = z.get_cell_lengths_and_angles()[:3]
    acell = atoms.get_cell_lengths_and_angles()[:3]
    repeat=[]
    for zc, ac in zip(zcell,acell):
        repeat.append(int(round(ac/zc)))

    rp = np.prod(repeat)
    Dict = {}
    j=0
    for i in range(rp):
        for s,t in enumerate(osites):
            for q in range(omult[s]):
                Dict[oinds[j]]=t
                j+=1

    return Dict

def label_tsites(atoms, code):

    z = framework(code)
    tsites,tmult,first = get_tsites(code)
    tinds = [atom.index for atom in atoms if atom.symbol!='O']

    zcell = z.get_cell_lengths_and_angles()[:3]
    acell = atoms.get_cell_lengths_and_angles()[:3]
    repeat=[]
    for zc, ac in zip(zcell,acell):
        repeat.append(int(round(ac/zc)))

    rp = np.prod(repeat)
    Dict = {}
    j=0
    for i in range(rp):
        for s,t in enumerate(tsites):
            for q in range(tmult[s]):
                Dict[tinds[j]]=t
                j+=1

    return Dict

def site_labels(atoms,code):

    tdict = label_tsites(atoms,code)
    odict = label_osites(aotms,code)
    all_labels = {**tdict,**odict}