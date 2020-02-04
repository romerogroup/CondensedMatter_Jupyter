# Minimulti

## Installation
```
python setup.py install --user
or 
pip install . --user
```
or in developer mode:
```
python setup.py develop --user
or
pip install -e . --user
```

To run the jupyter notebooks in the example directory, ipywidget and ipyvolume need to be installed and the jupyter extension need to be enabled.

```
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

```
pip install ipyvolume 
jupyter nbextension enable --py ipyvolume
jupyter nbextension enable --py widgetsnbextension
```



## Current status:

#### Lattice part:

- tested.

- only harmonic part implemented.

- can use the functionalities implemented in ase and phonopy.

####     Spin part: partially tested.

- need to implement post processing tools.

####     Spin lattice coupling:

- implemented, can run.

- no test yet, so should have tons of bugs.

- coupled dynamics quite primitive. (can only run one step for each)

####    post processing tools:

- unfolding of phonon/magnon band structure.

#### documentation:

- Jupyter notebook in notebook directory

-
