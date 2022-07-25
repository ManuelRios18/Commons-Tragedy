## Commons tragedy

## Installation guide

### Conda environment setup

We need a python 3.9.0 conda environment

```
conda create -n meltingpot_env python==3.9.0
conda activate meltingpot_env
```

From HOME directory clone meltingpot repository, given that currently this code does not work 
with the latest meltingpot release, we need to commit back to a previous version. To do so run:

```
git clone https://github.com/deepmind/meltingpot.git
cd meltingpot
git reset --hard eee0e6ecbff809b6abf0a354d4b519d4d10a1e14
```

Then install meltingpot as usual by running `./install.sh`
Do not forget to add the meltigpot to the basrhrc directory by running `
sudo nano ~/.bashrc` and then append to the end of the file the following `export PYTHONPATH=/home/manuel/meltingpot`.  
Finally, install the rey dependencies by running `pip3 install -e .[rllib]` on meltingpot directory.   
Close all terminal and test meltingpot:

``` 
 cd ~/meltingpot
 python3 meltingpot/python/human_players/play_clean_up.py
```

Then clone this repository on any desired directory

```
git clone https://github.com/ManuelRios18/Commons-Tragedy.git
```

Now, download dreamerv2 from my fork:

```
git clone https://github.com/ManuelRios18/dreamerv2.git
```

Then install dreamerv2 package on your environment, from  its root directory run:

```commandline
pip install -e .
```

If you get the error `'numpy.random._generator.Generator' object has no attribute 'randint'`
chage randint to integers in the highlighted file