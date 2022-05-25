<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

---

Implementation of **PVR for Control**, as presented
in [The (Un)Surprising Effectiveness of Pre-Trained Vision Models for Control](https://arxiv.org/abs/2203.03580).

<ins>The `main` branch reproduces the results presented in the paper.  
The `distributed` branch uses multiprocessing and distributed training for much
faster learning. It also has wrappers for other environments, including Adroit and DMC.
Results are slightly different, but we encourage to use it if you would like to
build upon our paper.</ins>

## Codebase Installation
```
conda create -n pvr_habitat python=3.8
conda activate pvr_habitat
git clone git@github.com:sparisi/pvr_habitat.git
cd pvr_habitat
pip install -r requirements.txt
```

## Habitat Installation
* Clone `https://github.com/sparisi/habitat-lab` and do a **full install** with `habitat_baselines`.
> The main differences between this and the original Habitat repository are:  
> 1) `STOP` action removed,  
> 2) Bugfix where the agent is placed slightly above the ground, and therefore the
terminal goal condition is never triggered.

* Download and extract Replica scenes in the root folder of `pvr_habitat`.
> WARNING! The dataset is very large!

```
sudo apt-get install pigz
git clone https://github.com/facebookresearch/Replica-Dataset.git
cd Replica-Dataset
./download.sh replica-path
```

If you have already downloaded it somewhere else, just make a symbolic link
```
ln -s path/to/Replica-Dataset Replica-Dataset
```

## How to Run Experiments
There are three main scripts to run behavioral cloning:
* `main_bc_1.py` loads raw trajectories saved as pickles, passes observations (images)
through the embedding, and then learns on the embedded observations.
* `main_bc_2.py` directly loads embedded observations that have already been passed
through the embedding, in order to save time.
* `main_bc_finetune.py` is used to finetune the random PVR.

For more details on how to generate trajectories and pickles, see the README in
the `behavioral_cloning` folder.

Pre-trained models can be downloaded [here](https://github.com/sparisi/pvr_habitat/releases/tag/models).
