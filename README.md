<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

---

Implementation of **PVR for Control**, as presented
in [The (Un)Surprising Effectiveness of Pre-Trained Vision Models for Control](https://arxiv.org/abs/2203.03580).

```
conda create -n pvr python=3.8
conda activate pvr
git clone git@github.com:sparisi/pvr.git
cd pvr
pip install -r requirements.txt
```

## Supported Embedding Models
- **ResNet**: already installed.
- **MoCo**: download model from [here](https://github.com/facebookresearch/moco).
- **CLIP**: run `pip install git+https://github.com/openai/CLIP.git` and download model
from [here](https://github.com/openai/CLIP).
- **MAE**: run `pip install timm==0.5.4` and download model
from [here](https://github.com/facebookresearch/mae).
- **Random ConvNet**: see `src/vision_models/generic.py`.

See `src/embeddings.py` for more info.

## Supported Environments
- **Gym**: already installed.
- **MiniGrid**: already installed.
- **Atari**: `pip install gym[atari] AutoROM; AutoROM`
- **DeepMind Control (DMC) Suite**: `pip install dm_control git+git://github.com/denisyarats/dmc2gym.git`
> If you run on a headless machine and get OpenGL error,
see [here](https://github.com/denisyarats/dmc2gym/issues/4).  
> If you run on a headless machine and get GLFW error,
try running with `MUJOCO_GL=egl` or see [here](https://github.com/deepmind/dm_control/issues/302).

- **Hand Manipulation Suite (HMS)**: see below.
- **Habitat**: see below.

See `src/gym_wrappers.py` for more info.


#### Install HMS
- Install [mujoco-py](https://github.com/openai/mujoco-py). This means you have to
download MuJoCo binaries and add them to your path.
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/your/path/to/.mujoco/mujoco210/bin
```
> To check that `mujoco-py` has been properly installed with GPU support, run
`python -c "import mujoco_py; print(mujoco_py.cymj)"` and be sure that the
output shows `linuxgpuextensionbuilder` (with **gpu**, not cpu).
If not, you need to build it from source. After cloning the repository, edit
`mujoco_py/builder.py` and change `LinuxCPUExtensionBuilder` to `LinuxGPUExtensionBuilder`.
Then run `pip install -e .`.

- Clone and install [mjrl](https://github.com/aravindr93/mjrl).
```
git clone https://github.com/aravindr93/mjrl.git
pip install -e .
```
- Clone and install [mj_envs](https://github.com/vikashplus/mj_envs),
and checkout `stable` branch.
```
git clone --recursive https://github.com/vikashplus/mj_envs.git
cd mj_envs
git checkout stable
pip install -e.
```
> If MuJoCo raises the `patchelf` error you need to run `conda install patchelf`.


#### Install Habitat
* Clone `https://github.com/sparisi/habitat-lab` and do a **full install** with `habitat_baselines`.
> The main differences between this and the original Habitat repository are:  
> 1) `STOP` action removed,  
> 2) Bugfix where the agent is placed slightly above the ground, and therefore the
terminal goal condition is never triggered.

* Download and extract Replica scenes in the root folder of `pvr`.
> WARNING! The dataset is very large!

```
sudo apt-get install pigz
git clone https://github.com/facebookresearch/Replica-Dataset.git
cd Replica-Dataset
./download.sh replica-path
```

If you have already downloaded it somewhere else, just make a symbolic link
```
ln -s path/to/Replica-Dataset .
```

> Habitat must run on GPU, so you must run the code with `--mp_start=spawn`.


## How to Run

### 1. Download PVRs From [Here](https://github.com/sparisi/pvr_habitat/releases/tag/models)
Place them in `PVR_MODEL_PATH`. This can be anywhere, just be sure to pass it
as environment variable when you run the code, or change the default path in
`src/embeddings.py`.

### 2. Get the Data
You must generate / download optimal trajectories for behavioral cloning.
* HabitatImageNav: use its native solver by running `data_generator/save_habitat_trajectories.py`.
* DeepMind Control Suite: coming soon.
* Hand Manipulation Suite: coming soon.

More data will be added in the future.

### 3. Convert the Data
* Data must be a `pickle` file with a dictionary with keys `obs`, `action`, `done`, `true_state`.
* Each dictionary entry must be a list, where each element is one trajectory.
* Each trajectory must be a `numpy.array` of `n` elements, with shapes
  * `obs.shape = (n, h, w, 3)`
  * `action.shape = (n,)` for discrete actions, `action.shape = (n, a)` for continuous actions,
  * `done.shape = (n,)`,
  * `true_state.shape = (n, s)`.

The script `data_generator/convert_data.py` shows how to convert data for DMC and HMS.

Finally, place the `.pickle` file in `data_path`. This can be anywhere, just be
sure to set it in `src/arguments.py`, or to pass it as argument when you run the code.

> Note that `obs` and `true_state` must be single-step, even if for training you
want to stack frames. For instance, in MuJoCo each step must have only one frame,
even if it is common practice to stack three consecutive frames to train the policy.
Frames will be stacked after the data is parsed, so that you can select how many
frames to stack regardless of how the data was generated.  

### 4. Parse the Data
To use frozen PVRs, you need to pass observations from optimal trajectories through
the vision models. You can do it simply by running `data_generator/save_embedded_obs.py`.
You can use `submitit_eo.py` to submit Slurm jobs.

### 5. Launch Behavioral Cloning
Run `main_bc.py` passing the desired scenes and embedding. You can use `submitit_bc.py`
to submit Slurm jobs (it supports distributed training and job auto-resume).
Results and policy model will be save in `save_path` (passed as argument).


## Full Example
```
python data_generator/save_habitat_trajectories.py --env=HabitatImageNav-room_0
python data_generator/save_habitat_trajectories.py --env=HabitatImageNav-hotel_0
python data_generator/save_embedded_obs.py --env=HabitatImageNav-room_0 --embedding_name=resnet50
python data_generator/save_embedded_obs.py --env=HabitatImageNav-hotel_0 --embedding_name=resnet50
PVR_MODEL_PATH=/your/path/to/pvrs python main_bc.py --env=HabitatImageNav-room_0,HabitatImageNav-hotel_0 --embedding_name=resnet50
```
When you launch `main_bc.py`, training and testing are performed in parallel.
Training temporarily stops if testing is not done for the previous epoch and new
testing has been requested. Below is an example of what you will see on screen.
```
   ___ 0 | HabitatImageNav-hotel_0 : return -5.112821574211057, success 0.0
   ___ 0 | HabitatImageNav-room_0 : return -5.018890399932798, success 0.0
   ### 1 | loss 0.8488612916018518, norm 1.102378161557042
   ___ 1 | HabitatImageNav-room_0 : return -0.2623981393739256, success 0.0
   ___ 1 | HabitatImageNav-hotel_0 : return -2.906053262925104, success 0.0
   ### 2 | loss 0.6933267779508896, norm 1.5788468601507564
   ___ 2 | HabitatImageNav-hotel_0 : return 2.3804228174887783, success 0.0
   ___ 2 | HabitatImageNav-room_0 : return 0.38858375724854854, success 0.0
   ### 3 | loss 0.6331057658810133, norm 1.5656620492636022
HabitatImageNav-room_0 @ 5:  46%|¦¦¦¦¦     | 23/50 [01:41<01:37,  3.62s/it]
epoch:  9%|¦¦¦¦¦¦    | 9/100 [17:49<3:43:04,  4.50s/it]
```
`   ___ @ <epoch>` denotes testing results.  
`   ### @ <epoch>` denotes training stats.  
`<env_id> @ <epoch>: <progress_bar>` denotes testing progress.  
`epoch: <progress_bar>` denotes training progress.  

Finally, please check carefully `src/arguments.py` for a list of all hyperparameters.


## Known Issues
* **Zombie processes.** If a run crashes or is killed (e.g., by `CTR+C`), test
  processes started by `torch.multiprocessing.Pool` may not be properly terminated.
  This may happen when Habitat simulations (for testing) are running.
  Check `htop` for any zombie process and manually kill them.
* **CUDA OOM.** Test processes must be started with the `spawn` method if the
  environment runs on GPU. Unfortunately, this creates copies of tensors shared
  over memory (the network model in our case). This means that you may get
  `CUDA error: out of memory` if too many environments / too large models are used.
* **Corrupted checkpoints.** If Slurm kills your job during `torch.save()`, your
  model will be corrupted (it is very rare, but may happen). When the run resumes you will get
  `RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory`.
