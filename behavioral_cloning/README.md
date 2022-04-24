Place / save here all data used for behavioral cloning (BC), i.e., all
optimal trajectories and the corresponding embedded observations (PVRs).

## Scripts Description

* `save_opt_trajectories.py` generates optimal trajectories using Habitat's native solver.
  Data is saved as `.pickle` and takes a lot of space.
* `save_opt_trajectories_png.py` same, but image observations are saved as `.png` to save space.
* `save_opt_trajectories_jpeg.py` same, but image observations are saved as `.jpeg`.
  This is a lossy format compared to `.png`, but it is also the same used by ImageNet.
  Used to collect data for pre-training vision models.
* `save_embedded_obs.py` passes all images of from the aforementioned trajectories
  through embedding models. This way, when we run BC we just load these files
  rather than raw trajectories (unless embeddings are trained / fine-tuned).

To recap, the steps to run BC are:
1. For every scene you want to do BC on, generate optimal trajectories using
   `save_opt_trajectories.py` or `save_opt_trajectories_png.py`.
2. Pass the images through the desired embedding using `save_embedded_obs.py`.
3. Run one of the `main_bc.py` scripts (in root folder), passing the right scenes and embedding.

## Example

From root folder run
```
python behavioral_cloning/save_opt_trajectories.py --env=HabitatImageNav-apartment_0
python behavioral_cloning/save_opt_trajectories.py --env=HabitatImageNav-frl_apartment_0
python behavioral_cloning/save_embedded_obs.py --env=HabitatImageNav-apartment_0 --embedding_name=resnet50 --source=pickle
python behavioral_cloning/save_embedded_obs.py --env=HabitatImageNav-frl_apartment_0 --embedding_name=resnet50 --source=pickle
python main_bc_2.py --env=HabitatImageNav-apartment_0,HabitatImageNav-frl_apartment_0 --embedding_name=resnet50
```
