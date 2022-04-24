#!/bin/bash
python behavioral_cloning/save_opt_trajectories.py --n_trajectories=10000 --env=HabitatImageNav-apartment_0
python behavioral_cloning/save_opt_trajectories.py --n_trajectories=10000 --env=HabitatImageNav-frl_apartment_0
python behavioral_cloning/save_opt_trajectories.py --n_trajectories=10000 --env=HabitatImageNav-room_0
python behavioral_cloning/save_opt_trajectories.py --n_trajectories=10000 --env=HabitatImageNav-hotel_0
python behavioral_cloning/save_opt_trajectories.py --n_trajectories=10000 --env=HabitatImageNav-office_0
