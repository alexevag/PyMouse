import numpy as np
from scipy import interpolate

from Behaviors.OpenField import OpenField
from Experiments.Approach import Experiment
from Stimuli.Panda import Panda

interp = (
    lambda x: interpolate.splev(
        np.linspace(0, len(x), 100),
        interpolate.splrep(np.linspace(0, len(x), len(x)), x),
    )
    if len(x) > 3
    else x
)

# define session parameters
session_params = {
    "trial_selection": "random",
    "noresponse_intertrial": True,
    "setup_conf_idx": 16,
    "max_reward": 1000,
}


exp = Experiment()
exp.setup(logger, OpenField, session_params)
conditions = []
rot_f = lambda: interp((np.random.rand(30) - 0.5) * 10)

objs_idx = [(3, 8), (3, 8)]

objs_pos = [(-0.4, 0.4), (0.4, -0.4)]

panda_obj = Panda()
panda_obj.fill_colors.set(
    {
        "background": (0, 0, 0),
        "start": (0.5, 0.5, 0.5),
        "punish": (0, 0, 0),
        "reward": (255, 255, 255),
    }
)

for idx, loc in enumerate(objs_idx):
    conditions += exp.make_conditions(
        stim_class=panda_obj,
        conditions={
            # stimulus parameters
            "obj_id": objs_idx[idx],
            "obj_dur": 240000,
            "obj_pos_x": objs_pos[idx],
            "obj_mag": 0.4,
            "obj_rot": (rot_f(), rot_f()),
            "obj_tilt": 0,
            "obj_yaw": 0,
            "fun": 3,
            # experiment parameters
            "trial_duration": 15000,
            "intertrial_duration": 1000,
            "reward_duration": 10000,
            # Behaviour parameters
            "reward_loc_x": objs_pos[idx][0],
            "reward_loc_y": objs_pos[idx][0],
            "response_loc_x": objs_pos[idx],
            "response_loc_y": objs_pos[idx],
            "response_ready": 100,
            "init_loc_x": 85,
            "init_loc_y": 110,
            "init_ready": 0,
            "trial_ready": 250,
            "radius": 50,
            "init_radius": 50,
            "reward_amount": 6,
            "response_port": 1,
            "reward_port": 1,
        },
    )

# run experiments
exp.push_conditions(conditions)
exp.start()
