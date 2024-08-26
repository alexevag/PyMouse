import multiprocessing as mp
import time
from typing import List

import datajoint as dj
import numpy as np

from core.Behavior import Behavior, behavior
from Interfaces.dlc import DLC
from utils.helper_functions import get_display_width_height, shared_memory_array


@behavior.schema
class OpenField(Behavior, dj.Manual):
    definition = """
    # This class handles the behavior variables for RP
    ->BehCondition
    ---
    model_path           : varchar(256)
    """

    class Response(dj.Part):
        definition = """
        # Lick response condition
        -> OpenField
        response_loc_y            : float            # response y location
        response_loc_x            : float            # response x location
        response_ready            : float
        radius                    : float
        response_port             : tinyint          # response port id
        """

    class Init(dj.Part):
        definition = """
        # Lick response condition
        -> OpenField
        init_loc_y            : float            # response y location
        init_loc_x            : float            # response x location
        init_ready            : float
        init_radius           : float
        """

    class Reward(dj.Part):
        definition = """
        # reward port conditions
        -> OpenField
        ---
        reward_loc_x              : float
        reward_loc_y              : float
        reward_port               : tinyint          # reward port id
        reward_amount=0           : float            # reward amount
        reward_type               : varchar(16)      # reward type
        radius                    : float
        """

    cond_tables = [
        "OpenField",
        "OpenField.Response",
        "OpenField.Init",
        "OpenField.Reward",
    ]
    required_fields = ["reward_loc_x", "reward_amount"]
    default_key = {"reward_type": "water", "response_port": 1, "reward_port": 1}

    def __init__(self):
        # constant parameters
        self.model_columns_len = 3  # x,y,prediction confidence constant

        # create a queue that returns the arena cornerns
        self.dlc_queue = mp.Queue(maxsize=1)
        self.dlc_queue.cancel_join_thread()

        # return only x,t,tmst and angle
        self.pose, self.sm = shared_memory_array(
            name="pose",
            rows_len=1,
            columns_len=4,
        )
        self.shared_memory_shape = (1, 4)

        self.response_locs = []
        self.reward_locs = []
        self.init_loc = ()
        self.position_tmst = 0
        self.affine_matrix = []
        self.corners = []
        self.response_loc = ()
        self.screen_width = None
        self.screen_pos = None
        self.dlc = None

        self.tmst_cur = None
        self.x_cur = None
        self.y_cur = None
        self.angle_cur = None

    def setup(self, exp):
        """setup is running one time at each session"""
        super(OpenField, self).setup(exp)

        # get screen parameters
        screen_params = self.logger.get(
            table="SetupConfiguration.Screen",
            key=f"setup_conf_idx={self.exp.params['setup_conf_idx']}",
            as_dict=True,
        )[0]
        self.screen_width, _ = get_display_width_height(
            screen_params["size"], screen_params["aspect"]
        )
        self.screen_pos = np.array(
            [[self.screen_width, 0], [self.screen_width, self.screen_width]]
        )

        if self.interface.camera is not None:
            # start DLC process
            self.dlc = DLC(
                self.interface.camera.process_queue,
                self.dlc_queue,
                model_path=self.exp.params["model_path"],
                shared_memory_shape=self.shared_memory_shape,
                logger=self.logger,
                arena_size=self.screen_width,
            )
        else:
            raise ValueError("Camera is not initialized")

        # save the corners/position
        self.affine_matrix, self.corners = self.dlc_queue.get()
        self.logger.put(
            table="Configuration.Arena",
            tuple={
                "affine_matrix": self.affine_matrix,
                "corners": self.corners,
                **self.logger.trial_key,
            },
            schema="behavior",
        )

    def prepare(self, condition):
        """prepare runs one time at the start of each a trial"""

        self.position_tmst = 0
        super().prepare(condition)

        # find real position of the objects
        self.response_locs = self.screen_pos_to_real_pos(
            self.curr_cond["response_loc_x"], const_dim=self.screen_width
        )
        self.reward_locs = self.screen_pos_to_real_pos(
            self.curr_cond["reward_loc_x"], const_dim=self.screen_width
        )

        self.init_loc = (self.curr_cond["init_loc_x"], self.curr_cond["init_loc_y"])

    def log_loc_activity(self, in_pos: int, response_loc) -> None:
        """Log activity with the given in_pos value.

        Args:
            in_pos (int): Flag indicating whether the animal is in a
            specific position (1) or not (0).

        Returns:
            None
        """
        act = {
            "animal_loc_x": self.x_cur,
            "animal_loc_y": self.y_cur,
            "time": self.tmst_cur,
            "in_pos": in_pos,
            "resp_loc_x": response_loc[0],
            "resp_loc_y": response_loc[1],
        }
        key = {**self.logger.trial_key, **act}
        self.logger.log("Activity", key, schema="behavior", priority=10)
        self.logger.log("Activity.Touch", key, schema="behavior")

    def position_in_radius(self, target_position, positions: List, radius):
        """
        Determines if a target position is within a specified radius of any positions in a list.

        Args:
            target_position (list or np.array): The target position as a list or numpy array [x, y].
            positions (List): A list of positions, where each position is a list [x, y].
            radius (float): The radius within which to check if the target position is included.

        Returns:
            The first position within the radius if any, otherwise None.
        """
        target_position = np.array(target_position)
        positions_array = np.array(positions)
        if positions_array.ndim == 1:
            positions_array = positions_array.reshape(-1, 1)
        distances = np.linalg.norm(positions_array - target_position, axis=1)
        indices_within_radius = np.where(distances <= radius)[0]

        if indices_within_radius.size > 0:
            # Return the index of the first position within the radius
            return positions[indices_within_radius[0]]

        return None  # Return None if no position is within the radius

    def in_location(self, locs, duration: int, radius: float = 0.0):
        """
        Checks if the animal's position has been in a specific location for a given duration and
        within a specified radius.

        This method updates the animal's response location based on the specified radius and logs
        the activity when the animal enters or exits the response location. It also checks if a
        specified duration has passed while the animal is in the response location.

        Args:
            duration (int): The duration in seconds to check if the animal has been in the response
            location.
            radius (float): The radius within which to check if the animal is in the response
            location. Defaults to 0.0.

        Returns:
            The response location if the animal has been in it for the specified duration,
            otherwise 0.
        """
        self.tmst_cur, self.x_cur, self.y_cur, self.angle_cur = self.pose[0]
        self.response_loc = self.position_in_radius([self.x_cur, self.y_cur], locs, radius)
        # self.update_locations()
        # TODO if the reponse loc change it will take it as a new response
        # loc to reset the position_tmst
        if self.response_loc is not None:
            # log position on only when it enters the location
            # which is when the position_tmst equal 0
            if self.position_tmst == 0:
                self.position_tmst = time.time()
                self.log_loc_activity(1, self.response_loc)
        # log position off only in case it was in a response location
        # which is when the position_tmst!=0
        elif self.position_tmst != 0:
            self.position_tmst = 0
            self.log_loc_activity(0, self.response_loc)

        if self.is_ready(duration):
            return self.response_loc
        return 0

    # def update_locations(self):
    #     # print(
    #     #     "self.get_resp_obj_pos(self.resp_objs) ",
    #     #     self.get_resp_obj_pos(self.resp_objs),
    #     # )
    #     # print(
    #     #     "self.get_resp_obj_pos(self.reward_objs) ",
    #     #     self.get_resp_obj_pos(self.reward_objs),
    #     # )

    #     self.response_locs = self.screen_pos_to_real_pos(
    #         self.get_obj_pos(self.resp_objs), self.screen_width
    #     )
    #     self.reward_locs = self.screen_pos_to_real_pos(
    #         self.get_obj_pos(self.reward_objs), self.screen_width
    #     )
    #     self.response_obj_loc_dict = dict(
    #         zip(self.response_locs, self.curr_cond["response_loc_x"])
    #     )

    # def get_obj_pos(self, objs_ids):
    #     return [self.objects[obj].get_x_Pos for obj in objs_ids]

    def is_ready(self, init_duration, since=False):
        """
        Determines if a certain duration has passed since the last recorded position timestamp.

        Args:
            self: The instance of the class.
            init_duration (int): The duration to check against, in milliseconds.
            since (bool, optional): If True, checks if the duration has passed since a specific
            timestamp. If False, checks against the last recorded position timestamp. Defaults to
            False.

        Returns:
            bool: True if the specified duration has passed, False otherwise.
        """
        # if response_ready<=0 means there is no need to wait in reponse loc
        if init_duration <= 0:
            return True
        elif self.position_tmst == 0:
            return False
        elif since:
            return self.position_tmst > since and (time.time() - self.position_tmst) > (
                init_duration / 1000
            )
        else:
            # if response_ready has pass in the duration return True
            return (time.time() - self.position_tmst) > (init_duration / 1000)

    def is_correct(self):
        """Check if the response location is correct.

        Returns:
            bool: True if the response location is in the reward locations, False otherwise.
        """
        if self.response_loc is not None:
            correct_loc = self.response_loc in self.reward_locs
            # log the activity because before move to the next state
            if correct_loc:
                self.log_loc_activity(1, self.response_loc)
            return correct_loc
        return False

    def reward(self, tmst=0):
        """give reward at latest licked port

        After the animal has made a correct response, give the reward at the
        first licked port.

        Args:
            tmst (int, optional): Time in milliseconds. Defaults to 0.

        Returns:
            bool: True if rewarded, False otherwise
        """

        licked_port = self.is_licking(since=tmst, reward=True)
        if licked_port:
            self.interface.give_liquid(licked_port)
            # self.log_reward(self.reward_amount[self.licked_port])
            self.log_reward(self.reward_amount[licked_port])
            # self.update_history(self.response.port, self.reward_amount[self.licked_port])
            # TODO: use response loc in the history
            self.update_history(licked_port, self.reward_amount[licked_port])
            return True
        return False

    def punish(self):
        # TODO: use response loc in the history
        self.update_history(self.response.port, punish=True)

    def screen_pos_to_real_pos(self, pos, const_dim):
        "Convert obj_pos in real coordinates"
        screen_pos = self.screen_pos
        # in a square positions of the screen only one dimension is change
        diff_screen_pos = screen_pos[0, :] - screen_pos[1, :]
        dim_change = np.where(diff_screen_pos != 0)[0][0]

        real_pos = (
            (-1 if diff_screen_pos[dim_change] >= 0 else 1) * np.array(pos) + 0.5
        ) * const_dim

        # Ensure real_pos is always a list
        real_pos = [real_pos] if isinstance(real_pos, float) else real_pos

        locs = [
            (
                list(np.array([screen_pos[0, 0], rl_pos]))
                if dim_change == 1
                else list(np.array([rl_pos, screen_pos[0, 1]]))
            )
            for rl_pos in real_pos
        ]

        return locs

    def exit(self):
        super().exit()
        print("cam stop")
        self.interface.cam.stop_rec()
        print("dlc close")
        self.dlc.stop()
        print("interface cleanup")
        self.interface.cleanup()
        # clear mp Queue
        self.dlc_queue.close()
        # self.process_q.close()
        # release shared memory
        # self.sm.close()
        # self.sm.unlink()
        # join processes
        time.sleep(0.5)
        self.logger.closeDatasets()
        print("self.logger.closeDatasets()")
        time.sleep(1)
