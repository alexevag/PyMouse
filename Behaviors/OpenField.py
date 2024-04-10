import multiprocessing as mp
import sys
import time

import datajoint as dj
import numpy as np

from core.Behavior import Behavior, behavior
from Interfaces.camera import WebCam
from Interfaces.dlc import DLC
from utils.helper_functions import shared_memory_array


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

        # create a mp Queue for the communication of dlc and camera
        self.process_q = mp.Queue(maxsize=2)
        self.process_q.cancel_join_thread()

        # return only x,t,tmst and angle
        self.pose, self.sm = shared_memory_array(
            name="pose",
            rows_len=1,
            columns_len=4,
        )
        self.shared_memory_shape = (1, 4)

        # read screen parameters
        # screen_params = self.logger.get(
        #     table="SetupConfiguration.Screen",
        #     key=f"setup_conf_idx={self.exp.params['setup_conf_idx']}",
        #     as_dict=True,
        # )[0]
        # screen_height, screen_width = self.screen_dimensions(screen_params["size"])
        # self.screen_size = int(screen_width * 10)  # mm
        self.screen_size = 310
        self.screen_pos = np.array(
            [[self.screen_size, 0], [self.screen_size, self.screen_size]]
        )

    def setup(self, exp):
        super(OpenField, self).setup(exp)

        # start camera recording process
        self.cam = WebCam(
            self.exp,
            logger=self.logger,
            process_queue=self.process_q,
        )

        # start DLC process
        self.dlc = DLC(
            self.process_q,
            self.dlc_queue,
            model_path=self.exp.params["model_path"],
            shared_memory_shape=self.shared_memory_shape,
            logger=self.logger,
        )

        # wait until the dlc setup hass finished initialization before start the experiment
        while not self.dlc.setup_ready.is_set():
            sys.stdout.write(
                "\rWaiting for the initialization of dlc" + "." * (int(time.time()) % 4)
            )
            sys.stdout.flush()
            time.sleep(0.1)
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
        self.logged_pos = False
        self.in_location = False
        self._resp_loc = None
        self.in_position_flag = False

        super().prepare(condition)

        # find real position of the objects
        self.response_locs = self.screen_pos_to_real_pos(
            self.curr_cond["response_loc_x"], const_dim=self.screen_size
        )
        self.rew_locs = self.screen_pos_to_real_pos(
            self.curr_cond["reward_loc_x"], const_dim=self.screen_size
        )

        self.position_tmst = 0
        self.resp_position_idx = -1

    def is_ready_start(self):
        tmst, x, y, angle = self.pose[0]
        r_x, r_y = self.curr_cond["init_loc_x"], self.curr_cond["init_loc_y"]
        return (
            np.sum((np.array([r_x, r_y]) - [x, y]) ** 2) ** 0.5
            < self.curr_cond["init_radius"]
        )

    def in_response_loc(self):
        """check if the animal position has been in a specific location"""
        self.tmst_cur, self.x_cur, self.y_cur, self.angle_cur = self.pose[0]
        # print("self.tmst_cur ", self.tmst_cur)
        # check if pos in response location
        self.resp_position_idx = self.position_in_radius(
            [self.x_cur, self.y_cur], self.response_locs, self.curr_cond["radius"]
        )
        if self.resp_position_idx != -1:
            self._resp_loc = self.response_locs[self.resp_position_idx]
            self.in_location = True
            if not self.logged_pos:
                act = {
                    "animal_loc_x": self.x_cur,
                    "animal_loc_y": self.y_cur,
                    "time": self.tmst_cur,
                    "in_pos": 1,
                    "resp_loc_x": self._resp_loc[0],
                    "resp_loc_y": self._resp_loc[1],
                }
                # print("act :", act)
                key = {**self.logger.trial_key, **act}
                self.logger.log("Activity", key, schema="behavior", priority=10)
                self.logger.log("Activity.Touch", key, schema="behavior")
                self.logged_pos = True

        if self.in_location is True and self.resp_position_idx == -1:
            self.logged_pos = False
            self.in_location = False
            act = {
                "animal_loc_x": self.x_cur,
                "animal_loc_y": self.y_cur,
                "time": self.tmst_cur,
                "in_pos": 0,
                "resp_loc_x": self._resp_loc[0],
                "resp_loc_y": self._resp_loc[1],
            }
            # print("act :", act)
            key = {**self.logger.trial_key, **act}
            self.logger.log("Activity", key, schema="behavior", priority=10)
            self.logger.log("Activity.Touch", key, schema="behavior")

        return self.resp_position_idx != -1

    def position_in_radius(self, target_position, positions_list, radius):
        target_position = np.array(target_position)
        positions_array = np.array(positions_list)

        distances = np.linalg.norm(positions_array - target_position, axis=1)
        indices_within_radius = np.where(distances <= radius)[0]

        if indices_within_radius.size > 0:
            # Return the index of the first position within the radius
            return indices_within_radius[0]
        else:
            return -1  # Return -1 if no position is within the radius

    def in_loc(self, is_reward):
        if self.in_response_loc():
            # if position_tmst==0 it it the first tmst in position set position_tmst to current time
            if self.position_tmst == 0:
                self.position_tmst = time.time()
            # find the response location
            resp_loc = self.response_locs[self.resp_position_idx]
            # if is_reward=True check if resp_loc in reward locations
            # else if is_reward=False check if it belong to a response loc that is not reward
            if (resp_loc in self.rew_locs) == is_reward:
                # check if the response ready time has passed
                return True
        elif self.position_tmst != 0:
            # reset position_tmst=0 since is not in any response loc
            self.position_tmst = 0

        return False

    def is_ready(self, ready_time):
        # if response_ready<=0 means there is no need to wait in reponse loc
        if ready_time <= 0:
            return True
        if self.position_tmst != 0:
            if time.time() - self.position_tmst > ready_time / 1000:
                # if response_ready has pass in the response position return True
                return True

        return False

    def in_punish_loc(self):
        # "check if the animal is in a punish location which is
        # all response locations except the reward"
        return self.in_loc(is_reward=False)

    def in_reward_loc(self):
        # "check if the animal is in a reward location"
        return self.in_loc(is_reward=True)

    def reward(self, tmst=0):
        """give reward at latest licked port

        After the animal has made a correct response, give the reward at the
        first licked port.

        Args:
            tmst (int, optional): Time in milliseconds. Defaults to 0.

        Returns:
            bool: True if rewarded, False otherwise
        """

        # # check that the last licked port ia also a reward port
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

    def screen_pos_to_real_pos(self, pos, const_dim=215):
        "Convert obj_pos in frame coordinates"
        screen_pos = self.screen_pos
        # in a square positions of the screen only one dimension is change
        diff_screen_pos = screen_pos[0, :] - screen_pos[1, :]
        dim_change = np.where(diff_screen_pos != 0)[0][0]

        if diff_screen_pos[dim_change] < 0:
            real_pos = (np.array(pos) + 0.5) * const_dim
        else:
            real_pos = (-np.array(pos) + 0.5) * const_dim

        locs = []
        if isinstance(real_pos, float):
            real_pos = [real_pos]
        for rl_pos in real_pos:
            if dim_change == 1:
                coords = np.array([screen_pos[0, 0], rl_pos])
            else:
                coords = np.array([rl_pos, screen_pos[0, 1]])
            locs.append(list(coords))
        return locs

    def exit(self):
        super().exit()
        print("cam stop")
        self.cam.stop_rec()
        print("dlc close")
        self.dlc.stop()
        print("interface cleanup")
        self.interface.cleanup()
        # clear mp Queue
        self.dlc_queue.close()
        self.process_q.close()
        # release shared memory
        # self.sm.close()
        # self.sm.unlink()
        # join processes
        time.sleep(0.5)
        self.logger.closeDatasets()
        print("self.logger.closeDatasets()")
        time.sleep(1)
