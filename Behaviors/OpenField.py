import multiprocessing as mp
import time
from typing import List, Optional, Tuple, Union

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

    SHARED_MEMORY_SHAPE = (1, 4)

    def __init__(self):

        # create a queue that returns the arena corners
        manager = mp.Manager()
        self.corners_dict = manager.dict()

        # Create shared memory array for pose
        self.pose, self.sm = shared_memory_array(
            name="pose",
            rows_len=self.SHARED_MEMORY_SHAPE[0],
            columns_len=self.SHARED_MEMORY_SHAPE[1],
        )

        self.response_locs: List[Tuple[float, float]] = []
        self._responded_loc: Tuple[float, float] = []
        self.reward_locs: List[Tuple[float, float]] = []
        self.init_loc: List[Tuple[float, float]] = ()
        self.position_tmst = 0
        self.affine_matrix: np.ndarray = np.array([])
        self.corners: List[Tuple[float, float]] = []
        self.response_loc: Optional[Tuple[float, float]] = None

        # To be set during setup
        self._screen_width: Optional[float] = None
        self.screen_pos: Optional[np.ndarray] = None
        self.dlc: Optional[DLC] = None

        # Current position and timestamp
        self.tmst_cur: Optional[float] = None
        self.x_cur: Optional[float] = None
        self.y_cur: Optional[float] = None
        self.angle_cur: Optional[float] = None

    def setup(self, exp):
        """Setup is running one time at each session"""
        super().setup(exp)

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

        self._initialize_dlc()

    def _initialize_dlc(self):
        if self.interface.camera is None:
            raise ValueError("Camera is not initialized")

        self.dlc = DLC(
            self.interface.camera.process_queue,
            self.corners_dict,
            model_path=self.exp.params["model_path"],
            shared_memory_shape=self.SHARED_MEMORY_SHAPE,
            logger=self.logger,
            arena_size=self.screen_width,
        )

        # Wait until the dictionary is filled with the values
        while 'affine_matrix' not in self.corners_dict or 'corners' not in self.corners_dict:
            time.sleep(0.1)
        self.affine_matrix = self.corners_dict['affine_matrix']
        self.corners = self.corners_dict['corners']
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
        """Prepare runs one time at the start of each trial"""
        super().prepare(condition)
        self.position_tmst = 0

        # find real position of the objects
        self.response_locs = self.screen_pos_to_real_pos(
            self.curr_cond["response_loc_x"], const_dim=self.screen_width
        )
        self.reward_locs = self.screen_pos_to_real_pos(
            self.curr_cond["reward_loc_x"], const_dim=self.screen_width
        )

        self.init_loc = [(self.curr_cond["init_loc_x"], self.curr_cond["init_loc_y"]),]

    def log_loc_activity(self, in_pos: int, response_loc: Tuple[float, float]) -> None:
        """Log activity with the given in_pos value."""
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

    def position_in_radius(
        self,
        target_position: Tuple[float, float],
        positions: List[Tuple[float, float]],
        radius: float,
    ) -> Optional[Tuple[float, float]]:
        """
        Determines if a target position is within a specified radius of any positions in a list.
        """
        positions_array = np.array(positions)
        if positions_array.ndim == 1:
            positions_array = positions_array.reshape(-1, 1)
        distances = np.linalg.norm(positions_array - np.array(target_position), axis=1)
        indices_within_radius = np.where(distances <= radius)[0]

        return (
            positions[indices_within_radius[0]]
            if indices_within_radius.size > 0
            else None
        )

    def in_location(
        self, locs: List[Tuple[float, float]],
        duration: float,
        radius: float = 0.0,
        log_act:bool = True,
    ) -> Union[Tuple[float, float], int]:
        """Check if the animal is in a location within a radius for a specified duration"""

        self.tmst_cur, self.x_cur, self.y_cur, self.angle_cur = self.pose[0]
        self.response_loc = self.position_in_radius(
            (self.x_cur, self.y_cur), locs, radius
        )

        if self.response_loc is not None:
            if self.position_tmst == 0:
                self.position_tmst = time.time()
                if log_act:
                    self._responded_loc = self.response_loc
                    self.log_loc_activity(1, self.response_loc)
        elif self.position_tmst != 0:
            self.position_tmst = 0
            if log_act:
                self.log_loc_activity(0, self._responded_loc)

        return self.response_loc if self.is_ready(duration) else 0

    def is_ready(self, init_duration: float, since: float = False) -> bool:
        """Check if the specified duration has passed since entering a location"""
        if init_duration <= 0:
            return True
        if self.position_tmst == 0:
            return False
        elapsed = time.time() - self.position_tmst
        if since:
            return self.position_tmst > since and elapsed > (init_duration / 1000)
        return elapsed > (init_duration / 1000)

    def is_correct(self) -> bool:
        """Check if the animal's response location is correct"""
        if self.response_loc is not None:
            correct_loc = self.response_loc in self.reward_locs
            if correct_loc:
                self.log_loc_activity(1, self.response_loc)
            return correct_loc
        return False

    def reward(self, tmst: float = 0) -> bool:
        """Give reward at latest licked port"""
        licked_port = self.is_licking(since=tmst, reward=True)
        if licked_port:
            self.interface.give_liquid(licked_port)
            self.log_reward(self.reward_amount[licked_port])
            self.update_history(licked_port, self.reward_amount[licked_port])
            return True
        return False

    def punish(self) -> None:
        self.update_history(self.response.port, punish=True)

    def screen_pos_to_real_pos(
        self, pos: Union[float, List[float]], const_dim: float
    ) -> List[Tuple[float, float]]:
        """Convert obj_pos in real coordinates"""
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

    def stop(self):
        """Stop the camera recording"""
        self.interface.cam.stop_rec()
        self.dlc.stop()
        time.sleep(0.5)
        self.logger.closeDatasets()
        print("Datasets closed")
        # self.process_q.close()
        # release shared memory
        # self.sm.close()
        # self.sm.unlink()
        # join processes

    def exit(self) -> None:
        """Clean up resources and exit"""
        super().exit()
        self.interface.cleanup()
