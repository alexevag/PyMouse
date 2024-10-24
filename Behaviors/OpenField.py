import multiprocessing as mp
import time
from typing import Dict, List, Optional, Tuple, Union

import datajoint as dj
import numpy as np

from core.Behavior import Behavior, BehCondition, behavior
from core.Logger import stimulus
from Interfaces.dlc import DLCContinuousPoseEstimator, DLCCornerDetector
from utils.helper_functions import shared_memory_array


@behavior.schema
class OpenField(Behavior, dj.Manual):
    """
    This class handles the behavior variables for an open field experiment.
    It manages response locations, reward conditions, and animal tracking.
    """

    definition = """
    # This class handles the behavior variables for OpenField
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
        # Location to initialize the trial start
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

    conf_tables = {"behavior.ConfigurationArena": "experiment.SetupConfigurationArena",
                   "behavior.ConfigurationArena.Port": "experiment.SetupConfigurationArena.Port",
                   "behavior.ConfigurationArena.Screen": "experiment.SetupConfigurationArena.Screen",
                   "behavior.ConfigurationArena.Models": "experiment.SetupConfigurationArena.Models"}

    required_fields = ["reward_loc_x", "reward_amount"]
    default_key = {"reward_type": "water", "response_port": 1, "reward_port": 1}

    SHARED_MEMORY_SHAPE = (1, 4)

    def __init__(self):
        """Initialize the OpenField behavior class."""
        # create a queue that returns the arena corners
        self.manager = mp.Manager()
        self.corners_dict: Dict = self.manager.dict()
        # Create an event for DLC initialization
        self.dlc_init_event = self.manager.Event()

        # Create shared memory array for pose
        self.pose, self.sm = shared_memory_array(
            name="pose",
            rows_len=self.SHARED_MEMORY_SHAPE[0],
            columns_len=self.SHARED_MEMORY_SHAPE[1],
        )
        self.shared_memory_conf: Dict = {"name": "pose",
                                         "shape": self.SHARED_MEMORY_SHAPE,
                                         "dtype": np.float32}

        self.response_locs: List[Tuple[float, float]] = []
        self._responded_loc: Tuple[float, float] = []
        self.reward_locs: List[Tuple[float, float]] = []
        self.init_loc: List[Tuple[float, float]] = ()
        self.position_tmst = 0
        self.affine_matrix: np.ndarray = np.array([])
        self.corners: List[Tuple[float, float]] = []
        self.response_loc: Optional[Tuple[float, float]] = None

        # To be set during setup
        self.screen_pos: Optional[np.ndarray] = None
        self.dlc_model_path: str = ""
        self.dlc: Optional[DLCContinuousPoseEstimator] = None
        self.dlc_corners_path: str = ""
        self.dlcCorners: Optional[DLCCornerDetector] = None

        # Current position and timestamp
        self.tmst_cur: Optional[float] = None
        self.x_cur: Optional[float] = None
        self.y_cur: Optional[float] = None
        self.angle_cur: Optional[float] = None
        self.stop_calls = 0

    def setup(self, exp) -> None:
        """
        Set up the experiment parameters and initialize DLC.

        Args:
            exp: The experiment object containing parameters.
        """
        super().setup(exp)
        self.logger.log_setup_confs(self.conf_tables)
        self.stop_calls = 0
        setup_conf_idx = exp.params['setup_conf_idx']
        self.Arena_params = self.logger.get(schema='experiment',
                                            table='SetupConfigurationArena',
                                            key={'setup_conf_idx': setup_conf_idx},
                                            as_dict=True
                                            )[0]

        self.Arena_Screens_pos = self.logger.get(schema='experiment',
                                                 table='SetupConfigurationArena.Screen',
                                                 key={'setup_conf_idx': setup_conf_idx},
                                                 as_dict=True,
                                                 )[0]

        self.dlc_corners_path = self.logger.get(schema='experiment',
                                                table='SetupConfigurationArena.Models',
                                                fields=['path'],
                                                key={'setup_conf_idx': setup_conf_idx,
                                                     'target': 'corners'})[0]

        self.dlc_body_path = self.logger.get(schema='experiment',
                                             table='SetupConfigurationArena.Models',
                                             fields=['path'],
                                             key={'setup_conf_idx': setup_conf_idx,
                                                  'target': 'bodyparts'})[0]
        self.arena_size = self.Arena_params['size']

        self.screen_pos = np.array(
            [[self.Arena_Screens_pos['start_x'], self.Arena_Screens_pos['start_y']],
             [self.Arena_Screens_pos['stop_x'], self.Arena_Screens_pos['stop_y']]]
        )

        self._initialize_dlc()

    def _initialize_dlc(self) -> None:
        """Initialize the DeepLabCut (DLC) object for pose estimation."""
        if self.interface.camera is None:
            raise ValueError("Camera is not initialized")
        self.dlcCorners = DLCCornerDetector(frame_queue=self.interface.camera.process_queue,
                                            model_path=self.dlc_corners_path,
                                            arena_size=self.arena_size,
                                            result=self.corners_dict,
                                            logger=self.logger
                                            )
        self.dlcCorners.dlc_process.join(timeout=60)
        if self.dlcCorners.dlc_process.is_alive():
            raise Exception("Cannot find DLC corners!!")
        self.dlcCorners.dlc_process.close()
        self.affine_matrix = self.corners_dict["affine_matrix"]
        self.corners = self.corners_dict["corners"]

        self.dlc = DLCContinuousPoseEstimator(frame_queue=self.interface.camera.process_queue,
                                              model_path=self.dlc_body_path,
                                              logger=self.logger,
                                              shared_memory_conf=self.shared_memory_conf,
                                              affine_matrix=self.affine_matrix,
                                              wait_for_setup=True)

    def prepare(self, condition: Dict) -> None:
        """
        Prepare the trial with the given condition.

        Args:
            condition: A dictionary containing trial conditions.
        """
        super().prepare(condition)
        self.position_tmst = 0

        # find real position of the objects
        self.response_locs = self.screen_pos_to_real_pos(
            self.curr_cond["response_loc_x"], const_dim=self.arena_size
        )
        self.reward_locs = self.screen_pos_to_real_pos(
            self.curr_cond["reward_loc_x"], const_dim=self.arena_size
        )

        self.init_loc = [
            (self.curr_cond["init_loc_x"], self.curr_cond["init_loc_y"]),
        ]

    def log_loc_activity(self, in_pos: int, response_loc: Tuple[float, float]) -> None:
        """
        Log the animal's location activity.

        Args:
            in_pos: Integer indicating whether the animal is in position (1) or not (0).
            response_loc: Tuple of (x, y) coordinates of the response location.
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
        self.logger.log("Activity.Openfield", key, schema="behavior")

    def position_in_radius(
        self,
        target_position: Tuple[float, float],
        positions: List[Tuple[float, float]],
        radius: float,
    ) -> Optional[Tuple[float, float]]:
        """
        Determine if a target position is within a specified radius of any positions in a list.

        Args:
            target_position: Tuple of (x, y) coordinates of the target position.
            positions: List of tuples of (x, y) coordinates to check against.
            radius: The radius within which to check for proximity.

        Returns:
            The position within the radius if found, None otherwise.
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
        self,
        locs: List[Tuple[float, float]],
        duration: float,
        radius: float = 0.0,
        log_act: bool = True,
    ) -> Union[Tuple[float, float], int]:
        """
        Check if the animal is in a location within a radius for a specified duration.

        Args:
            locs: List of tuples of (x, y) coordinates to check.
            duration: The duration the animal needs to stay in the location.
            radius: The radius around the location to consider.
            log_act: Whether to log the activity.

        Returns:
            The response location if the animal is in position, 0 otherwise.
        """
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
        """
        Check if the specified duration has passed since entering a location.

        Args:
            init_duration: The duration to check against (in milliseconds).
            since: If True, check if the position timestamp is after the 'since' time.

        Returns:
            True if the duration has passed, False otherwise.
        """
        if init_duration <= 0:
            return True
        if self.position_tmst == 0:
            return False
        elapsed = time.time() - self.position_tmst
        if since:
            return self.position_tmst > since and elapsed > (init_duration / 1000)
        return elapsed > (init_duration / 1000)

    def is_correct(self) -> bool:
        """
        Check if the animal's response location is correct.

        Returns:
            True if the response location is correct, False otherwise.
        """
        if self.response_loc is not None:
            correct_loc = self.response_loc in self.reward_locs
            if correct_loc:
                self.log_loc_activity(1, self.response_loc)
            return correct_loc
        return False

    def reward(self, tmst: float = 0) -> bool:
        """
        Give reward at the latest licked port.

        Args:
            tmst: The timestamp to check licking since.

        Returns:
            True if reward was given, False otherwise.
        """
        licked_port = self.is_licking(since=tmst, reward=True)
        if licked_port:
            self.interface.give_liquid(licked_port)
            self.log_reward(self.reward_amount[licked_port])
            self.update_history(licked_port, self.reward_amount[licked_port])
            return True
        return False

    def punish(self) -> None:
        """Update history with a punishment."""
        self.update_history(self.response.port, punish=True)

    def screen_pos_to_real_pos(
        self, pos: Union[float, List[float]], const_dim: float
    ) -> List[Tuple[float, float]]:
        """
        Convert screen positions to real coordinates.

        Args:
            pos: The position(s) to convert.
            const_dim: The constant dimension (screen width).

        Returns:
            A list of tuples representing the real positions.
        """
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
        # self.interface.camera.release()
        self.stop_done += 1
        print("interface release")
        self.interface.release()
        print("dlc close")
        self.dlc.stop()
        print("interface cleanup")
        self.interface.cleanup()

    def __del__(self):
        """Destructor to ensure shared memory is cleaned up"""
        if hasattr(self, 'sm'):
            try:
                self.sm.close()
                self.sm.unlink()
            except FileNotFoundError:
                # The shared memory has already been unlinked or doesn't exist
                pass

    def exit(self) -> None:
        """Clean up resources and exit"""
        super().exit()
        if self.stop_calls == 0:
            self.stop()
        self.interface.cleanup()


@behavior.schema
class ConfigurationArena(dj.Manual):
    definition = """
    # Camera information
    arena_idx                : tinyint
    -> behavior.Configuration
    ---
    size                      : int
    discription               : varchar(256)
    """

    class Port(dj.Part):
        definition = """
        # Camera information
        -> master
        port                      : tinyint
        type="Lick"               : enum('Lick','Proximity')
        ---
        position_x                : float
        position_y                : float
        discription               : varchar(256)
        """

    class Screen(dj.Part):
        definition = """
        # Camera information
        -> master
        screen_idx              : tinyint UNSIGNED
        ---
        start_x                   : float
        start_y                   : float
        stop_x                    : float
        stop_y                    : float
        discription               : varchar(256)
        """

    class Corners(dj.Part):
        definition = """
        # Arena position
        -> master
        ---
        corners                  : blob             # corners position
        affine_matrix            : blob             # affine matrix from image to real space
        """

    class Models(dj.Part):
        definition = """
        # Arena position
        -> master
        name              : varchar(256)
        ---
        path              : varchar(256)
        target="bodyparts"      : enum('bodyparts','corners')
        """
