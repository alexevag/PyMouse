import multiprocessing as mp
import os
import sys
import time
from multiprocessing.shared_memory import SharedMemory

import datajoint as dj
import numpy as np
import yaml
from cv2 import getAffineTransform, invertAffineTransform

from core.Behavior import Behavior, behavior
from Interfaces.camera import WebCam
from Interfaces.dlc import DLC


@behavior.schema
class OpenField(Behavior, dj.Manual):
    """_summary_

    Args:
        Behavior (_type_): _description_
        dj (_type_): _description_

    Raises:
        Exception: _description_
        Exception: _description_
        Exception: _description_

    Returns:
        _type_: _description_
    """

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

    def setup(self, exp):
        super(OpenField, self).setup(exp)
        self.init_params()

        # start camera recording process
        self.animal_id = self.logger.trial_key["animal_id"]
        self.session = self.logger.trial_key["session"]
        animal_id_session_str = f"animal_id_{self.animal_id}_session_{self.session}"
        self.cam = WebCam(
            self.exp,
            source_path=self.camera_source_path,
            target_path=self.camera_target_path,
            filename=animal_id_session_str,
            logger_timer=self.logger.logger_timer,
            logger=self.logger,
            process_queue=self.process_q,
            resolution=self.resolution,
        )

        # start DLC process
        self.dlc = DLC(
            self.process_q,
            self.calibration_queue,
            path=self.dlc_model_path,
            shared_memory_shape=self.shared_memory_shape,
            logger=self.logger,
            joints=self.all_joints_names,
        )

        # Wait for 15 seconds for the value to be available in the queue
        # if it takes more something is not working properly
        try:
            self.corners = self.calibration_queue.get(timeout=30)
        except mp.queues.Empty:
            raise Exception("Cannot get plane corners coordinates.")

        # wait until the dlc setup has finished initialization before start the experiment
        while not self.dlc.setup_ready.is_set():
            sys.stdout.write(
                "\rWaiting for the initialization of dlc" + "." * (int(time.time()) % 4)
            )
            sys.stdout.flush()
            time.sleep(0.1)

        self.M, self.M_inv = self.affine_transform(self.corners, self.screen_size)

        # self.pixel_unit(self.corners, self.screen_size)
        corners_dist = np.linalg.norm(self.corners[0] - self.corners[1])
        self.pixel_unit = self.screen_size / corners_dist

    def shared_memory_array(self, name, rows_len, columns_len, _dtype="float32"):
        _bytes = np.dtype(_dtype).itemsize
        n_bytes = rows_len * columns_len * _bytes
        try:
            # create the shared memory
            sm = SharedMemory(name=name, create=True, size=n_bytes)
        except FileExistsError:
            # sometimes when its not close correctly the sharedmemory remains
            # this is a workaround but it can create issues when the new array is not
            # the same size
            sm = SharedMemory(name=name, create=False, size=n_bytes)
        except Exception as shm_e:
            raise Exception("Error:" + str(shm_e))

        # create a new numpy array that uses the shared memory
        _data = np.ndarray((rows_len, columns_len), dtype=_dtype, buffer=sm.buf)
        _data.fill(0)

        return _data, sm

    def screen_dimensions(self, diagonal_inches, aspect_ratio=16 / 9):
        """returns the width and height of a screen based on ints
        diagonal

        Args:
            diagonal_inches (float): the diagonal of the screen in inch
            aspect_ratio (float, optional): Defaults to 16/9.

        Returns:
            float, float: height, width
        """
        diagonal_cm = diagonal_inches * 2.54

        # Calculate the width (x) and height (y) using the Pythagorean theorem
        x_cm = np.sqrt((diagonal_cm**2) / (1 + aspect_ratio**2))
        y_cm = aspect_ratio * x_cm

        return x_cm, y_cm

    def init_params(self):
        # constant parameters
        self.frame_tmst = mp.Value("d", 0)
        self.model_columns_len = 3  # x,y,prediction confidence constant

        # self.camera_source_path = os.path.abspath(os.getcwd())+'/source/'
        self.camera_source_path = "/home/eflab/Desktop/PyMouse_latest/PyMouse/video/"
        self.camera_target_path = "/mnt/lab/data/OpenField/"

        # read camera parameters
        camera_params = self.logger.get(
            table="SetupConfiguration.Camera",
            key=f"setup_conf_idx={self.exp.params['setup_conf_idx']}",
            as_dict=True,
        )[0]
        # read screen parameters
        screen_params = self.logger.get(
            table="SetupConfiguration.Screen",
            key=f"setup_conf_idx={self.exp.params['setup_conf_idx']}",
            as_dict=True,
        )[0]
        # screen_height, screen_width = self.screen_dimensions(screen_params["size"])

        # self.screen_size = int(screen_width * 10)  # mm
        self.screen_size = 215
        print(
            "self.screen_size self.screen_size self.screen_sizeself.screen_sizeself.screen_size ",
            self.screen_size,
        )
        self.screen_pos = np.array([[self.screen_size, 0], [self.screen_size, self.screen_size]])
        self.resolution = (camera_params["resolution_x"], camera_params["resolution_y"])

        # create a queue that returns the arena cornerns
        self.calibration_queue = mp.Queue(maxsize=1)
        self.calibration_queue.cancel_join_thread()

        # create share mp Queue
        self.process_q = mp.Queue(maxsize=2)
        self.process_q.cancel_join_thread()

        self.dlc_model_path = self.exp.params["model_path"]

        self.all_joints_names = self.read_yalm(
            path=self.dlc_model_path,
            filename="pose_cfg.yaml",
            variable="all_joints_names",
        )

        self.all_joints_name_len = len(self.all_joints_names)
        self.pose, self.sm = self.shared_memory_array(
            name="pose",
            rows_len=self.all_joints_name_len,
            columns_len=self.model_columns_len,
        )
        self.shared_memory_shape = (self.all_joints_name_len, self.model_columns_len)

    def read_yalm(self, path, filename, variable):
        """check if file exist and return the joint names"""
        # read dlc_pose_cfg and find number and joints names
        if os.path.exists(path + filename):
            stream = open(path + filename, "r", encoding="UTF-8")
            dlc_pose_cfg = yaml.safe_load(stream)
            all_joints_names = dlc_pose_cfg[variable]
        else:
            raise Exception(f"there is no file {filename} in directory: {path}")

        return all_joints_names

    def affine_transform(self, corners, screen_size):
        """_summary_

        Args:
            corners (_type_): _description_
            screen_size (_type_): _description_

        Returns:
            _type_: _description_
        """
        pts1 = np.float32([corners[0][:2], corners[1][:2], corners[2][:2]])

        pts2 = np.float32([[0, 0], [screen_size, 0], [0, screen_size]])

        M = getAffineTransform(pts1, pts2)
        M_inv = invertAffineTransform(M)
        return M, M_inv

    def prepare(self, condition):
        self.logged_pos = False
        self.in_location = False
        self._resp_loc = None
        self.in_position_flag = False

        super().prepare(condition)
        
        self.init_radius_p = int(self.curr_cond["init_radius"] / self.pixel_unit)
        self.radius_p = int(self.curr_cond["radius"] / self.pixel_unit)

        self.response_locs = self.screen_pos_to_frame_pos(
            self.curr_cond["response_loc_x"]
        )
        self.rew_locs = self.screen_pos_to_frame_pos((self.curr_cond["reward_loc_x"],))

        self.position_tmst = 0
        self.resp_position_idx = -1

    def screen_pos_to_frame_pos(self, pos, const_dim=250):
        "Convert obj_pos in frame coordinates"
        screen_pos = self.screen_pos
        diff_screen_pos = screen_pos[0, :] - screen_pos[1, :]
        dim_change = np.where(diff_screen_pos != 0)[0][0]

        if diff_screen_pos[dim_change] < 0:
            real_pos = (np.array(pos) + 0.5) * const_dim
        else:
            real_pos = (-np.array(pos) + 0.5) * const_dim

        locs = []
        if isinstance(real_pos, float): real_pos = [real_pos]
        for rl_pos in real_pos:
            if dim_change == 1:
                coords = np.dot(self.M_inv, np.array([screen_pos[0, 0], rl_pos, 1]))
            else:
                coords = np.dot(self.M_inv, np.array([rl_pos, screen_pos[0, 1], 1]))
            locs.append(list(coords))
        return locs

    def is_ready_start(self):
        x, y, tmst, angle = self.get_position()
        start_pos = np.dot(
            self.M_inv,
            np.array([self.curr_cond["init_loc_x"], self.curr_cond["init_loc_y"], 1]),
        )
        r_x, r_y = start_pos[0], start_pos[1]
        return np.sum((np.array([r_x, r_y]) - [x, y]) ** 2) ** 0.5 < self.init_radius_p

    def in_response_loc(self):
        """check if the animal position has been in a specific location"""
        self.x_cur, self.y_cur, self.tmst_cur, self.angle_cur = self.get_position()

        # check if pos in response location
        self.resp_position_idx = self.position_in_radius(
            [self.x_cur, self.y_cur], self.response_locs, self.radius_p
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

    def get_position(self):
        # Example coordinates for triangle vertices and square vertices
        # pose[0]->nose, pose[1]->ear_left, pose[2]->ear_right

        triangle_vertices = np.array(self.pose[0:4, 0:2])
        scores = np.array(self.pose[0:4][2])

        # Step 1: Find the centroid of the triangle
        centroid_triangle = self.find_centroid(triangle_vertices)
        # Step 2: Compute the vector from the centroid to nose of the triangle
        vector_to_nose = self.compute_vector(triangle_vertices[0, :], centroid_triangle)
        # Step 3: Compute the angle (phase) between the vectors
        angle = self.compute_angle(
            vector_to_nose, np.array([1, 0])
        )  # Assuming reference vector is [1, 0]

        # return centroid_triangle[0], centroid_triangle[1], self.frame_tmst.value, angle
        return self.pose[0, 0], self.pose[0, 1], self.frame_tmst.value, angle

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
        self.response.port = licked_port
        if licked_port:
            self.interface.give_liquid(licked_port)
            # self.log_reward(self.reward_amount[self.licked_port])
            self.log_reward(self.reward_amount[self.licked_port])
            # self.update_history(self.response.port, self.reward_amount[self.licked_port])
            self.update_history(self.response.port, self.reward_amount[self.licked_port])
            return True
        return False

    def punish(self):
        self.update_history(self.response.port, punish=True)

    def find_centroid(self, vertices):
        return np.mean(vertices, axis=0)

    def compute_vector(self, point1, centroid):
        return point1 - centroid

    def compute_angle(self, v1, v2):
        dot_product = np.dot(v1, v2)
        cross_product = np.cross(v1, v2)
        angle = np.arctan2(cross_product, dot_product)
        angle_degrees = np.degrees(angle)
        return angle_degrees

    def exit(self):
        super().exit()

        self.cam.stop_rec()
        self.dlc._close()
        self.interface.cleanup()
        # clear mp Queue
        self.calibration_queue.close()
        self.process_q.close()
        # release shared memory
        # self.sm.close()
        # self.sm.unlink()
        # join processes
        self.camera_process.join()
        self.camera_process.close()
        time.sleep(0.5)
        print("self.camera_process.join()")
        self.logger.closeDatasets()
        print("self.logger.closeDatasets()")
        time.sleep(1)
        print("self.dlc_live_process.join()")
        self.dlc_live_process.terminate()
        print("self.dlc_live_process.join()")
