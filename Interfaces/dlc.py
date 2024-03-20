import multiprocessing as mp
import os
import shutil
import sys
import time
from datetime import datetime
from multiprocessing.shared_memory import SharedMemory

import cv2
import numpy as np
from cv2 import getAffineTransform, invertAffineTransform
from dlclive import DLCLive, Processor

from utils.helper_functions import read_yalm

np.set_printoptions(suppress=True)

os.environ["DLClight"] = "True"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class DLC:
    """
    DeepLabCut (DLC) integration for processing video frames and inferring pose.

    Args:
        path (str): Path to the DLC model.
        shared_memory_shape: Shape of the shared memory array.
        logger: Logger object for recording data.
        joints: List of joints for pose estimation.

    Attributes:
        path (str): Path to the DLC model.
        nose_y (int): Nose position along the y-axis.
        theta (int): Angle parameter.
        timestamp (int): Timestamp for pose data.
        sm (SharedMemory): Shared memory block for pose data.
        data (np.ndarray): Numpy array using shared memory for pose data.
        setup_ready (mp.Event): Event indicating DLC setup readiness.
        close (mp.Event): Event to signal the termination of DLC processing.
        source_path (str): Source path for DLC data.
        target_path (str): Target path for saving DLC data.
        joints (list): List of joints for pose estimation.
        logger: Logger object for recording data.
        curr_pose (np.ndarray): Current pose data.
        frame_process (mp.Queue): Queue for receiving video frames for processing.
        dlc_proc (Processor): DLC processor instance.
        dlc_live: DLCLive instance for live inference.
        pose_hdf5: HDF5 dataset for raw pose data.
        pose_hdf5_processed: HDF5 dataset for processed pose data.
        frame: Current video frame.
    """

    def __init__(
        self,
        frame_process,
        dlc_queue,
        model_path: str,
        shared_memory_shape,
        logger,
    ):
        self.model_path = model_path
        self.nose_y = 0
        self.theta = 0
        self.timestamp = 0
        self.rot_angle = self.calculate_rotation_angle(side=1, base=0.8)


        # attach another shared memory block
        self.sm = SharedMemory("pose")
        # create a new numpy array that uses the shared memory
        self._dlc_pose = np.ndarray(
            shared_memory_shape, dtype=np.float32, buffer=self.sm.buf
        )

        self.setup_ready = mp.Event()
        self.setup_ready.clear()
        self.close = mp.Event()
        self.close.clear()
        self.logger = logger
        folder = (f"Recordings/{self.logger.trial_key['animal_id']}"
                  f"_{self.logger.trial_key['session']}/")
        self.source_path = self.logger.source_path + folder
        self.target_path = self.logger.target_path + folder
        self.joints = read_yalm(
            path= self.model_path,
            filename="pose_cfg.yaml",
            variable="all_joints_names",
        )

        h5s_filename = (f"{self.logger.trial_key['animal_id']}_"
                        f"{self.logger.trial_key['session']}_"
                        f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.h5")
        self.filename_dlc = "dlc_" + h5s_filename
        self.logger.log_recording(
            dict(
                rec_aim="openfield",
                software="EthoPy",
                version="0.1",
                filename=self.filename_dlc,
                source_path=self.source_path,
                target_path=self.target_path,
            )
        )

        self.filename_dlc_infer = "dlc_infer_" + h5s_filename
        self.logger.log_recording(
            dict(
                rec_aim="openfield",
                software="EthoPy",
                version="0.1",
                filename=self.filename_dlc_infer,
                source_path=self.source_path,
                target_path=self.target_path,
            )
        )
        self.filename_dlc_processed = "dlc_processed_" + h5s_filename
        self.logger.log_recording(
            dict(
                rec_aim="openfield",
                software="EthoPy",
                version="0.1",
                filename=self.filename_dlc_processed,
                source_path=self.source_path,
                target_path=self.target_path,
            )
        )
        self.screen_size = 310
        self.screen_pos = np.array(
            [[self.screen_size, 0], [self.screen_size, self.screen_size]]
        )

        self.frame_process = frame_process
        self.dlc_queue = dlc_queue

        self.dlc_proc = Processor()
        self.dlc_live = None

        self.pose_hdf5 = None
        self.pose_hdf5_processed = None
        self.pose_hdf5_infer = None

        self.frame = None

        self.dlc_live_process = mp.Process(
            target=self.setup,
            args=(self.frame_process, self.dlc_queue),
        )
        self.dlc_live_process.start()

    def setup(self, frame_process, dlc_queue):
        """
        Perform DLC setup and initialization.

        Args:
            frame_process (mp.Queue): Queue for receiving video frames for processing.
            dlc_queue (mp.Queue): Queue for sending calibration data.
        """
        if self.logger is not None:
            joints_types = [("tmst", np.double)]
            points = ["_x", "_y", "_score"]
            for joint in self.joints:
                for p in points:
                    joints_types.append((joint + p, np.double))

            self.pose_hdf5 = self.logger.createDataset(
                dataset_name="dlc",
                dataset_type=np.dtype(joints_types),
                filename=self.filename_dlc,
                log = True
            )

            self.pose_hdf5_infer = self.logger.createDataset(
                dataset_name="dlc_infer",
                dataset_type=np.dtype(joints_types),
                filename=self.filename_dlc_infer,
                log = False
            )

            joints_types_processed = [
                ("tmst", np.double),
                ("head_x", np.double),
                ("head_y", np.double),
                ("angle", np.double),
            ]

            self.pose_hdf5_processed = self.logger.createDataset(
                dataset_name="dlc_processed",
                dataset_type=np.dtype(joints_types_processed),
                filename=self.filename_dlc_processed,
                log = False
            )

        self.frame_process = frame_process
        # find corners of the arena
        self.corners = self.find_corners()
        self.M, self.M_inv = self.affine_transform(self.corners, self.screen_size)
        dlc_queue.put((self.M, self.corners))
        # initialize dlc models
        self.dlc_live = DLCLive(self.model_path, processor=self.dlc_proc)
        self.dlc_live.init_inference(self.frame_process.get()[1] / 255)

        # flag to indicate that all the dlc inits has finished before start experiment
        self.setup_ready.set()

        # start processing the camera frames
        self.process()

    def find_corners(self):
        """
        Find the corners of the arena based on the pixels of the image
        to define the real space using a DL model.

        Returns:
            np.ndarray: The 4 corners of the arena.
        """
        dlc_corners_path = (
            "/home/eflab/Desktop/"
            "Openfield_test_box-Konstantina-2023-11-20/exported-models/"
            "DLC_Openfield_test_box_resnet_50_iteration-3_shuffle-1"
        )

        # get a frame from the queue of the camera in order to init the dlc_live
        _, frame = self.frame_process.get()
        _frame = frame / 255
        dlc_proc = Processor()
        dlc_live = DLCLive(dlc_corners_path, processor=dlc_proc)
        dlc_live.init_inference(_frame)
        corners = []

        # TODO: add a check here to make sure that corners are reasonable distanced
        # use 4 images and find the mean of the corners
        # in order to avoid frames with no visible corners
        while len(corners) < 4:
            _, _frame = self.frame_process.get()
            pose = dlc_live.get_pose(_frame / 255)
            sys.stdout.write(
                "\rWait for high confidence corners" + "." * (int(time.time()) % 4)
            )
            sys.stdout.flush()
            if np.all(pose[:, 2] > 0.85):
                corners.append(dlc_live.get_pose(_frame / 255))
        corners = np.mean(np.array(corners), axis=0)

        # draw the corners in the last acquired frame
        for i in range(len(corners)):
            _frame = cv2.circle(
                _frame,
                (int(corners[i, 0]), int(corners[i, 1])),
                radius=7,
                color=(255, 0, 0),
                thickness=-1,
            )
        # save the corners image to make sure that all 4 corners are vissible and correctly detected
        cv2.imwrite("plane_corners.jpeg", _frame)

        return corners

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

        m = getAffineTransform(pts1, pts2)  # image to real
        m_inv = invertAffineTransform(m)  # real to image
        return m, m_inv

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

    def rotate_point(self, origin, point, angle_rad):
        """
        Rotate a point by a given angle around a given origin.

        Parameters:
        origin (tuple): The coordinates of the origin point (O).
        point (tuple): The coordinates of the point to rotate (A).
        angle (float): The angle of rotation in radians.

        Returns:
        tuple: The coordinates of the rotated point (A').
        """
        # Calculate the vector from the origin to the point
        vector_OA = np.array(point) - np.array(origin)

        # Calculate the components of the rotation matrix
        rotation_matrix = np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)],
            ]
        )

        # Apply rotation matrix to the vector
        rotated_vector = np.dot(rotation_matrix, vector_OA)

        # Calculate the new position of the rotated point
        return tuple(np.array(origin) + rotated_vector)

    def infer_apex(self, vertex1, vertex2, scaling_factor=0.8):
        """
        Infer the coordinates of the apex in an isosceles triangle given two vertices.

        Parameters:
        vertex1 (tuple): The coordinates of the first vertex.
        vertex2 (tuple): The coordinates of the second vertex.
        scaling_factor (float): Scaling factor for determining the distance of the apex from the midpoint.

        Returns:
        tuple: The coordinates of the inferred apex.
        """
        # Convert vertices to numpy arrays
        vertex1 = np.array(vertex1)
        vertex2 = np.array(vertex2)

        # Calculate the distance between the given vertices
        distance = np.linalg.norm(vertex2 - vertex1)

        # Calculate the midpoint of the given vertices
        midpoint = (vertex1 + vertex2) / 2

        # Calculate the unit vector along the line connecting the two vertices
        line_vector = (vertex2 - vertex1) / distance

        # Calculate the direction vector perpendicular to the line connecting the two vertices
        perpendicular_vector = np.array([-line_vector[1], line_vector[0]])

        # Calculate the length of the sides of the isosceles triangle
        side_length = scaling_factor * distance

        # Calculate the coordinates of the inferred apex
        apex = midpoint + perpendicular_vector * side_length

        return tuple(apex)

    def calculate_rotation_angle(self, side, base):
        """
        Calculate the rotation angle in radians using the law of cosines.

        Parameters:
        side (float): Length of one side of the triangle (nose_ear_right or nose_ear_left).
        base (float): Length of the base of the triangle (ear_ear).

        Returns:
        float: The rotation angle in radians.
        """
        # Calculate the cosine of the angle using the law of cosines
        cos_angle = (side**2 + side**2 - base**2) / (2 * side * side)

        # Calculate the angle in radians using arccosine
        angle_rad = np.arccos(cos_angle)

        return angle_rad

    def update_position(self, pose, prev_pose,  threshold=0.85):
        """
        Update the position based on the confidence of detected body parts.

        Args:
            pose (np.ndarray): Current pose information.
            prev_pose (np.ndarray): Previous pose information.
            threshold (float, optional): Confidence threshold. Defaults to 0.97.

        Returns:
            np.ndarray: Updated pose information.
        """
        scores = pose[:3, 2]  # Extract confidence scores for body parts
        low_conf = scores < threshold
        p_pose = pose[:3, :-1]

        if np.sum(low_conf) > 1:
            # If more than one point has low confidence, do not update the pose
            pose = prev_pose
        elif np.sum(low_conf) == 1:
            high_conf_points = p_pose[np.logical_not(low_conf)]
            if low_conf[0]:
                # if nose has low confidence
                p_pose[low_conf] = self.infer_apex(p_pose[2, :], p_pose[1, :])
            else:
                # if ear left has low confidence rotate with positive angle else negative
                angle = self.rot_angle if low_conf[1] else -self.rot_angle
                p_pose[low_conf] = self.rotate_point(
                    high_conf_points[0], high_conf_points[1], angle
                )
        pose[:3, :-1] = p_pose

        return pose

    def init_curr_pos(self, threshold=0.95):
        """
        Wait for the first pose with three high-confidence points in the head of the animal.

        Args:
            threshold (float, optional): Confidence threshold. Defaults to 0.85.
        """
        high_conf_points = False
        while not high_conf_points:
            if self.frame_process.qsize() > 0:
                _, self.frame = self.frame_process.get()
                p = self.dlc_live.get_pose(self.frame / 255)
                scores = np.array(p[0:3][:, 2])
                sys.stdout.write(
                    "\rWait for high confidence pose" + "." * (int(time.time()) % 4)
                )
                sys.stdout.flush()
                print("scores  ", scores)
                if len(np.where(scores >= threshold)[0]) == 3:
                    curr_pose = p
                    high_conf_points = True
                time.sleep(0.1)
        return curr_pose

    def process(self):
        """
        Run on a different process, wait to take an image, and return it.
        """
        # wait until a frame has all the 3 head point(nose and ear left/right)
        prev_pose = self.init_curr_pos()

        # run until close flag is set and frame_process is empty
        while not self.close.is_set() or self.frame_process.qsize() > 0:
            if self.frame_process.qsize() > 0:
                tmst, self.frame = self.frame_process.get()

                # get pose from frame
                dlc_pose_raw = self.dlc_live.get_pose(self.frame / 255)
                self.pose_hdf5.append(
                    "dlc", np.insert(np.double(dlc_pose_raw.ravel()), 0, tmst)
                )
                # check if position need any intervation
                curr_pose = self.update_position(dlc_pose_raw, prev_pose)
                final_pose = self.get_position(curr_pose, tmst)
                # save pose to the shared memory
                self._dlc_pose[:] = final_pose
                # save in the hdf5 files
                self.pose_hdf5_infer.append(
                    "dlc_infer", np.insert(np.double(curr_pose.ravel()), 0, tmst)
                )
                self.pose_hdf5_processed.append(
                    "dlc_processed",
                    final_pose,
                )
                prev_pose = curr_pose
            else:
                time.sleep(0.001)

    def get_position(self, pose, tmst):
        """Example coordinates for triangle vertices and square vertices
        pose[0]->nose, pose[1]->ear_left, pose[2]->ear_right
        """
        triangle_vertices = np.array(pose[0:3, 0:2])

        # Step 1: Find the centroid of the triangle
        centroid_triangle = self.find_centroid(triangle_vertices)
        # Step 2: Compute the vector from the centroid to nose of the triangle
        vector_to_nose = self.compute_vector(triangle_vertices[0, :], centroid_triangle)
        # Step 3: Compute the angle (phase) between the vectors
        angle = self.compute_angle(
            vector_to_nose, np.array([1, 0])
        )  # Assuming reference vector is [1, 0]

        centroid_triangle = np.dot(
            self.M,
            np.array([centroid_triangle[0], centroid_triangle[1], 1]),
        )
        # centroid_triangle = np.dot(
        #     self.M,
        #     np.array([pose[0, 0], pose[0, 1], 1]),
        # )
        return tmst, centroid_triangle[0], centroid_triangle[1], angle

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

    def move_files(self):
        """
        Find which files from the source_path are not in the target path and
        copy them.
        """
        if os.path.exists(self.source_path):
            files = [
                file for _, _, files in os.walk(self.source_path) for file in files
            ]
            files_target = [
                file for _, _, files in os.walk(self.target_path) for file in files
            ]
            for file in files:
                if file not in files_target:
                    shutil.copy2(
                        os.path.join(self.source_path, file),
                        os.path.join(self.target_path, file),
                    )
                    print(f"Transferred file: {file}")

    def stop(self):
        """stop processing"""
        self.close.set()
        self.sm.close()
        self.sm.unlink()
        self.logger.closeDatasets()
        self.move_files()
        self.dlc_live_process.terminate()
