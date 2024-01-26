import multiprocessing as mp
import os
import sys
import time
from multiprocessing.shared_memory import SharedMemory
import shutil

import cv2
import numpy as np
from dlclive import DLCLive, Processor

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

    Methods:
        setup(frame_process, calibration_queue, frame_tmst): Perform DLC setup and initialization.
    """

    def __init__(
        self,
        frame_process,
        calibration_queue,
        path: str,
        shared_memory_shape,
        logger,
        joints,
    ):
        self.path = path
        self.nose_y = 0
        self.theta = 0
        self.timestamp = 0

        # attach another shared memory block
        self.sm = SharedMemory("pose")
        # create a new numpy array that uses the shared memory
        self.data = np.ndarray(
            shared_memory_shape, dtype=np.float32, buffer=self.sm.buf
        )

        self.setup_ready = mp.Event()
        self.setup_ready.clear()
        self.close = mp.Event()
        self.close.clear()

        self.source_path = "/home/eflab/Desktop/PyMouse_latest/PyMouse/dlc/"
        self.target_path = "/mnt/lab/data/OpenField/"
        self.joints = joints
        self.logger = logger

        self.curr_pose = np.zeros((3, 3))

        self.frame_process = frame_process
        self.calibration_queue = calibration_queue

        self.dlc_proc = Processor()
        self.dlc_live = None

        self.pose_hdf5 = None
        self.pose_hdf5_processed = None
        self.frame_tmst = None

        self.frame = None

        self.dlc_live_process = mp.Process(
            target=self.setup,
            args=(self.frame_process, self.calibration_queue, self.frame_tmst),
        )
        self.dlc_live_process.start()

    def setup(self, frame_process, calibration_queue, frame_tmst):
        """
        Perform DLC setup and initialization.

        Args:
            frame_process (mp.Queue): Queue for receiving video frames for processing.
            calibration_queue (mp.Queue): Queue for sending calibration data.
            frame_tmst (_type_): Timestamp data.
        """
        self.frame_process = frame_process
        self.frame_tmst = frame_tmst
        # find corners of the arena
        calibration_queue.put(self.find_corners())

        # initialize dlc models
        self.dlc_live = DLCLive(self.path, processor=self.dlc_proc)
        self.dlc_live.init_inference(self.frame_process.get()[1] / 255)

        self.setup_ready.set()
        if self.logger is not None:
            joints_types = [("tmst", np.double)]
            points = ["_x", "_y", "_score"]
            for joint in self.joints:
                for p in points:
                    joints_types.append((joint + p, np.double))

            _, self.pose_hdf5 = self.logger.createDataset(
                self.source_path,
                self.target_path,
                dataset_name="dlc",
                dataset_type=np.dtype(joints_types),
            )
            # joints_types_processed = [
            #     ("tmst", np.double),
            #     ("head_x", np.double),
            #     ("head_y", np.double),
            #     ("angle", np.double),
            # ]

            _, self.pose_hdf5_processed = self.logger.createDataset(
                self.source_path,
                self.target_path,
                dataset_name="dlc_processed",
                dataset_type=np.dtype(joints_types),
            )

            # self.exp.log_recording(dict(rec_aim='OpenField', software='PyMouse', version='0.1',
            #                 filename=filename_dlc, source_path=self.source_path,
            #                 target_path=self.target_path, rec_type='behavioral'))

        # start processing the camera frames
        self.process()

    def find_corners(self):
        """
        Find the corners of the arena based on the pixels of the image
        to define the real space using a DL model.

        Returns:
            np.ndarray: The 4 corners of the arena.
        """
        dlc_model_path = (
            "/home/eflab/Desktop/"
            "Openfield_test_box-Konstantina-2023-11-20/exported-models/"
            "DLC_Openfield_test_box_resnet_50_iteration-2_shuffle-1"
        )

        # get a frame from the queue of the camera in order to init the dlc_live
        _, frame = self.frame_process.get()
        _frame = frame / 255
        dlc_proc = Processor()
        dlc_live = DLCLive(dlc_model_path, processor=dlc_proc)
        dlc_live.init_inference(_frame)
        corners = []

        # TODO: add a check here to make sure that corners are reasonable distanced
        # TODO: use a more robust way instead of the 30 images to define the corners
        # like use only images with very high confidence results

        # use 30 images and find the median of the corners
        # in order to avoid frames with no visible corners
        while len(corners) < 10:
            _, _frame = self.frame_process.get()
            pose = dlc_live.get_pose(_frame / 255)
            # print("pose " , pose)
            # print("pose[:,2]  ", pose[:,2])
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

    def check_pred_confidence(self, scores, threshold=0.7):
        """
        Check the prediction confidence for every point and
        return the points with high confidence.

        Args:
            scores (np.ndarray): Confidence scores for each point.
            threshold (float, optional): Confidence threshold. Defaults to 0.7.

        Returns:
            np.ndarray: Indices of points with confidence below the threshold.
        """
        return np.where(scores <= threshold)[0]

    def rotate_vector(self, vector, angle):
        """
        Rotate a 2D vector by a given angle.

        Args:
            vector (np.ndarray): Input vector.
            angle (float): Rotation angle in radians.

        Returns:
            np.ndarray: Rotated vector.
        """
        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        rotated_vector = np.dot(rotation_matrix, vector)
        return rotated_vector

    def find_midpoint(self, point1, point2):
        """
        Find the midpoint between two points.

        Args:
            point1 (np.ndarray): First point.
            point2 (np.ndarray): Second point.

        Returns:
            np.ndarray: Midpoint coordinates.
        """
        point1 = np.array(point1)
        point2 = np.array(point2)
        midpoint = (point1 + point2) / 2
        return midpoint

    def find_rotation_and_missing_point(self, a, b, c, a_prime, b_prime):
        """
        Find the rotation angle and missing point given two sets of points.

        Args:
            a (np.ndarray): First point in the original set.
            b (np.ndarray): Second point in the original set.
            c (np.ndarray): Third point in the original set.
            a_prime (np.ndarray): First point in the rotated set.
            b_prime (np.ndarray): Second point in the rotated set.

        Returns:
            Tuple[float, np.ndarray, np.ndarray]: Rotation angle, missing point, rotated vector.
        """
        # Calculate the original angle formed by vector ab and the x-axis
        original_angle = np.arctan2(b[1] - a[1], b[0] - a[0])

        # Calculate the angle formed by the new vector a'b' and the x-axis
        new_angle = np.arctan2(b_prime[1] - a_prime[1], b_prime[0] - a_prime[0])

        # Compute the rotation angle
        rotation_angle = new_angle - original_angle

        middle = self.find_midpoint(a, b)
        middle_prime = self.find_midpoint(a_prime, b_prime)
        translation_middle = middle_prime - middle

        rotated_c = self.rotate_vector(c, rotation_angle)

        # Find the missing point d' using the rotated vectors
        d_prime = rotated_c + translation_middle

        return rotation_angle, d_prime, rotated_c

    def infer_missing_point(self, points, unknown_point):
        """
        Infer the position of a missing point based on the orientation and
        translation of the other 2 points.

        Args:
            points (List[np.ndarray]): Known points.
            unknown_point (np.ndarray): Missing point.

        Returns:
            np.ndarray: Inferred missing point.
        """
        # known and unknown points must be on the same order
        _, missing_point, _ = self.find_rotation_and_missing_point(
            points[0], points[1], points[2], unknown_point[0], unknown_point[1]
        )
        return missing_point

    def update_position(self, pose, threshold=0.85):
        """
        Update the position based on the confidence of detected body parts.

        Args:
            pose (np.ndarray): Current pose information.
            threshold (float, optional): Confidence threshold. Defaults to 0.85.

        Returns:
            np.ndarray: Updated pose information.
        """
        scores = np.array(pose[0:4][:, 2])
        high_conf_bdparts = np.where(scores >= threshold)[0]
        len_high_conf_bdparts = len(high_conf_bdparts)

        if len_high_conf_bdparts < 2:
            pose = self.curr_pose
        elif len_high_conf_bdparts == 2:
            curr_pose = np.array(self.curr_pose[0:4, 0:2])
            body_parts = np.array(pose[0:4, 0:2])
            order_points_prev = np.append(
                curr_pose[high_conf_bdparts],
                curr_pose[np.where(scores < threshold)[0]],
                axis=0,
            )
            order_points_new = np.append(
                body_parts[high_conf_bdparts],
                body_parts[np.where(scores < threshold)[0]],
                axis=0,
            )
            missing_point = self.infer_missing_point(
                order_points_prev, order_points_new
            )
            order_points_new[np.where(scores < threshold)[0]] = missing_point
            pose[0:4, 0:2] = order_points_new

        return pose

    def init_curr_pos(self, threshold=0.85):
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
                scores = np.array(p[0:4][:, 2])
                sys.stdout.write(
                    "\rWait for high confidence" + "." * (int(time.time()) % 4)
                )
                sys.stdout.flush()
                if len(np.where(scores >= threshold)[0]) == 3:
                    self.curr_pose = p
                    high_conf_points = True
                time.sleep(0.1)

    def process(self):
        """
        Run on a different process, wait to take an image, and return it.
        """
        # wait until a frame has all the 3 head point(nose and ear left/right)
        self.init_curr_pos()

        # run until close flag is set and frame_process is empty
        while not self.close.is_set() or self.frame_process.qsize() > 0:
            if self.frame_process.qsize() > 0:
                tmst, self.frame = self.frame_process.get()
                # get pose froma frame
                p = self.dlc_live.get_pose(self.frame / 255)
                # check if position need any intervation
                self.curr_pose = self.update_position(p)
                # save pose to the shared memory
                self.data[:] = self.curr_pose
                self.frame_tmst.value = tmst
                # save in the hdf5 files
                self.pose_hdf5.append("dlc", np.insert(np.double(p.ravel()), 0, tmst))
                self.pose_hdf5_processed.append(
                    "dlc_processed",
                    np.insert(np.double(self.curr_pose.ravel()), 0, tmst),
                )
            else:
                time.sleep(0.001)

    def move_hdf(self):
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

    def _close(self):
        """stop processing"""
        self.close.set()
        self.sm.close()
        self.sm.unlink()
        self.logger.closeDatasets()
        self.move_hdf()
        print("self.dlc_live_process.join()")
        self.dlc_live_process.terminate()
        print("self.dlc_live_process.join()")
