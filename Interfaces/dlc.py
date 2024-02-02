import multiprocessing as mp
import os
import shutil
import sys
import time
from multiprocessing.shared_memory import SharedMemory

import cv2
import numpy as np
from cv2 import getAffineTransform, invertAffineTransform
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
    """

    def __init__(
        self,
        exp,
        frame_process,
        dlc_queue,
        path: str,
        shared_memory_shape,
        logger,
        joints,
        # beh_hash
    ):
        self.exp =exp
        self.path = path
        self.nose_y = 0
        self.theta = 0
        self.timestamp = 0
        # self.beh_hash = beh_hash

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

        self.source_path = "/home/eflab/alex/PyMouse/dlc/"
        self.target_path = "/mnt/lab/data/OpenField/"
        self.joints = joints
        self.logger = logger

        self.screen_size = 215
        self.screen_pos = np.array(
            [[self.screen_size, 0], [self.screen_size, self.screen_size]]
        )

        self.curr_pose = np.zeros((3, 3))

        self.frame_process = frame_process
        self.dlc_queue = dlc_queue

        self.dlc_proc = Processor()
        self.dlc_live = None

        self.pose_hdf5 = None
        self.pose_hdf5_processed = None

        self.frame = None

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

            filename_dlc, self.pose_hdf5_processed = self.logger.createDataset(
                self.source_path,
                self.target_path,
                dataset_name="dlc_processed",
                dataset_type=np.dtype(joints_types),
            )
            # TODO: remove sleep
            time.sleep(1)
            self.exp.log_recording(dict(rec_aim='body',
                                        software='Ethopy',
                                        version='0.1',
                                        filename=filename_dlc,
                                        source_path=self.source_path,
                                        target_path=self.target_path 
                                        ))

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
        self.frame_process = frame_process
        # find corners of the arena
        self.corners = self.find_corners()
        self.M, self.M_inv = self.affine_transform(self.corners, self.screen_size)
        dlc_queue.put((self.M, self.M_inv))
        # corner = {"beh_hash":self.beh_hash, "corners":self.corners, "affine_matrix":self.M}
        # self.logger.log("OpenField.Corners", corner, schema="behavior")
        # initialize dlc models
        self.dlc_live = DLCLive(self.path, processor=self.dlc_proc)
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
            "DLC_Openfield_test_box_resnet_50_iteration-2_shuffle-1"
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

        m = getAffineTransform(pts1, pts2) # image to real
        m_inv = invertAffineTransform(m) # real to image
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
            # if more than one point missing do not update pose
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
                self.final_pose = self.get_position(self.curr_pose, tmst)
                # save pose to the shared memory
                self.data[:] = self.final_pose
                # save in the hdf5 files
                self.pose_hdf5.append("dlc", np.insert(np.double(p.ravel()), 0, tmst))
                self.pose_hdf5_processed.append(
                    "dlc_processed",
                    np.insert(np.double(self.curr_pose.ravel()), 0, tmst),
                )
            else:
                time.sleep(0.001)

    def get_position(self, pose, tmst):
        # Example coordinates for triangle vertices and square vertices
        # pose[0]->nose, pose[1]->ear_left, pose[2]->ear_right

        triangle_vertices = np.array(pose[0:4, 0:2])

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
        return centroid_triangle[0], centroid_triangle[1],tmst, angle
    
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

    def stop(self):
        """stop processing"""
        self.close.set()
        self.sm.close()
        self.sm.unlink()
        self.logger.closeDatasets()
        # self.move_hdf()
        print("self.dlc_live_process.join()")
        self.dlc_live_process.terminate()
        print("self.dlc_live_process.join()")
