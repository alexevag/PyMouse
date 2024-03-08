import io
import multiprocessing as mp
import os
import shutil
import threading
import time
import warnings
from datetime import datetime
from queue import Queue
from typing import Optional, Tuple

import numpy as np

from utils.Timer import Timer

# Libraries that only required in specific classes
try:
    from skvideo.io import FFmpegWriter

    IMPORT_SKVIDEO = True
except ImportError:
    IMPORT_SKVIDEO = False

try:
    import picamera

    IMPORT_PICAMERA = True
except ImportError:
    IMPORT_PICAMERA = False

try:
    import cv2

    IMPORT_CV2 = True
except ImportError:
    IMPORT_CV2 = False


class Camera:
    """
    A meta-class for capturing and recording video from different sources.

    Args:
        source_path (str, optional): The path(local for speed) where video frames
        will be saved. Defaults to None.
        target_path (str, optional): The path where recorded videos will be
        stored (copy from source_path). Defaults to None.
        filename (str, optional): The filename for recorded videos. Defaults to None.
        fps (int, optional): Frames per second for recording. Defaults to 30.
        logger_timer (Timer, optional): A common timer with experiment module for
        logging the frame timestamps. Defaults to None.
        process_queue (Queue): Additional keyword arguments.

    Attributes:
        initialized (threading.Event)
        recording (multyprocessing.Event)
        stop

    Raises:
        ImportError: If required modules are not installed.

    """

    def __init__(
        self,
        exp,
        fps: int = 30,
        process_queue: bool = False,
        logger=None,
    ):
        self.exp = exp
        self.initialized = threading.Event()
        self.initialized.clear()

        self.recording = mp.Event()
        self.recording.clear()

        self.stop = mp.Event()
        self.stop.clear()

        self.fps = fps
        self._cam = None
        self.logger = logger
        self.animal_id = self.logger.trial_key["animal_id"]
        self.session = self.logger.trial_key["session"]
        self.filename = f"animal_id_{self.animal_id}_session_{self.session}"
        self.source_path = "/home/eflab/alex/PyMouse/video/"
        self.target_path = "/mnt/lab/data/OpenField/"
        self.logger_timer = self.logger.logger_timer
        self.process_queue = process_queue
    
        camera_params = self.logger.get(
            table="SetupConfiguration.Camera",
            key=f"setup_conf_idx={self.exp.params['setup_conf_idx']}",
            as_dict=True,
        )[0]
        self.resolution = (camera_params["resolution_x"], camera_params["resolution_y"])
        
        if not globals()["IMPORT_SKVIDEO"]:
            raise ImportError(
                "you need to install the skvideo: sudo pip3 install sk-video"
            )

        # log video recording
        self.logger.log_recording(
            dict(
                rec_aim="openfield",
                software="EthoPy",
                version="0.1",
                filename=self.filename + ".mp4",
                source_path=self.source_path,
                target_path=self.target_path,
            )
        )
        h5s_filename = f"animal_id_{self.logger.trial_key['animal_id']}_session_{self.logger.trial_key['session']}.h5"
        self.filename_tmst = "video_tmst_" + h5s_filename
        self.logger.log_recording(
            dict(
                rec_aim="sync",
                software="EthoPy",
                version="0.1",
                filename=self.filename_tmst,
                source_path=self.source_path,
                target_path=self.target_path,
            )
        )

        self.camera_process = mp.Process(target=self.start_rec)
        self.camera_process.start()

    @property
    def filename(self):
        """str: The filename for recorded videos."""
        return self._filename

    @filename.setter
    def filename(self, filename):
        if filename is None:
            filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self._filename = filename

    @property
    def source_path(self) -> str:
        """str: The path where video frames will be saved."""
        return self._source_path

    @source_path.setter
    def source_path(self, source_path):
        # Make folder if it doesn't exist
        if not os.path.exists(source_path) and not self.recording.is_set():
            os.makedirs(source_path)

        # Check that folder has been made correctly
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"The path '{source_path}' does not exist.")

        self._source_path = source_path

    @property
    def target_path(self) -> str:
        """str: The path where recorded videos will be stored."""
        return self._target_path

    @target_path.setter
    def target_path(self, target_path):
        # Make folder if it doesn't exist
        if not os.path.exists(target_path) and not self.recording.is_set():
            os.makedirs(target_path)

        # Check that folder has been made correctly
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"The path '{target_path}' does not exist.")

        self._target_path = target_path

    def clear_local_videos(self):
        """
        Move video files from source_path to target_path.
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

    def setup(self):
        """
        Set up the camera and frame processing threads.
        """
        self.video_output = FFmpegWriter(
            self.source_path + self.filename + ".mp4",
            inputdict={
                "-r": str(self.fps),
            },
            outputdict={
                "-vcodec": "libx264",
                "-pix_fmt": "yuv420p",
                "-r": str(self.fps),
                "-preset": "ultrafast",
            },
        )

        if self.logger is not None:
            filename_tmst, self.dataset = self.logger.createDataset(
                self.source_path,
                self.target_path,
                dataset_name="frame_tmst",
                dataset_type=np.dtype([("tmst", np.double)]),
                filename=self.filename_tmst,
            )

        self.frame_queue = Queue()
        self.capture_runner = threading.Thread(target=self.rec, args=())
        self.write_runner = threading.Thread(
            target=self.dequeue, args=(self.frame_queue,)
        )

    def start_rec(self):
        """
        Start video recording.
        """
        self.setup()
        self.capture_runner.start()
        self.write_runner.start()
        self.capture_runner.join()
        self.write_runner.join()
        self.video_output.close()

    def dequeue(self, frame_queue: "Queue"):
        """
        Dequeue frames from the frame_queue and write them.
        """
        while not self.stop.is_set() or self.recording.is_set():
            if not frame_queue.empty():
                self.write_frame(frame_queue.get())
            else:
                time.sleep(0.01)

    def stop_rec(self):
        """
        Stop video recording.
        """
        self.stop.set()
        time.sleep(1)
        # TODO: use join and close (possible issue due to h5 files)
        self.camera_process.terminate()

    def rec(self):
        """
        Capture video frames.
        """
        raise NotImplementedError

    def write_frame(self, item):
        """
        Write a video frame.
        """
        raise NotImplementedError


class WebCam(Camera):
    """
    A class representing a webcam for capturing video frames.

    Args:
        Camera (class): The parent class for capturing and recording video frames.

    Attributes:
        stream (list): List for storing stream data.
        fps (int): Frames per second for recording.
        time (int): Time attribute.
        iframe (int): iframe attribute.
        reported_framerate (int): Reported framerate attribute.
        recording (bool): Flag indicating whether recording is active.
        camera (cv2.VideoCapture): OpenCV VideoCapture instance for accessing the webcam.

    Raises:
        RuntimeError: If there is no available camera.

    """

    def __init__(
        self,
        *args,
        resolution: tuple = (640, 480),
        **kwargs,
    ):
        """
        Initializes a WebCam instance.

        Args:
            resolution (Tuple[int, int], optional): Resolution of the webcam.
            Defaults to (640, 480).

        Raises:
            ImportError: If the cv2 package is not installed.
            RuntimeError: If there is no available camera.

        """
        self.fps = 30
        self.iframe = 0

        if not globals()["IMPORT_CV2"]:
            raise ImportError(
                "The cv2 package could not be imported. "
                "Please install it before using WebCam.\n"
                "You can install cv2 using pip:\n"
                'sudo pip3 install opencv-python"'
            )

        self.camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not self.camera.isOpened():
            raise RuntimeError(
                "No camera is available. Please check if the camera is connected and functional."
            )

        # self.camera.set(cv2.CAP_PROP_FPS, self.fps)
        self.set_resolution(resolution[1], resolution[0])
        self.width, self.height = resolution[0], resolution[1]

        super(WebCam, self).__init__(*args, **kwargs)

        # self.setup()

    def set_resolution(self, width, height):
        """set the resolution of the webcamera if it is possible
        However, the efficiency of changing the resolution may depend on the camera and
        the OpenCV backend being used. In some cases, changing the resolution may involve
        renegotiating the camera settings, and the efficiency could vary across different
        camera models and platforms.

        It's recommended to test and profile the performance with your specific camera to
        ensure that changing the resolution meets your performance requirements. If efficiency
        is a critical factor, you might want to consider using the camera's native resolution
        whenever possible.

        Args:
            width (int): width of frame
            height (int): height of frame
        """
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        check, image = self.get_frame()
        if check:
            shape = np.shape(image)
            self.width, self.height = shape[0], shape[1]

    def get_frame(self) -> Tuple[bool, np.ndarray]:
        """
        Capture a frame from the webcam.

        Returns:
            Tuple[bool, np.ndarray]: A tuple indicating success and the captured frame.
        """
        check, image = self.camera.read()
        if check:
            # If the capture was successful, convert the image to grayscale
            image = np.squeeze(np.mean(image, axis=2))
        return check, image

    def write_frame(self, item: Tuple[float, np.ndarray]) -> None:
        """
        Write a video frame to the output stream and update the timestamp dataset.

        Args:
            item (Tuple[float, np.ndarray]): A tuple containing the timestamp and the image frame.
        """
        img = item[1].copy()
        self.video_output.writeFrame(img)
        # Append the timestamp to the 'frame_tmst' h5 dataset
        self.dataset.append("frame_tmst", [np.double(item[0])])

    def rec(self):
        """
        Continuously capture video frames, update timestamp, and enqueue frames for processing.

        The method runs in a loop until the 'stop' event is set. It captures a frame from
        the webcam,records the elapsed time, increments the frame counter, and puts the
        timestamped frame into the 'frame_queue'. If a separate processing queue
        ('process_queue') is provided, the frame is also put into that queue, ensuring it
        doesn't exceed its maximum size. We need for the process_queue(size:2) the latest image
        so if it is full get a frame and put the latest one.
        """
        self.recording.set()
        # first_tmst = self.logger_timer.elapsed_time()
        # cam_tmst_first = self.camera.get(cv2.CAP_PROP_POS_MSEC)
        while not self.stop.is_set():
            check, image = self.get_frame()
            tmst = self.logger_timer.elapsed_time()
            # tmst = first_tmst + (self.camera.get(cv2.CAP_PROP_POS_MSEC)-cam_tmst_first)
            self.iframe += 1
            if check:
                self.frame_queue.put((tmst, image))
                # Check if a separate process queue is provided
                if self.process_queue is not False:
                    # Ensure the process queue doesn't exceed its maximum size
                    if self.process_queue.full():
                        self.process_queue.get()
                    self.process_queue.put_nowait((tmst, image))
        self.camera.release()
        self.recording.clear()

    def stop_rec(self):
        """
        Stop video recording and release resources.

        If video recording is in progress, the method releases the camera resources,
        closes the video output stream, clears the recording flag, and performs cleanup
        by removing local video files.
        """
        # TODO: check the stop_rec function and define a function release to be called by the process
        # if self.recording.is_set():
        # Release camera resources if recording is in progress
        # self.camera.release()

        # Call the superclass method to perform additional cleanup
        super().stop_rec()

        # Remove local video files
        self.clear_local_videos()


class PiCamera(Camera):
    def __init__(
        self,
        *args,
        sensor_mode: int = 0,
        resolution: tuple = (1280, 720),
        shutter_speed: int = 0,
        video_format: str = "rgb",
        logger_timer: "Timer" = None,
        **kwargs,
    ):
        if not globals()["IMPORT_PICAMERA"]:
            raise ImportError(
                "The picamera package could not be imported. "
                "Please install it before using this feature.\n"
                "You can install picamera using pip:\n"
                "pip install picamera"
            )

        self.video_format = video_format

        super(PiCamera, self).__init__(*args, **kwargs)
        self.resolution = resolution
        self.logger_timer = logger_timer
        self.video_type = self.check_video_format(video_format)
        self.shutter_speed = shutter_speed
        self.sensor_mode = sensor_mode
        self.cam = None
        self._picam_writer = None

    @property
    def sensor_mode(self) -> int:
        return self._sensor_mode

    @sensor_mode.setter
    def sensor_mode(self, sensor_mode: int):
        self._sensor_mode = sensor_mode
        if self.initialized.is_set():
            self.cam.sensor_mode = self._sensor_mode

    @property
    def fps(self) -> int:
        return self._fps

    @fps.setter
    def fps(self, fps: int):
        self._fps = fps
        if self.initialized.is_set():
            self.cam.framerate = self._fps

    def setup(self):
        if "compress" == self.check_video_format(self.video_format):
            self.stream = "compress"
            self.video_output = io.open(
                self.source_path + self.filename + "." + self.video_format, "wb"
            )
        else:
            self.stream = "raw"
            # buffering = 1 means that every line is buffered
            self.tmst_output = io.open(
                f"{self.source_path}tmst_{self.filename}.txt", "w", 1
            )
            out_vid_fn = self.source_path + self.filename + ".mp4"
            self.video_output = FFmpegWriter(
                out_vid_fn,
                inputdict={
                    "-r": str(self.fps),
                },
                outputdict={
                    "-vcodec": "libx264",
                    "-pix_fmt": "yuv420p",
                    "-r": str(self.fps),
                    "-preset": "ultrafast",
                },
            )
        super().setup()

    def rec(self):
        if self.recording.is_set():
            warnings.warn("Camera is already recording!")
            return

        self.recording_init()
        self.cam.start_recording(self._picam_writer, self.video_format)

    def recording_init(self):
        self.stop.clear()
        self.recording.set()

        self.cam = self.init_cam()
        self._picam_writer = self.PiCamOutput(
            self.cam,
            resolution=self.resolution,
            frame_queue=self.frame_queue,
            video_type=self.video_type,
            logger_timer=self.logger_timer,
        )

    def init_cam(self) -> "picamera.PiCamera":
        cam = picamera.PiCamera(
            resolution=self.resolution, framerate=self.fps, sensor_mode=self.sensor_mode
        )
        if self.shutter_speed != 0:
            cam.shutter_speed = self.shutter_speed
        self.initialized.set()

        return cam

    def stop_rec(self):
        if self.recording.is_set():
            self.cam.stop_recording()
            self.cam.close()
        super().stop_rec()

        self.video_output.close()
        if self.stream == "raw":
            self.tmst_output.close()

        self.recording.clear()
        self._cam = None
        self.clear_local_videos()

    def write_frame(self, item):
        if not self.stop.is_set():
            if self.stream == "compress":
                self.video_output.write(item[1])
            elif self.stream == "raw":
                img = item[1].copy()
                self.video_output.writeFrame(img)
                self.tmst_output.write(f"{item[0]}\n")
            else:
                warnings.warn(
                    "Recording is neither raw or stream so the results aren't saved"
                )
                return

    def check_video_format(self, video_format: str):
        if video_format in ["h264", "mjpeg"]:
            return "compress"
        if video_format in ["yuv", "rgb", "rgba", "bgr", "bgra"]:
            return "raw"
        raise Exception(
            f"the video format: {video_format} is not supported by picamera!!"
        )

    class PiCamOutput:
        def __init__(
            self,
            camera,
            resolution: Tuple[int, int],
            frame_queue,
            video_type,
            logger_timer,
        ):
            self.camera = camera
            self.resolution = resolution
            self.frame_queue = frame_queue
            self.logger_timer = logger_timer

            self.first_tmst = None
            self.tmst = 0
            self.i_frames = 0
            self.video_type = video_type
            self.frame = None

        def write(self, buf):
            """
            Write timestamps of each frame:
            https://forums.raspberrypi.com/viewtopic.php?f=43&t=106930&p=736694#p741128
            """
            if self.video_type == "raw":
                if self.camera.frame.complete and self.camera.frame.timestamp:
                    # TODO: Use camera timestamps   # [fixme]
                    # the first time consider the first timestamp as zero
                    # if self.first_tmst is None:
                    #     self.first_tmst = self.camera.frame.timestamp # first timestamp of camera
                    # self.tmst = (self.camera.frame.timestamp-self.first_tmst)+self.logger_timer.elapsed_time()
                    # print("self.logger_timer.elapsed_time() :", self.logger_timer.elapsed_time())

                    self.tmst = self.logger_timer.elapsed_time()
                    self.frame = np.frombuffer(
                        buf,
                        dtype=np.uint8,
                        count=self.camera.resolution[0] * self.camera.resolution[1] * 3,
                    ).reshape((self.camera.resolution[1], self.camera.resolution[0], 3))
                    self.frame_queue.put_nowait((self.tmst, self.frame))
            else:
                tmst_t = self.camera.frame.timestamp
                if tmst_t is not None:
                    # TODO:Fix timestamps in camera is in μs but in timer in ms
                    if self.first_tmst is None:
                        self.first_tmst = tmst_t  # first timestamp of camera
                    else:
                        self.tmst = (
                            tmst_t - self.first_tmst + self.logger_timer.elapsed_time()
                        )
                self.frame_queue.put_nowait((self.tmst, buf))
