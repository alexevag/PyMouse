import io

import cv2
import imageio
import numpy as np
import pygame
from pygame.locals import *

from core.Stimulus import *


@stimulus.schema
class MultyGratings(Stimulus, dj.Manual):
    definition = """
    # This class handles the presentation orientation
    -> StimCondition
    ---
    thetas                  : blob       # in degrees (0-360)
    spatial_freqs           : blob       # cycles/deg
    phases                  : blob       # initial phase in rad
    contrasts               : blob       # 0-100 Michelson contrast
    temporal_freqs          : blob       # cycles/sec
    spatial_regions         : blob       # region of grating
    mask                    : blob       # mask to be applied
    duration                : smallint   # grating duration
    """

    cond_tables = ['MultyGratings']
    default_key = {'thetas'              : [0],
                   'spatial_freqs'       : [.05],
                   'phases'              : [0],
                   'contrasts'           : [100],
                   'square'              : [0],
                   'temporal_freqs'      : [1],
                   'spatial_regions'     : [{'x_start': 0, 'x_end': 1, 'y_start': 0, 'y_end': 1}],
                   'mask'                : [{'type':None}],
                   'duration'            : 3000,
                   }

    class Movie(dj.Part):
        definition = """
        # object conditions
        -> MultyGratings
        file_name                : varchar(256)
        ---
        clip                     : longblob     
        """

    def init(self, exp):
        super().init(exp)
        self.size = (self.monitor['resolution_x'], self.monitor['resolution_y'])    # window size
        self.color = [i*256 for i in self.monitor['background_color']]
        self.fps = self.monitor['fps']
        self.flag_no_stim = False

        # setup pygame
        pygame.init()
        self.clock = pygame.time.Clock()
        #self.screen = pygame.display.set_mode((0, 0), HWSURFACE | DOUBLEBUF | NOFRAME, display=self.screen_idx-1) #---> this works but minimizes when clicking (Emina)
        self.screen = pygame.display.set_mode(self.size, pygame.FULLSCREEN)
        self.unshow()
        pygame.mouse.set_visible(0)
        ymonsize = self.monitor['monitor_size'] * 2.54 / np.sqrt(1 + self.monitor['monitor_aspect'] ** 2)  # cm Y monitor size
        fov = np.arctan(ymonsize / 2 / self.monitor['monitor_distance']) * 2 * 180 / np.pi  # Y FOV degrees
        self.px_per_deg = self.size[1]/fov
      
    def make_conditions(self, conditions=[]):
        self.path = os.path.dirname(os.path.abspath(__file__)) + '/movies/'
        if not os.path.isdir(self.path):  # create path if necessary
            os.makedirs(self.path)
        super().make_conditions(conditions)
        for i,cond in enumerate(conditions):
            filename = self._get_filename(cond)

            filename = self._get_filename(cond)
            video_filename = self.exp.logger.get(schema='stimulus', table='MultyGratings.Movie',
                                        key={**cond, 'file_name': filename}, fields=['stim_hash'])
            if not video_filename:
                print('Making movie %s', filename)

                spatial_frequencies = cond['spatial_freqs']  # Spatial frequencies for each orientation
                temporal_frequency = cond['temporal_freqs']
                theta_phases = cond['theta_phases']
                fps = 30
                frames = fps*5 # seconds
                n_repeats = np.ceil(((cond['duration']/1000) * 30 / frames)).astype(int)
                if n_repeats==1 : frames = (np.ceil(cond['duration']/1000 * 30)).astype(int)
                print(f"frames: {frames}, (cond['duration']/1000) * 30 {(cond['duration']/1000) * 30}, repeats: {n_repeats},")

                # Define the spatial regions for each orientation
                spatial_regions,mask = cond['spatial_regions'], cond['mask']

                # Save the movie to a file
                movie = MovingGratingsMovie(spatial_frequencies, temporal_frequency, theta_phases, frames, 
                                            spatial_regions, mask, self.monitor['resolution_x'], self.monitor['resolution_y'])

                images = np.array(movie.generate_movie())[:,:,:,0]
                images = np.transpose(images[:, :, :], [0, 2, 1])
                self._im2mov(self.path + filename, images, repeats=n_repeats)
                self.logger.log('MultyGratings.Movie', {**cond, 'file_name': filename,
                                                'clip': np.fromfile(self.path + filename, dtype=np.int8)},
                                schema='stimulus', priority=2, block=True, validate=True)
            else:
                print(f"the condition exist {video_filename}")
        return conditions

    def prepare(self, curr_cond, stim_period=''):
        if super().prepare(curr_cond, stim_period): return
        self.curr_frame = 1
        self.clock = pygame.time.Clock()
        if stim_period!='':  
            self.curr_cond.update(dict(filename=self._get_filename(curr_cond[stim_period])))
        else:
            self.curr_cond.update(dict(filename=self._get_filename(curr_cond)))
        clip = self.exp.logger.get(schema='stimulus', table='MultyGratings.Movie', key=self.curr_cond, fields=['clip'])
        self.vid = imageio.get_reader(io.BytesIO(clip[0].tobytes()), 'mov')
        self.vsize = self.vid.get_meta_data()['size']
        self.vfps = self.vid.get_meta_data()['fps']
        self.isrunning = True
        self.timer.start()

    def present(self):
        # if self.flag_no_stim: return
        if self.timer.elapsed_time() < self.curr_cond['duration']:
            py_image = pygame.image.frombuffer(self.vid.get_next_data(),  self.vsize, "RGB")
            self.screen.blit(py_image, (0, 0))
            self.flip()
            self.curr_frame += 1
            self.clock.tick_busy_loop(self.vfps)
        else:
            self.isrunning = False
            self.vid.close()
            #self.unshow()

    def ready_stim(self):
        self.unshow([i*256 for i in self.monitor['ready_color']])

    def reward_stim(self):
        self.unshow([i*256 for i in self.monitor['reward_color']])

    def punish_stim(self):
        self.unshow([i*256 for i in self.monitor['punish_color']])

    def start_stim(self):
        self.unshow([i*256 for i in self.monitor['start_color']])

    def stop(self):
        self.unshow([i*256 for i in self.monitor['background_color']])
        self.log_stop()
        self.isrunning = False

    def flip(self):
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == QUIT: pygame.quit()
        self.flip_count += 1

    def unshow(self, color=False):
        if not color:
            color = self.color
        self.screen.fill(color)
        self.flip()

    def exit(self):
        pygame.mouse.set_visible(1)
        pygame.display.quit()
        pygame.quit()

    def _gray2rgb(self, im, c=1):
        return np.transpose(np.tile(im, [c, 1, 1]), (1, 2, 0))

    def _get_filename(self, cond, stim_period=''):
        # if self.flag_no_stim: return
        basename = ''.join([c for c in cond['stim_hash'] if c.isalpha()])
        pname = '_'.join('{}'.format(p) for p in self.monitor.values())
        return basename + '-' + pname + '.mov'

    def _im2mov(self, fn, images, repeats=1):
        w = imageio.get_writer(fn, fps=self.fps)
        for r in range(repeats):
            print('\r' + ('repeat %d' % (r)), end='')
            for i,frame in enumerate(images):
                w.append_data(frame)
        w.close()


class GratingRP(MultyGratings):
    """ This class handles the presentation of Gratings with an optimized library for Raspberry pi"""

    def setup(self):
        # setup parameters
        self.path = os.path.dirname(os.path.abspath(__file__)) + '/movies/'
        self.size = (self.monitor['resolution_x'], self.monitor['resolution_y'])     # window size
        self.color = [i*256 for i in self.monitor['background_color']]  # default background color
        self.phd_size = (50, 50)    # default photodiode signal size in pixels

        # setup pygame
        if not pygame.get_init():
            pygame.init()
        self.screen = pygame.display.set_mode(self.size,  pygame.FULLSCREEN)
        self.unshow() 
        pygame.mouse.set_visible(0)
        pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

        # setup movies
        from omxplayer import OMXPlayer
        self.player = OMXPlayer
        # store local copy of files
        if not os.path.isdir(self.path):  # create path if necessary
            os.makedirs(self.path)
        for cond in self.conditions:
            file = self.get_clip_info(cond, 'MultyGratings.Movie', 'file_name')
            filename = self.path + file
            if not os.path.isfile(filename):
                print('Saving %s ...' % filename)
                clip = self.get_clip_info(cond, 'MultyGratings.Movie', 'clip')
                clip[0].tofile(filename)
        # initialize player
        self.vid = self.player(filename, args=['--aspect-mode', 'stretch', '--no-osd'],
                    dbus_name='org.mpris.MediaPlayer2.omxplayer1')
        self.vid.stop()

    def prepare(self, curr_cond, stim_period=''):
        if super().prepare(curr_cond, stim_period): return
        # self.curr_cond = curr_cond
        self.unshow([i*256 for i in self.monitor['start_color']])
        self._init_player()
        self.isrunning = True
        self.timer.start() 

    def present(self):
        if self.flag_no_stim: return
        if self.timer.elapsed_time() < self.curr_cond['duration']:
            try:
                self.vid.play()
            except:
                self._init_player()
                self.vid.play()
        else: 
            self.isrunning = False
            self.vid.quit()

    def stop(self):
        try:
            self.vid.quit()
        except:
            self._init_player()
            self.vid.quit()
        self.unshow([i*256 for i in self.monitor['background_color']])
        self.log_stop()
        self.isrunning = False

    def _init_player(self):
        self.filename = self.path + self.get_clip_info(self.curr_cond, 'MultyGratings.Movie', 'file_name')

        try:
            self.vid.load(self.filename)
        except:
            self.vid = self.player(self.filename, args=['--aspect-mode', 'stretch', '--no-osd'],
                        dbus_name='org.mpris.MediaPlayer2.omxplayer1')
        self.vid.pause()

    def get_clip_info(self, key, table, *fields):
        key['file_name'] = self._get_filename(key)
        return key['file_name']


class MovingGratingsMovie:
    def __init__(self, spatial_frequencies, temporal_frequency, 
                theta_phases, duration, spatial_regions, mask,
                resolution_x,
                resolution_y):
        self.spatial_frequencies = spatial_frequencies
        self.temporal_frequency = temporal_frequency
        self.theta_phases = theta_phases
        self.duration = duration
        self.spatial_regions = spatial_regions
        self.mask = mask
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y

    def generate_movie(self):
        frame_rate = 30  # Frames per second
        total_frames = int(self.duration)
        frame_size = (self.resolution_x, self.resolution_y)  # Size of each frame

        # Create circular masks
        x = np.arange(frame_size[1])  # Width
        y = np.arange(frame_size[0])  # Height
        X, Y = np.meshgrid(x, y)

        # Initialize the mask as an array of False
        self.circle_masks = np.zeros(frame_size, dtype=bool)
        for circle in self.mask:
            circle_center = circle['center']
            circle_radius = circle['radius']

            # Compute the distance of each point in the grid from the center of the circle
            circle_mask = (X - circle_center[0])**2 + (Y - circle_center[1])**2 < circle_radius**2
            # Combine the circle mask with the overall mask
            self.circle_masks = self.circle_masks | circle_mask  

        movie_frames = []

        for frame_num in range(total_frames):
            print('\r' + ('frame %d' % (frame_num)), end='')
            time = frame_num / frame_rate
            grating_frame = self.generate_grating_frame(frame_size, time)
            grating_frame = grating_frame * self.circle_masks[:, :, None]
            movie_frames.append(grating_frame)

        return movie_frames

    def generate_grating_frame(self, frame_size, time):
        x = np.linspace(0, frame_size[1], frame_size[1])
        y = np.linspace(0, frame_size[0], frame_size[0])
        X, Y = np.meshgrid(x, y)

        # Calculate spatial phase
        spatial_phase = np.zeros(frame_size)
        for i, region in enumerate(self.spatial_regions):
            theta = np.radians(region['orientation'])
            # print("theta                                          " , theta)
            x_start, x_end = int(region['x_start'] * frame_size[1]), int(region['x_end'] * frame_size[1])
            y_start, y_end = int(region['y_start'] * frame_size[0]), int(region['y_end'] * frame_size[0])

            region_mask = np.zeros(frame_size, dtype=bool)
            region_mask[y_start:y_end, x_start:x_end] = True

            spatial_phase[region_mask] = 2 * np.pi * self.spatial_frequencies[i] * (np.cos(theta) * X[region_mask] + np.sin(theta) * Y[region_mask]) + self.theta_phases[i]

        # Calculate temporal phase
        temporal_phase = 2 * np.pi * self.temporal_frequency * time

        # Generate grating frame
        grating = np.sin(spatial_phase + temporal_phase)
        grating_frame = np.uint8((grating + 1) * 127.5)
        grating_frame = np.repeat(grating_frame[:, :, np.newaxis], 3, axis=2)

        return grating_frame

    def play_movie(self):
        pygame.init()

        frame_size = (self.resolution_x, self.resolution_y)
        screen = pygame.display.set_mode(frame_size)
        clock = pygame.time.Clock()

        movie_frames = self.generate_movie()

        running = True
        frame_num = 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if frame_num >= len(movie_frames):
                frame_num = 0

            screen.fill((0, 0, 0))
            frame_surface = pygame.surfarray.make_surface(movie_frames[frame_num])
            screen.blit(frame_surface, (0, 0))

            pygame.display.flip()
            frame_num += 1
            clock.tick(30)  # Frame rate of 30 frames per second

        pygame.quit()

    def save_movie(self, filename):
        frame_rate = 30  # Video frame rate
        frame_size = (self.resolution_x, self.resolution_y)  # Note that OpenCV expects width x height

        # Create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' for .avi
        video = cv2.VideoWriter(filename, fourcc, frame_rate, frame_size)

        movie_frames = self.generate_movie()
        for frame in movie_frames:
            # Convert the frame to BGR color format
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # Rescale to 8-bit (0 - 255) and convert to uint8
            rescaled = (255.0 / frame_bgr.max() * (frame_bgr - frame_bgr.min())).astype(np.uint8)
            # Add the frame to the video
            video.write(rescaled)

        # Release the VideoWriter
        video.release()
