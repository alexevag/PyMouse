import base64
import functools
import hashlib
import logging
import os
import sys
import threading
import time
from datetime import datetime
from getpass import getpass
from itertools import product
from multiprocessing import Event as ProcessEvent
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Dict, List, Tuple, Union

import datajoint as dj
import numpy as np
from scipy import ndimage

try:
    import yaml
    IMPORT_YALM = True
except ImportError:
    IMPORT_YALM = False

try:
    from scipy import ndimage
    IMPORT_SCIPY = True
except ImportError:
    IMPORT_SCIPY = False


def create_virtual_modules(schemata, create_tables=True,  create_schema=True):
    try:
        if dj.config["database.password"] is None:
            dj.config["database.password"] = getpass(prompt="Please enter DataJoint password: ")

        # Create virtual modules
        public_conn = dj.Connection(
            dj.config["database.host"],
            dj.config["database.user"],
            dj.config["database.password"],
        )
        virtual_modules = {}
        for name, schema in schemata.items():
            virtual_modules[name] = dj.create_virtual_module(name,
                                                             schema,
                                                             create_tables=create_tables,
                                                             create_schema=create_schema,
                                                             connection=public_conn)
        return virtual_modules, public_conn
    except Exception as e:
        error_message = (f"Failed to connect to the database due "
                         f"to an internet connection error: {e}")
        logging.error("ERROR %s", error_message)
        raise Exception(error_message) from e


def sub2ind(array_shape, rows, cols):
    return rows * array_shape[1] + cols


def flat2curve(I, dist, mon_size, **kwargs):
    def cart2pol(x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return (phi, rho)

    def pol2cart(phi, rho):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, y)

    if not globals()["IMPORT_SCIPY"]:
        raise ImportError(
            "you need to install the scipy: sudo pip3 install scipy"
        )

    params = dict({'center_x': 0, 'center_y': 0, 'method': 'index'},
                  **kwargs)  # center_x, center_y points in normalized x coordinates from center

    # Shift the origin to the closest point of the image.
    nrows, ncols = np.shape(I)
    [yi, xi] = np.meshgrid(np.linspace(1, ncols, ncols),np.linspace(1, nrows, nrows))
    yt = yi - ncols/2 + params['center_y']*nrows - 0.5
    xt = xi - nrows/2 - params['center_x']*nrows - 0.5

    # Convert the Cartesian x- and y-coordinates to cylindrical angle (theta) and radius (r) coordinates
    [theta, r] = cart2pol(yt, xt)

    # Compute spherical radius
    diag = np.sqrt(sum(np.array(np.shape(I)) ** 2))  # diagonal in px
    dist_px = dist / 2.54 / mon_size * diag  # closest distance from the monitor in px
    phi = np.arctan(r / dist_px)

    h = np.cos(phi / 2) * dist_px
    r_new = 2 * np.sqrt(dist_px ** 2 - h ** 2)

    # Convert back to the Cartesian coordinate system. Shift the origin back to the upper-right corner of the image.
    [ut, vt] = pol2cart(theta, r_new)
    ui = ut + ncols / 2 - params['center_y'] * nrows
    vi = vt + nrows / 2 + params['center_x'] * nrows

    # Tranform image
    if params['method'] == 'index':
        idx = (vi.astype(int), ui.astype(int))
        transform = lambda x: x[idx]
    elif params['method'] == 'interp':
        transform = lambda x: ndimage.map_coordinates(x, [vi.ravel() - 0.5, ui.ravel() - 0.5], order=1,
                                                      mode='nearest').reshape(x.shape)
    return (transform(I), transform)


def reverse_lookup(dictionary, target):
    return next(key for key, value in dictionary.items() if value == target)


def factorize(cond: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Factorizes conditions into individual combinations.

    This function takes a dictionary of conditions and generates all possible combinations
    of conditions, where each combination consists of one value for each key in the input
    dictionary.

    Args:
    - cond (Dict[str, Any]): A dictionary representing conditions.

    Returns:
    - List[Dict[str, Any]]: List of factorized conditions.

    Example:
    Suppose we have the following conditions:
    cond = {'param1': [1, 2], 'param2': [3, 4], 'param3': 'value', 'param4': (5, 6)}
    This function will generate the following combinations:
    [{'param1': 1, 'param2': 3, 'param3': 'value', 'param4': (5, 6)},
     {'param1': 1, 'param2': 4, 'param3': 'value', 'param4': (5, 6)},
     {'param1': 2, 'param2': 3, 'param3': 'value', 'param4': (5, 6)},
     {'param1': 2, 'param2': 4, 'param3': 'value', 'param4': (5, 6)}]
    """
    # Ensure all values are wrapped in lists
    values = [v if isinstance(v, list) else [v] for v in cond.values()]

    # Generate all combinations of conditions
    conds = []
    for combination in product(*values):
        # Create a dictionary representing each combination
        combined_cond = dict(zip(cond.keys(), combination))
        # Convert lists to tuples for immutability
        combined_cond = {
            k: tuple(v) if isinstance(v, list) else v for k, v in combined_cond.items()
        }
        conds.append(combined_cond)

    return conds


def make_hash(cond):
    def make_hashable(cond):
        if isinstance(cond, (tuple, list)):
            return tuple((make_hashable(e) for e in cond))
        if isinstance(cond, dict):
            return tuple(sorted((k, make_hashable(v)) for k, v in cond.items()))
        if isinstance(cond, (set, frozenset)):
            return tuple(sorted(make_hashable(e) for e in cond))
        return cond

    hasher = hashlib.md5()
    hasher.update(repr(make_hashable(cond)).encode())

    return base64.b64encode(hasher.digest()).decode()


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr): return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def iterable(v):
    return np.array([v]) if type(v) not in [np.array, np.ndarray, list, tuple] else v


class DictStruct:

    def __init__(self, dictionary):
        self.__dict__.update(**dictionary)

    def set(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

    def values(self):
        return self.__dict__.values()


def generate_conf_list(folder_path):
    contents = []
    files = os.listdir(folder_path)
    current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for i, file_name in enumerate(files):
        contents.append([i, file_name, '', current_datetime])
    return contents


def read_yalm(path: str, filename: str, variable: str) -> Any:
    """
    Read a YAML file and return a specific variable.

    Parameters:
        path (str): The path to the directory containing the file.
        filename (str): The name of the YAML file.
        variable (str): The name of the variable to retrieve from the YAML file.

    Returns:
        Any: The value of the specified variable from the YAML file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        KeyError: If the specified variable is not found in the YAML file.
    """
    if not globals()["IMPORT_YALM"]:
        raise ImportError(
            "you need to install the skvideo: sudo pip3 install PyYAML"
        )

    file_path = os.path.join(path, filename)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="UTF-8") as stream:
            file_yaml = yaml.safe_load(stream)
            try:
                return file_yaml[variable]
            except KeyError as exc:
                raise KeyError(f"The variable '{variable}' is not found in the YAML file.") from exc
    else:
        raise FileNotFoundError(f"There is no file '{filename}' in directory: '{path}'")


def shared_memory_array(name: str, rows_len: int, columns_len: int, dtype: str = "float32") -> tuple:
    """
    Creates or retrieves a shared memory array.

    Parameters:
        name (str): Name of the shared memory.
        rows_len (int): Number of rows in the array.
        columns_len (int): Number of columns in the array.
        dtype (str, optional): Data type of the array. Defaults to "float32".

    Returns:
        tuple(numpy.ndarray, multiprocessing.shared_memory.SharedMemory): 
        Shared memory array and SharedMemory object.
    """
    try:
        dtype_obj = np.dtype(dtype)
        bytes_per_item = dtype_obj.itemsize
        n_bytes = rows_len * columns_len * bytes_per_item

        # Create or retrieve the shared memory
        sm = SharedMemory(name=name, create=True, size=n_bytes)
    except FileExistsError:
        # Shared memory already exists, retrieve it
        sm = SharedMemory(name=name, create=False, size=n_bytes)
    except Exception as e:
        raise RuntimeError('Error creating/retrieving shared memory: ' + str(e)) from e

    # Create a numpy array that uses the shared memory
    shared_array = np.ndarray((rows_len, columns_len), dtype=dtype_obj, buffer=sm.buf)
    shared_array.fill(0)

    return shared_array, sm


def get_display_width_height(size: float, aspect_ratio: float) -> Tuple[float, float]:
    """
    Calculate the width and height of the screen in millimeters.

    Args:
        size (float): The diagonal size of the display in inches.
        aspect_ratio (float): The aspect ratio of the display.

    Returns:
        Tuple[float, float]: The width and height of the screen in millimeters.
    """
    # Calculate the diagonal size of the screen in millimeters
    screen_diagonal_mm = size * 25.4

    # Calculate the height of the screen in millimeters
    height = screen_diagonal_mm / np.sqrt(aspect_ratio**2 + 1)

    # Calculate the width of the screen in millimeters
    width = aspect_ratio * height

    return round(width, 2), round(height, 2)


def wait_for_flag(msg: str, flag: Union[threading.Event, ProcessEvent]) -> float:
    """
    Waits until the specified flag is set, printing a message and elapsed time.

    Args:
        msg (str): The message to be printed while waiting.
        flag: The event indicating whether the condition is met. It can be either threading.Event
        or multiprocessing.Event.

    Returns:
        float: elapsed time
    """
    if flag.is_set():
        return 0.  # If the flag is already set, return 0 as elapsed time

    start_time = time.time()
    while not flag.is_set():
        elapsed_time = time.time() - start_time
        for char in '|/-\\':
            sys.stdout.write(
                f'\r{msg}... {int(elapsed_time)}s {char}'
            )
            sys.stdout.flush()
            time.sleep(0.1)
    return elapsed_time
