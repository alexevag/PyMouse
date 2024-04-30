import multiprocessing as mp
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Optional

import serial

from core.Interface import Interface, Port
from utils.Timer import Timer


class SerialPorts(Interface):
    """Class representing communication with a serial port."""

    def __init__(
        self,
        serial_port: str = "/dev/ttyUSB0",
        max_workers: int = 4,
        **kwargs
    ):
        """
        Initialize SerialPorts instance.

        Args:
            serial_port (str): The serial port to connect to.
            max_workers (int): The maximum number of workers for the ThreadPoolExecutor.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)

        self.serial = serial.serial_for_url(serial_port)
        self.serial.dtr = False
        self.timer = Timer()
        self.channels: Dict[str, Dict[int, Callable]] = {
            "Liquid": {1: self.serial.setDTR},
            "Lick": {1: self.serial.getDSR},
        }
        self.thread = ThreadPoolExecutor(max_workers=max_workers)
        self.stop_flag = mp.Event()
        self.stop_flag.clear()
        self.lick_thread = ThreadPoolExecutor(max_workers=max_workers)
        self.lick_thread.submit(self.check_events)
        self.resp_time_p = -1

    def give_liquid(self, port: Port, duration: Optional[int] = None):
        """
        Trigger liquid delivery on a specified port.

        Args:
            port (Port): Port object representing the target port.
            duration (int, optional): Duration of liquid delivery in milliseconds.
                                      Defaults to the configured duration for the specified port.
        """
        duration = duration or self.duration[port]
        self.thread.submit(self._give_pulse, port, duration)

    def check_events(self):
        """Monitor lick events on the serial port and log corresponding activity."""
        while not self.stop_flag.is_set():
            # check if any of the channels is activated
            for channel in self.channels["Lick"]:
                if self.channels["Lick"][channel]():
                    self._lick_port_activated(channel)
            time.sleep(0.035)

    def _lick_port_activated(self, port: Port):
        """
        Handle activation of the lick port and log the corresponding activity.

        Args:
            port (Port): The port that was activated.
        """
        self.resp_tmst = self.logger.logger_timer.elapsed_time()
        self.response = self.ports[Port(type="Lick", port=port) == self.ports][0]
        self.beh.log_activity({**self.response.__dict__, "time": self.resp_tmst})

    def _give_pulse(self, port: Port, duration: int):
        """
        Deliver a liquid pulse on the specified port for the given duration.

        Args:
            port (Port): Port object representing the target port.
            duration (int): Duration of the liquid pulse in milliseconds.
        """
        self.channels["Liquid"][port](True)
        time.sleep(duration / 1000)
        self.channels["Liquid"][port](False)

    def cleanup(self):
        """Clean up resources and stop asynchronous tasks."""
        self.stop_flag.set()
        with self.thread, self.lick_thread:
            print(
                "\rWaiting for thread pool tasks in serial ports interface to complete"
                + "." * (int(time.time()) % 4),
                end="",
            )
            sys.stdout.flush()
