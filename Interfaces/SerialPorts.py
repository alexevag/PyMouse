import multiprocessing as mp
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import serial

from core.Interface import Interface, Port
from utils.Timer import Timer


class SerialPorts(Interface):
    """Class representing communication with a serial port."""

    def __init__(self, **kwargs):
        """Initialize SerialPorts instance.

        Args:
            **kwargs: Additional keyword arguments.

        Attributes:
            serial: Serial port instance.
            timer: Timer instance for timing operations.
            frequency: Communication frequency setting.
            thread: ThreadPoolExecutor for asynchronous tasks.
            stop_flag: Event flag to signal stopping asynchronous tasks.
            lick_thread: ThreadPoolExecutor for monitoring lick events.
            resp_time_p: Previous response time.
        """
        super(SerialPorts, self).__init__(**kwargs)

        serial_port = "/dev/ttyUSB0"
        self.serial = serial.serial_for_url(serial_port)
        self.serial.dtr = False
        self.timer = Timer()
        self.frequency = 10
        self.channels = {
            "Liquid": {
                1: self.serial.setDTR,
            },
            "Lick": {
                1: self.serial.getDSR,
            },
        }
        self.thread = ThreadPoolExecutor(max_workers=4)
        self.stop_flag = mp.Event()
        self.stop_flag.clear()
        self.lick_thread = ThreadPoolExecutor(max_workers=4)
        self.lick_thread.submit(self.check_events)
        self.resp_time_p = -1

    def give_liquid(self, port, duration=False, log=True):
        """Trigger liquid delivery on a specified port.

        Args:
            port: Port object representing the target port.
            duration: Duration of liquid delivery in milliseconds.
                      Defaults to the configured duration for the specified port.
        """
        if not duration:
            duration = self.duration[port]
        self.thread.submit(self._give_pulse, port, duration)

    def check_events(self):
        """Monitor lick events on the serial port and log corresponding activity."""
        while not self.stop_flag.is_set():
            # check if any of the channels is activated
            for channel in self.channels["Lick"]:
                if self.channels["Lick"][channel]() is True:
                    self._lick_port_activated(channel)
            time.sleep(0.05)

    def _lick_port_activated(self, port):
        """Handle activation of the lick port and log the corresponding activity."""
        resp_time = self.logger.logger_timer.elapsed_time()
        if self.resp_tmst == resp_time:
            return
        self.resp_tmst = resp_time
        self.response = self.ports[Port(type="Lick", port=port) == self.ports][0]
        self.beh.log_activity({**self.response.__dict__, "time": self.resp_tmst})

    def _give_pulse(self, port, duration):
        """Deliver a liquid pulse on the specified port for the given duration.

        Args:
            port: Port object representing the target port.
            duration: Duration of the liquid pulse in milliseconds.
        """
        # TODO: use a port and find the serial connection based on the dictionary
        self.channels["Liquid"][port](True)
        time.sleep(duration / 1000)
        self.channels["Liquid"][port](False)

    def cleanup(self):
        """Clean up resources and stop asynchronous tasks."""
        self.stop_flag.set()
        with self.thread, self.lick_thread:
            # This will wait for all thread pool tasks to complete
            sys.stdout.write(
                "\rWaiting for thread pool tasks in serial ports interface to complete"
                + "." * (int(time.time()) % 4)
            )
            sys.stdout.flush()
