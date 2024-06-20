import time

import datajoint as dj

from core.Experiment import ExperimentClass, State, experiment


@experiment.schema
class Condition(dj.Manual):
    class Approach(dj.Part):
        definition = """
        # Approach experiment conditions
        -> Condition
        ---
        trial_selection='staircase' : enum('fixed','random','staircase','biased') 
        max_reward=3000             : smallint
        min_reward=500              : smallint
        bias_window=5               : smallint
        staircase_window=20         : smallint
        stair_up=0.7                : float
        stair_down=0.55             : float
        noresponse_intertrial=1     : tinyint(1)
        incremental_punishment=1    : tinyint(1)

        init_ready                  : int
        trial_ready                 : int
        difficulty                  : int   
        trial_duration              : int
        intertrial_duration         : int
        reward_duration             : int
        punish_duration             : int
        abort_duration              : int
        """


class Experiment(State, ExperimentClass):
    cond_tables = ["Approach"]
    required_fields = []
    default_key = {
        "trial_selection": "staircase",
        "max_reward": 1500,
        "min_reward": 500,
        "bias_window": 5,
        "staircase_window": 20,
        "stair_up": 0.7,
        "stair_down": 0.55,
        "noresponse_intertrial": True,
        "incremental_punishment": True,
        "init_ready": 0,
        "trial_ready": 0,
        "difficulty": 0,
        "trial_duration": 1000,
        "intertrial_duration": 1000,
        "reward_duration": 500,
        "punish_duration": 1000,
        "abort_duration": 0,
    }

    def entry(self):
        """
        updates stateMachine from Database entry - override for timing critical transitions
        """
        self.logger.curr_state = self.name()
        self.start_time = self.logger.log("Trial.StateOnset", {"state": self.name()})
        self.resp_ready = False
        self.state_timer.start()


class Entry(Experiment):
    def entry(self):
        pass

    def next(self):
        return "PreTrial"


class PreTrial(Experiment):
    def entry(self):
        self.prepare_trial()
        if not self.is_stopped():
            self.beh.prepare(self.curr_cond)
            self.stim.prepare(self.curr_cond)
            super().entry()
            self.logger.ping()
            self.stim.start_stim()

    def next(self):
        if self.is_stopped():
            return "Exit"
        elif self.beh.is_sleep_time():
            return 'Offtime'
        elif self.beh.is_ready_start():
            return "Trial"
        else:
            return "PreTrial"


class Trial(Experiment):
    def entry(self):
        super().entry()
        self.stim.start()

    def run(self):
        self.stim.present()
        self.logger.ping()
        self.response = self.beh.get_response(self.start_time)

    def next(self):
        if self.response and self.beh.is_correct():
            return "Reward"
        elif self.response and not self.beh.is_correct():
            return "Punish"
        elif self.state_timer.elapsed_time() > self.stim.curr_cond["trial_duration"]:
            return "Abort"
        elif self.is_stopped():
            return "Exit"
        else:
            return "Trial"

    def exit(self):
        self.stim.stop()


class Abort(Experiment):
    def entry(self):
        super().entry()
        self.beh.update_history()
        self.logger.log("Trial.Aborted")

    def next(self):
        if self.state_timer.elapsed_time() >= self.curr_cond["abort_duration"]:
            return "InterTrial"
        elif self.is_stopped():
            return "Exit"
        else:
            return "Abort"


class Reward(Experiment):
    def entry(self):
        super().entry()
        self.stim.reward_stim()

    def run(self):
        self.rewarded = self.beh.reward(self.start_time)

    def next(self):
        if self.rewarded:
            return "InterTrial"
        elif self.state_timer.elapsed_time() >= self.curr_cond["reward_duration"]:
            self.beh.update_history(reward=0)
            return "InterTrial"
        elif self.is_stopped():
            return "Exit"
        else:
            return "Reward"


class Punish(Experiment):
    def entry(self):
        self.beh.punish()
        super().entry()
        self.punish_period = self.curr_cond["punish_duration"]
        if self.params.get("incremental_punishment"):
            self.punish_period *= self.beh.get_false_history()

    def run(self):
        self.stim.punish_stim()

    def next(self):
        if self.state_timer.elapsed_time() >= self.punish_period:
            return "InterTrial"
        elif self.is_stopped():
            return "Exit"
        else:
            return "Punish"

    def exit(self):
        self.stim.fill()


class InterTrial(Experiment):
    def run(self):
        if self.beh.is_licking() and self.params.get("noresponse_intertrial"):
            self.state_timer.start()
        self.logger.ping()

    def next(self):
        if self.is_stopped() or self.beh.is_sleep_time() or self.beh.is_hydrated():
            return "Exit"
        # elif self.beh.is_sleep_time() and not self.beh.is_hydrated(self.params['min_reward']):
        #     return 'Hydrate'
        # elif self.beh.is_sleep_time() or self.beh.is_hydrated():
        #     return 'Offtime'
        elif self.state_timer.elapsed_time() >= self.curr_cond["intertrial_duration"]:
            return "PreTrial"
        else:
            return "InterTrial"

    def exit(self):
        self.stim.fill()

# class Offtime(Experiment):
#     def entry(self):
#         super().entry()
#         self.stim.fill([0, 0, 0])
#         self.release()

#     def run(self):
#         if self.logger.setup_status not in ['sleeping', 'wakeup'] and self.beh.is_sleep_time():
#             self.logger.update_setup_info({'status': 'sleeping'})
#         self.logger.ping()
#         time.sleep(1)

#     def next(self):
#         if self.is_stopped():  # if wake up then update session
#             return 'Exit'
#         elif not self.beh.is_sleep_time():
#             return 'PreTrial'
#         else:
#             return 'Offtime'

#     def exit(self):
#         if self.logger.setup_status in ['wakeup', 'sleeping']:
#             self.logger.update_setup_info({'status': 'running'})


class Exit(Experiment):
    def entry(self):
        self.release()
        self.beh.exit()

    def run(self):
        self.stop()
