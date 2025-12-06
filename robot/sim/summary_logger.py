import numpy as np

class SummaryLogger:
    def __init__(self, log_freq, logs, writer, record_results):
        self.log_freq = log_freq
        self.logs = logs
        self.writer = writer
        self.record_results = record_results
        self.recent_vals = {}

    def log(self, key, val, global_step):
        if key not in self.logs:
            self.logs[key] = []
        if key not in self.recent_vals:
            self.recent_vals[key] = []

        self.recent_vals[key].append(val)
        if (global_step + 1) % self.log_freq == 0:
            mean_val = np.mean(self.recent_vals[key])
            if not self.record_results:
                self.writer.add_scalar(f"charts/{key}", mean_val, global_step)
            self.logs[key].append({'t': global_step, key: mean_val})
            self.recent_vals[key] = []
