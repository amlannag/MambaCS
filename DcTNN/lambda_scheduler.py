import math


class LambdaScheduler:
    def __init__(self, schedule, start, end, total_epochs):
        self.schedule = schedule
        self.start = start
        self.end = end
        self.total_epochs = total_epochs

    def get_lambda(self, epoch):
        if self.schedule == "constant":
            return self.start
        t = epoch / max(self.total_epochs - 1, 1)   # 0.0 → 1.0
        if self.schedule == "linear":
            return self.start + t * (self.end - self.start)
        if self.schedule == "cosine":
            return self.end + 0.5 * (self.start - self.end) * (1 + math.cos(math.pi * t))
        raise ValueError(f"Unknown lambda schedule: '{self.schedule}'. "
                         f"Choose from: 'cosine', 'linear', 'constant'")
