class AverageMeter():
    def __init__(self):
        self.reset()
    def reset(self):
        self.count = 0
        self.average = 0
        self.sum = 0
        self.val = 0
    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count