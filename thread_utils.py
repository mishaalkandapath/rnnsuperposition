import threading

class ThreadSafeLossList:
    def __init__(self, num_models):
        self.data = [[] for _ in range(num_models)]
        self.lock = threading.Lock()

    def add_item(self, item, model_idx):
        with self.lock:
            self.data[model_idx].append(item)

    def get_item(self, index):
        with self.lock:
            return list(self.data[index])

    def remove_item(self, item):
        # with self.lock:
        #     if item in self.data:
        #         self.data.remove(item)
        raise NotImplementedError