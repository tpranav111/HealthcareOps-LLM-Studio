import queue
import threading
import time


class BatchItem:
    def __init__(self, messages, params):
        self.messages = messages
        self.params = params
        self.event = threading.Event()
        self.result = None
        self.error = None


class RequestBatcher:
    def __init__(self, runner, max_batch_size=4, max_wait_ms=50):
        self.runner = runner
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, messages, params):
        item = BatchItem(messages, params)
        self._queue.put(item)
        item.event.wait()
        if item.error:
            raise item.error
        return item.result

    def _loop(self):
        while not self._stop.is_set():
            batch = []
            start = time.time()
            while len(batch) < self.max_batch_size:
                timeout = max(0.0, (self.max_wait_ms / 1000.0) - (time.time() - start))
                try:
                    item = self._queue.get(timeout=timeout)
                    batch.append(item)
                except queue.Empty:
                    break
            if not batch:
                continue
            try:
                messages_list = [item.messages for item in batch]
                params = batch[0].params
                outputs = self.runner.generate_batch(
                    messages_list,
                    max_new_tokens=params["max_new_tokens"],
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                )
                for item, output in zip(batch, outputs):
                    item.result = output
                    item.event.set()
            except Exception as exc:
                for item in batch:
                    item.error = exc
                    item.event.set()

    def shutdown(self):
        self._stop.set()
