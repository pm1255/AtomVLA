"""Tiny TensorBoard scalar writer with no PyTorch dependency."""
import time
from pathlib import Path
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

class ScalarWriter:
    def __init__(self, directory):
        self._writer = EventFileWriter(str(Path(directory)))
    def add_scalar(self, tag, value, step):
        self._writer.add_event(Event(wall_time=time.time(), step=int(step), summary=Summary(value=[Summary.Value(tag=tag, simple_value=float(value))])))
    def flush(self): self._writer.flush()
    def close(self): self._writer.close()
