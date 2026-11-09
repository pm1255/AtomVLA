"""TensorBoard event writer without TensorFlow/PyTorch dependency."""
import json
import time
from pathlib import Path
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary


class Monitor:
    def __init__(self, root, run):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.writer = EventFileWriter(str(self.root / 'tensorboard' / run))
        self.run = run

    def write(self, step, **metrics):
        self.writer.add_event(Event(wall_time=time.time(), step=int(step), summary=Summary(
            value=[Summary.Value(tag=k, simple_value=float(v)) for k,v in metrics.items()])))
        self.writer.flush()
        status = self.root / (self.run.replace('/', '_') + '_status.json')
        status.write_text(json.dumps({'updated_unix': time.time(), 'step': step, **metrics}, indent=2))

    def close(self):
        self.writer.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    for backend in ['libero', 'robotwin']:
        m = Monitor(args.root, backend)
        m.write(0, **{'status/waiting_for_environment': 1,
                     'selection/completed_episodes': 0, 'evaluation/completed_episodes': 0,
                     'evaluation/running': 0})
        m.close()
