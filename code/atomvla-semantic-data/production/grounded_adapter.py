"""Source adapters for measured event evidence, independent of text recognition."""
try:
 import h5py
except ModuleNotFoundError:
 h5py=None
import numpy as np
from libero_state_evidence import extract
from trajectory_events import build


def evidence_for(item,pose,grip):
 source=item['source'];m=item['meta'];scene=None;issues=[]
 fps=float(m.get('fps',0))
 if source=='libero':
  if h5py is None:raise RuntimeError('h5py is required for LIBERO state evidence')
  fps=fps or 20.
  with h5py.File(m['source_hdf5'],'r') as f:
   g=f['data'][m['source_group']]
   try:scene=extract(m['model_file'],g['states'][()],g['obs/joint_states'][()],g['obs/gripper_states'][()])
   except ValueError as exc:issues.append('scene_state_unavailable: '+str(exc))
  units='aperture_metres'
 elif source=='droid':
  if fps<=0:raise ValueError('DROID source frame rate required')
  units='normalized_open_positive'
 elif source=='robotwin':
  # No calibrated timestamps in the current archive adapter.  Infer only the
  # gripper *unit family* from the recorded numeric range; the assumption is
  # retained in the receipt and never presented as a control command.
  fps=fps or 20.;issues.append('robotwin_timebase_assumed_20Hz_requires_collector_check')
  g=np.asarray(grip,float)
  lo,hi=float(np.nanmin(g)),float(np.nanmax(g))
  if lo >= -.1 and hi <= 1.1 and hi-lo >= .08:
   units='normalized_open_positive';issues.append('robotwin_gripper_units_inferred_normalized_range')
  elif lo >= -.01 and hi <= .2 and hi-lo >= .006:
   units='aperture_metres';issues.append('robotwin_gripper_units_inferred_aperture_range')
  else:
   units='unknown';issues.append('robotwin_gripper_units_unresolved')
 else:raise ValueError('excluded source')
 events=build(pose,grip,np.arange(len(pose))/fps,units,scene)
 events.update(source=source,issues=issues,fps=fps,timebase_status='assumed' if source=='robotwin' else 'source_metadata',scene_alignment=scene['alignment'] if scene else None)
 return events,scene
