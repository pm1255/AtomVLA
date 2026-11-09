"""Conservative, source-grounded semantic candidates.

This module does not invent a missing action.  It only emits an atom when the
recorded state contains the event sequence required by the atom schema and the
task text/scene provide a symbolic binding.  The candidates are still marked
as automatic candidates; they are never upgraded to human-verified ground
truth.
"""
import re
import numpy as np

from auto_only_policy import consensus, materialize


_STOP = {
    "the", "a", "an", "on", "onto", "to", "in", "into", "at", "from",
    "and", "then", "it", "itself", "up", "down", "of", "with", "for",
}


def _words(value):
    return {x for x in re.findall(r"[a-z0-9]+", str(value).lower()) if x not in _STOP}


def _safe(value, fallback="task_goal"):
    value = re.sub(r"[^A-Za-z0-9_./:-]+", "_", str(value)).strip("_")
    return value or fallback


def _binding(instruction, scene):
    """Return a conservative object/target binding from task text and scene."""
    text = str(instruction or "")
    words = _words(text)
    objects = list((scene or {}).get("objects", {}))
    articulations = list((scene or {}).get("articulations", {}))

    def score(name):
        tokens = _words(name.replace("_joint0", ""))
        # Singular/plural and common LIBERO suffixes should not change binding.
        return len(tokens & words) + 0.2 * len(tokens) if tokens else 0.0

    ranked = sorted(((score(x), x) for x in objects), reverse=True)
    obj = ranked[0][1] if ranked and ranked[0][0] > 0 else None

    # Explicit source names are preferred over an invented natural-language ID.
    target = None
    low = text.lower()
    for phrase in ("inside", "into", "onto", "on top of", "on", "to", "at"):
        m = re.search(r"\b" + re.escape(phrase) + r"\s+(?:the\s+)?([a-z][a-z0-9 _-]*)", low)
        if not m:
            continue
        candidate_words = _words(m.group(1))
        choices = sorted(
            ((len(candidate_words & _words(x.replace("_joint0", ""))), x)
             for x in objects + articulations), reverse=True
        )
        if choices and choices[0][0] > 0:
            target = choices[0][1]
            break
        if candidate_words:
            # Keep a task noun such as stove/table/shelf as a stable symbol.
            target = _safe("_".join(sorted(candidate_words)))
            break
    if target is None:
        # A few common LIBERO articulated targets are named through the task
        # text while the XML uses a button/slide joint ID.
        for alias in ("stove", "drawer", "door", "cabinet", "microwave", "oven"):
            if alias in low:
                target = alias
                break
    return obj, target or "task_goal"


def _events(events, arm):
    return sorted((x for x in events.get("events", []) if x.get("arm") == arm),
                  key=lambda x: (int(x["start"]), int(x["end"])))


def _candidate(verb, obj, target, arm, start, end, event_rows, observation,
               channels, physical=False):
    roles = [role for role, _ in event_rows]
    atom = {
        "kind": "primitive", "verb": verb, "start": int(start), "end": int(end),
        "arguments": {
            "object": _safe(obj, "gripper"), "target": _safe(target),
            "arm": arm, "completion": roles[-1],
        },
        "events": [{"role": r, "row": int(t)} for r, t in event_rows],
        "quality": "silver", "observation": observation,
        "evidence": {"method": "source_state_task_contract",
                     "channels": list(channels), "physical_evidence": bool(physical)},
        "review_reasons": ["automatic_source_contract_candidate"],
    }
    # A candidate generated once cannot vote for itself under multiple names.
    # Keep its provenance, but require an independent verifier before training.
    atom['auto_policy'] = 'atomvla_source_candidate_v2'
    atom['auto_decision'] = {'status': 'auto_unresolved',
                             'reason': 'independent_physical_validation_required'}
    atom['train_eligible'] = False
    atom['verified_by_human'] = False
    return atom



def candidates(pose, grip, evidence, scene, instruction, source):
    """Build non-overlapping state/task candidates for one episode.

    The function is intentionally incomplete for sources without object state.
    Returning no candidate means quarantine, never a guessed ``move``.
    """
    del pose, grip  # event evidence is already derived from the validated arrays
    obj, target = _binding(instruction, scene)
    if obj is None and source == "libero":
        return [], ["libero_object_binding_unresolved"]
    text = str(instruction or "").lower()
    atoms = []
    issues = []
    for arm in evidence.get("physical_arms", []):
        evs = _events(evidence, arm)
        closes = [x for x in evs if x["kind"] == "gripper_closing"]
        opens = [x for x in evs if x["kind"] == "gripper_opening"]
        motions = [x for x in evs if x["kind"] == "object_motion"]
        articulations = [x for x in evs if x["kind"] == "articulation_change"]

        # Articulated state changes are only bound when the task explicitly
        # requests the corresponding interaction.  A changing joint alone is
        # never silently labelled as a press/toggle.
        if articulations and re.search(r"\b(turn|toggle|switch|press|activate|deactivate|on|off)\b", text):
            for x in articulations:
                joint = x["evidence"].get("joint") or x["evidence"].get("body")
                if not joint:
                    continue
                atoms.append(_candidate(
                    "toggle", joint, joint, arm, x["start"], x["end"],
                    [("changed", x["end"])],
                    "The recorded articulated joint changes during the requested toggle.",
                    ("trajectory_state", "task_contract"), physical=False))

        # A pick requires closure followed by measured object motion with a
        # positive vertical displacement.  This prevents a gripper closure
        # near an empty table from becoming a pick.
        for close in closes:
            later = [m for m in motions if m["start"] <= close["end"] < m["end"]
                     and float(m["evidence"].get("displacement_xyz", [0, 0, 0])[2]) >= .015]
            if not later or obj is None:
                continue
            motion = min(later, key=lambda x: x["end"])
            if motion["evidence"].get("object") not in (None, obj):
                continue
            atoms.append(_candidate(
                "pick", obj, "above_support", arm, close["start"], motion["end"],
                [("closed", close["end"]), ("lifted", motion["end"])],
                "Measured gripper closure is followed by object motion with vertical lift.",
                ("trajectory_state", "task_contract"), physical=True))
            # Pair the first subsequent opening with the same object motion for
            # a place candidate.  Support must precede release strictly.
            after = [o for o in opens if o["start"] >= motion["end"]]
            if after:
                opening = min(after, key=lambda x: x["start"])
                support = [m for m in motions if m["start"] >= motion["end"]
                           and m["end"] <= opening["end"]]
                support_row = max((m["end"] for m in support), default=opening["start"])
                if motion["end"] < support_row < opening["end"]:
                    atoms.append(_candidate(
                        "place", obj, target, arm, motion["end"], opening["end"],
                        [("supported", support_row), ("released", opening["end"])],
                        "Measured object support is followed by gripper opening.",
                        ("trajectory_state", "task_contract"), physical=True))

        if source == "libero" and not atoms and not (closes or opens or articulations):
            issues.append("no_contract_matching_state_event")
    # Do not return overlapping candidates.  A repeated closure is retained
    # only when the intervals are disjoint; ambiguous repeated-object traces
    # stay quarantined for the visual channel.
    kept = []
    for atom in sorted(atoms, key=lambda x: (x["arguments"]["arm"], x["start"], x["end"], x["verb"])):
        arm = atom["arguments"]["arm"]
        if any(a["arguments"]["arm"] == arm and max(a["start"], atom["start"]) < min(a["end"], atom["end"])
               for a in kept):
            issues.append("overlapping_source_candidates")
            continue
        kept.append(atom)
    return kept, sorted(set(issues))


def canonicalize_prediction(raw, window, tokens, evidence, instruction, scene, source):
    """Normalize model event rows to measured rows before consensus.

    Models may call the same object ``bottle``/``spray_bottle`` and choose a
    nearby sampled frame.  Symbolic binding and event rows come from the source
    evidence; the model supplies only the operation hypothesis.
    """
    import json
    d = raw if isinstance(raw, dict) else json.loads(raw)
    if not isinstance(d, dict):
        return d
    verb = d.get("verb")
    obj, target = _binding(instruction, scene)
    arm = window["arm"]; s, e = window["start"], window["end"]
    rel = [x for x in evidence.get("events", [])
           if x.get("arm") == arm and max(s, x["start"]) < min(e, x["end"])]
    closes = [x for x in rel if x["kind"] == "gripper_closing"]
    opens = [x for x in rel if x["kind"] == "gripper_opening"]
    motions = [x for x in rel if x["kind"] == "object_motion"]
    def token(row):
        row = int(max(s + 1, min(e, row)))
        candidates = sorted(((abs(int(v) - row), k, int(v)) for k, v in tokens.items()))
        return candidates[0][1]
    if verb == "pick" and closes and (obj is not None or source != "libero"):
        close = max(closes, key=lambda x: x["end"])
        lift = max((m["end"] for m in motions if m["end"] > close["end"]), default=e)
        d.update(object=obj or d.get("object"), target="above_support",
                 events={"closed": token(close["end"]), "lifted": token(lift)})
    elif verb == "place" and opens and (obj is not None or source != "libero"):
        opening = min(opens, key=lambda x: x["start"])
        support = max((m["end"] for m in motions if m["end"] < opening["end"]),
                      default=opening["start"])
        if support > s:
            d.update(object=obj or d.get("object"), target=target,
                     events={"supported": token(support), "released": token(opening["end"])})
    elif verb in ("move", "approach"):
        d.update(object=obj or d.get("object", "gripper"), target=target,
                 events={"arrived": token(e)})
    return d
