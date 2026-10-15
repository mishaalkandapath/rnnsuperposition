"""Refined explanations of the RL features that matter and need more than a firing rule.

Each explanation is a short verbal statement plus what it is scored by:
  firing+modifier   a firing predicate (F1 >= 0.8 on test) and at most one small magnitude modifier on top
  firing~+modifier  the same, with a firing predicate just under 0.8 (recall capped by window-first steps,
                    whose previous trial lies outside the window)
  magnitude         no clean firing boundary (weak, patchy background firing); a discrete magnitude
                    explanation: size fixed per condition, or off / low / medium / high by a stated rule
Modifiers are one of: a few levels by one cue, a fixed ratio between two cases, steady growth (or fading)
from an onset. Every explanation is re-scored here: firing F1 on the full dev / test splits
(circuit.rl_task.autointerp score) and R^2 beyond the phase x episode-decile timing baseline of the
magnitude terms (circuit.rl_task.autointerp_magnitude, observed-only). `{P}` in the terms is the predicate.
"The clock" in the words is one shared late-episode profile (CLOCK, given once to every simulator): the shape of
the value head's late fall, which the late-onset features mirror (correlation 0.81-1.00 for the candidate features
active from trial 50 on).
For firing explanations the terms are also scored as the words read ("gated"): zero outside the predicate,
the fitted terms inside it, with no timing baseline added, and still compared with the timing baseline
(1 - SSE / SSE(timing only)). The standard fit adds the timing baseline everywhere, which predicts a
silent feature at its phase-decile average.

    python -m circuit.rl_task.refined_explanations --out rl_features/rl_refined
"""
import argparse
import json
import os

import numpy as np

from circuit.rl_task.autointerp import Interp
from circuit.rl_task.autointerp import score as firing_score
from circuit.rl_task.autointerp_magnitude import Mag

FG = {"reset": "reset", "update": "update", "candidate": "hidden"}
CLOCK = ("The clock: some descriptions say a value follows the clock (from some trial). The clock is the same "
         "everywhere: as a share of that value's full strength it is about 0.1 at trial 60, 0.2 at trial 70, 0.35 at "
         "trial 80, 0.5 at trial 85, 0.75 at trial 90 and full from trial 95 on (and 0 before any stated onset).")
D1_RAMP = "np.maximum(ep_trial - np.where(prev_action == 2, 70, 80), 0)"

SPECS = [
    # ---- firing rule + one small modifier
    dict(gate="update", id=18, kind="firing+modifier",
         words="Fires at D1 after an a1 choice that reached S1 (rewarded or not), and after any other trial that did "
               "not end in S2 (a2->S1, or a window's first trial) while S1 is believed the better state; stronger "
               "after a1->S1. Silent after every S2 visit.",
         predicate="((phase == D1) & (prev_action == 1) & (prev_state == 1)) | "
                   "((phase == D1) & (prev_state != 2) & (belief > 0.5))",
         terms="{P} ;; ({P}) * (prev_action == 1) * (prev_state == 1)", modifier="ratio: a1->S1 stronger"),
    dict(gate="candidate", id=208, kind="firing+modifier",
         words="Fires at D1 after an unrewarded trial that did not end in S1 (an unrewarded S2 visit, or a window's "
               "first trial); about three times stronger when that S2 visit came by the rare a1->S2 route.",
         predicate="(phase == D1) & (in_rew == 0) & (prev_state != 1)",
         terms="{P} ;; ({P}) * (prev_action == 1) * (prev_state == 2)", modifier="ratio: rare a1->S2 stronger"),
    dict(gate="candidate", id=11, kind="firing+modifier",
         words="Fires at D1 when no reward is shown and the last trial did not end in S1 (an unrewarded S2 visit, or "
               "a window's first trial), and only before trial 80; low, medium or high as the belief that S1 is good "
               "is low, middling or high.",
         predicate="(phase == D1) & (prev_state != 1) & (in_rew == 0) & (ep_trial < 80)",
         terms="{P} ;; ({P}) * (belief > 0.3) ;; ({P}) * (belief > 0.7)", modifier="levels by belief"),
    dict(gate="update", id=5, kind="firing+modifier",
         words="Fires at GO after an unrewarded common a1->S1 trial; low, medium or high as the belief that S1 is "
               "good is low, middling or high.",
         predicate="(phase == GO) & (prev_reward == 0) & (prev_action == 1) & (prev_common == 1)",
         terms="{P} ;; ({P}) * (belief > 0.3) ;; ({P}) * (belief > 0.7)", modifier="levels by belief"),
    dict(gate="reset", id=18, kind="firing+modifier",
         words="Fires at D1 after an a1 choice that went rare to S2 (rewarded or not), and also after an unrewarded "
               "S2 visit while S1 is strongly believed good; much stronger in the a1->S2 case.",
         predicate="((phase == D1) & (prev_action == 1) & (prev_state == 2)) | "
                   "((phase == D1) & (prev_state != 1) & (prev_reward == 0) & (belief > 0.85))",
         terms="{P} ;; ({P}) * (prev_action == 1) * (prev_state == 2)", modifier="ratio: a1->S2 stronger"),
    dict(gate="candidate", id=36, kind="firing+modifier",
         words="Fires at D1 after a rewarded S1 visit: from trial 70 if the visit came by the rare route (a2->S1), "
               "from trial 80 if by the common route (a1->S1). From its onset it grows steadily with each trial, at "
               "the same rate for both routes, so after a rare-route reward it is always about 10 trials ahead.",
         predicate="(phase == D1) & (prev_reward == 1) & (((prev_action == 2) & (prev_state == 1) & (ep_trial >= 70))"
                   " | ((prev_action == 1) & (prev_state == 1) & (ep_trial >= 80)))",
         terms="{P} ;; ({P}) * " + D1_RAMP, modifier="grows from onset"),
    dict(gate="update", id=150, kind="firing+modifier",
         words="Fires at GO after an unrewarded trial: from trial 40 if the agent took the rare route a1->S2, and "
               "from trial 70 after any unrewarded trial while S1 is confidently believed good. From its onset it "
               "grows steadily with each trial.",
         predicate="((phase == GO) & (prev_reward == 0) & (prev_action == 1) & (prev_state == 2) & (ep_trial >= 40))"
                   " | ((phase == GO) & (prev_reward == 0) & (belief > 0.85) & (ep_trial >= 70))",
         terms="{P} ;; ({P}) * np.where((prev_action == 1) & (prev_state == 2), np.maximum(ep_trial - 40, 0), "
               "np.maximum(ep_trial - 70, 0))", modifier="grows from onset"),
    dict(gate="candidate", id=212, kind="firing+modifier",
         words="Fires at D1 after an unrewarded trial that went a2 to S2 by the common route, but only before trial "
               "80; strongest early, weakening steadily with each trial.",
         predicate="(phase == D1) & (prev_reward == 0) & (prev_action == 2) & (prev_common == 1) & (ep_trial < 80)",
         terms="{P} ;; ({P}) * ep_trial", modifier="fades over the episode"),
    dict(gate="candidate", id=114, kind="firing~+modifier",
         words="Fires at D1 after an unrewarded common a1->S1 trial; larger when S1 is still believed good and when "
               "the trial before was also unrewarded: high with both, medium with one, low with neither.",
         predicate="(phase == D1) & (prev_reward == 0) & (prev_action == 1) & (prev_common == 1)",
         terms="{P} ;; ({P}) * (belief > 0.7) ;; ({P}) * (tv('reward', -2) == 0)",
         modifier="levels: count of two conditions"),
    dict(gate="candidate", id=8, kind="firing+modifier",
         words="Fires at D1 after a rewarded visit to S2: strong after the rare a1->S2 route, about a quarter as "
               "strong after the common a2->S2 route.",
         predicate="(phase == D1) & (in_rew == 1) & (prev_state == 2)",
         terms="{P} ;; ({P}) * (prev_action == 1)", modifier="ratio: rare a1->S2 stronger"),
    dict(gate="candidate", id=204, kind="firing+modifier",
         words="Fires at D2 after choosing a2 when it led rarely to S1 (about half strength early, rising with the "
               "clock to full), and weakly after other a2 choices from trial 40 and after common transitions from "
               "trial 80, following the clock up to about half strength.",
         predicate="(phase == D2) & (((in_action == 2) & ((in_state == 1) | (ep_trial >= 40))) | "
                   "((ep_trial >= 80) & (common == 1)))",
         terms="{P} ;; ({P}) * (in_action == 2) * (in_state == 1)", modifier="ratio: rare a2->S1 stronger"),
    # ---- firing rule just under 0.8 + one small modifier
    dict(gate="candidate", id=60, kind="firing~+modifier",
         words="Fires at D1 late in the episode (from trial 85) after an unrewarded a1 trip that landed in S2 by the "
               "rare route; from its onset it follows the clock.",
         predicate="((phase == D1) & (prev_reward == 0) & (prev_action == 1) & (prev_state == 2) & (ep_trial >= 85))"
                   " | ((phase == D1) & (prev_reward == 0) & (prev_action == -1) & (ep_trial >= 85) & (belief > 0.8))",
         terms="{P} ;; ({P}) * np.maximum(ep_trial - 85, 0)", modifier="grows from onset"),
    dict(gate="update", id=181, kind="firing~+modifier",
         words="Fires at D1 after an unrewarded S1 visit: from trial 72 if it came by the common route (a1->S1), only "
               "from trial 88 if by the rare route (a2->S1); from its onset it grows steadily with each trial.",
         predicate="((phase == D1) & (prev_reward == 0) & (prev_action == 1) & (prev_state == 1) & (ep_trial >= 72))"
                   " | ((phase == D1) & (prev_reward == 0) & (prev_action == 2) & (prev_state == 1) & (ep_trial >= 88))",
         terms="{P} ;; ({P}) * np.maximum(ep_trial - np.where(prev_action == 1, 72, 88), 0)",
         modifier="grows from onset"),
    dict(gate="candidate", id=183, kind="firing~+modifier",
         words="Fires at D1 after an unrewarded common a1->S1 trial from about trial 72, and weakly after an "
               "unrewarded rare-route trial in the last few trials; the a1->S1 response grows steadily from onset.",
         predicate="((phase == D1) & (prev_reward == 0) & (prev_action == 1) & (prev_state == 1) & (ep_trial >= 72))"
                   " | ((phase == D1) & (prev_reward == 0) & (prev_common == 0) & (ep_trial >= 97))",
         terms="{P} ;; ({P}) * np.maximum(ep_trial - 72, 0) * (prev_common == 1)", modifier="grows from onset"),
    # ---- discrete magnitude only (no clean firing boundary)
    dict(gate="candidate", id=66, kind="magnitude",
         words="At D1 after a rewarded S1 visit: high after a rare a2->S1 transition; after the common a1->S1 route, "
               "low and growing steadily with how surprising the reward was (near zero when S1 was confidently good).",
         terms="(in_rew == 1) * (prev_state == 1) * (prev_action == 2) ;; (in_rew == 1) * (prev_state == 1) * "
               "surprisal_reward"),
    dict(gate="reset", id=112, kind="magnitude",
         words="Only at D1: high after an unrewarded common a1->S1 trial, and below its usual level after a trial "
               "that reached S2.",
         terms="(phase == D1) * last_is('a1 S1 u') ;; (phase == D1) * (prev_state == 2)"),
    dict(gate="candidate", id=24, kind="magnitude",
         words="At D1 after an unrewarded S1 visit: low after a single miss, high only after two or more unrewarded "
               "trials in a row, and stronger after the common (a1) route.",
         terms="(phase == D1) * last_is('a1 S1 u') * (tv('reward', -2) == 0) ;; "
               "(phase == D1) * last_is('a2 S1 u') * (tv('reward', -2) == 0) ;; "
               "(phase == D1) * (prev_state == 1) * (prev_reward == 0)"),
    dict(gate="candidate", id=200, kind="magnitude",
         words="Only at D2, and zero at S1: on reaching S2 after a rewarded trial, most if that reward was at S1 or "
               "S2 was reached by a1.",
         terms="(in_state == 1) ;; (in_state == 2) * (prev_reward == 1) * (prev_state == S1) ;; "
               "(in_state == 2) * (action == 1) * (prev_reward == 1)"),
    dict(gate="candidate", id=215, kind="magnitude",
         words="At D1 after a rewarded trip to S2: high after the rare a1->S2 route; after a2->S2 near zero, rising "
               "late with the clock.",
         terms="(phase == D1) * last_is('a1 S2 r') ;; (phase == D1) * last_is('a2 S2 r') * np.maximum(ep_trial - 70, 0)"),
    dict(gate="candidate", id=16, kind="magnitude",
         words="Only at D1: high after an unrewarded S1 trial two back or an unrewarded a1->S1 trial just before; "
               "lower after a reward.",
         terms="(phase == D1) * (tv('state', -2) == S1) * (tv('reward', -2) == 0) ;; (phase == D1) * last_is('a1 S1 u')"
               " ;; (phase == D1) * (prev_reward == 1)"),
    dict(gate="candidate", id=250, kind="magnitude",
         words="At D1 after an unrewarded common a1->S1 trial: higher when S1 was believed good, and about a fifth "
               "as strong when the trial before was also unrewarded.",
         terms="(phase == D1) * last_is('a1 S1 u') ;; (phase == D1) * last_is('a1 S1 u') * (tv('reward', -2) == 0) ;; "
               "(phase == D1) * last_is('a1 S1 u') * (belief > 0.7)"),
    dict(gate="candidate", id=95, kind="magnitude",
         words="At D1 after a rewarded rare a1->S2 trial: high before trial 70, low after; medium after a rewarded "
               "a2->S2 trial that followed an a1 choice.",
         terms="(phase == D1) * last_is('a1 S2 r') * (ep_trial < 70) ;; (phase == D1) * last_is('a1 S2 r') * "
               "(ep_trial >= 70) ;; (phase == D1) * last_is('a2 S2 r') * (tv('action', -2) == 1)"),
    dict(gate="candidate", id=120, kind="magnitude",
         words="At D1 after a rewarded rare a2->S1 trial: high before trial 50, medium from 50 to 69, near zero "
               "from trial 70.",
         terms="(phase == D1) * last_is('a2 S1 r') * (ep_trial < 50) ;; "
               "(phase == D1) * last_is('a2 S1 r') * (ep_trial >= 50) * (ep_trial < 70)"),
    dict(gate="candidate", id=185, kind="magnitude",
         words="At D1 after a rewarded rare a2->S1 trial; larger when S2 was believed good and when the trial before "
               "was rewarded: high with both, medium with one, low with neither.",
         terms="(phase == D1) * last_is('a2 S1 r') ;; (phase == D1) * last_is('a2 S1 r') * (belief < 0.4) ;; "
               "(phase == D1) * last_is('a2 S1 r') * (tv('reward', -2) == 1)"),
]


def gated(m: Mag, terms: str):
    """Fit the terms (no timing, no intercept) on dev; zero outside them; R^2 beyond timing per split."""
    exprs, X = m.terms(terms)
    dev = m.rows["dev"]
    w = np.linalg.lstsq(X[dev], m.act[dev], rcond=None)[0]
    out = {}
    for sp in ("dev", "test"):
        rows = m.rows[sp]
        y, p, base = m.act[rows], X[rows] @ w, m.timing[rows]
        out[sp] = {"r2_beyond_timing": float(1 - ((y - p) ** 2).sum() / ((y - base) ** 2).sum()),
                   "r2": float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()), "weights": w.tolist()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--store", default="rl_features/rl_chosen/chosen_rl_v1_r128_store")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", help="gate/id items to (re)score, default all")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ip = Interp(args.store)
    for s in SPECS:
        if args.only and f"{s['gate']}/{s['id']}" not in args.only:
            continue
        g = FG[s["gate"]]
        rec = {k: v for k, v in s.items()}
        terms = s["terms"].replace("{P}", f"({s['predicate']})") if s.get("predicate") else s["terms"]
        rec["terms"] = terms
        if s.get("predicate"):
            rec["firing"] = {sp: firing_score(ip, g, s["id"], s["predicate"], sp) for sp in ("dev", "test")}
        m = Mag(args.store, g, s["id"], observed_only=True)
        rec["magnitude"] = {sp: m.evaluate(terms, split=sp) for sp in ("dev", "test")}
        if s.get("predicate"):
            rec["magnitude_onoff"] = {sp: m.evaluate(f"({s['predicate']})", split=sp) for sp in ("dev", "test")}
            rec["gated"], rec["gated_onoff"] = gated(m, terms), gated(m, f"({s['predicate']})")
        for src, d in (("firing_record", "rl_features/rl_autointerp"), ("magnitude_record", "rl_features/rl_magnitude")):
            p = os.path.join(d, f"{g}_{s['id']}.json")
            if os.path.exists(p):
                r = json.load(open(p))
                rec[src] = {"path": p, "label": r.get("final_label") or r.get("label"),
                            "test_f1" if src == "firing_record" else "test_r2_beyond_timing":
                                r["test_scores"]["f1" if src == "firing_record" else "r2_beyond_timing"]}
        json.dump(rec, open(os.path.join(args.out, f"{g}_{s['id']}.json"), "w"), indent=1)
        f = rec.get("firing", {}).get("test")
        print(f"{s['gate']:9s} {s['id']:4d} {s['kind']:17s} "
              + (f"F1 {f['f1']:.3f} (P {f['precision']:.2f} R {f['recall']:.2f}; dev {rec['firing']['dev']['f1']:.3f}) "
                 f"on/off {rec['magnitude_onoff']['test']['r2_beyond_timing']:.2f} -> " if f else " " * 50)
              + f"R2b {rec['magnitude']['test']['r2_beyond_timing']:.3f} (dev {rec['magnitude']['dev']['r2_beyond_timing']:.3f})"
              + (f" | gated on/off {rec['gated_onoff']['test']['r2_beyond_timing']:.2f} -> "
                 f"{rec['gated']['test']['r2_beyond_timing']:.3f}" if f else ""), flush=True)


if __name__ == "__main__":
    main()
