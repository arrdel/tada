"""Sanity tests for the simulator, the baselines, and TADA.

    cd task3 && python -m pytest -q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tad import peers                                    # noqa: E402
from tad.simulator import Config, run                    # noqa: E402
from tad.trust import one_coin_posterior                 # noqa: E402

FAST = dict(n_queries=2000)


def test_one_coin_single_answer_confidence_is_accuracy():
    rng = np.random.default_rng(0)
    _, conf = one_coin_posterior([2], [0.7], rng)
    assert abs(conf - 0.7) < 1e-9


def test_one_coin_ties_are_broken_randomly():
    rng = np.random.default_rng(0)
    picks = {one_coin_posterior([0, 1], [0.6, 0.6], rng)[0] for _ in range(50)}
    assert picks == {0, 1}


def test_colluders_share_one_wrong_answer():
    rng = np.random.default_rng(0)
    fleet = peers.build_fleet(20, 0.5, peers.COLLUDE, rng)
    ids = np.flatnonzero(fleet.adversary)
    answers = {fleet.answer(int(p), 0, qid=123) for p in ids}
    assert len(answers) == 1 and peers.CORRECT not in answers


def test_ppai_matches_single_best_honest_peer_without_attack():
    r = run(Config(policy="ppai", adversary_fraction=0.0, **FAST))
    assert abs(r["accuracy"] - r["oracle_single"]) < 0.05


def test_simulator_is_stable_below_capacity():
    r = run(Config(policy="majority", adversary_fraction=0.0, arrival_rate=50, **FAST))
    assert r["latency_p95_ms"] < 1000


def test_trust_blind_routing_is_captured_by_liars():
    r = run(Config(policy="ppai", adversary_type=peers.SKILL_LIAR,
                   adversary_fraction=0.2, **FAST))
    assert r["adversary_exposure"] > 0.9 and r["accuracy"] < 0.3


def test_dawid_skene_is_captured_by_collusion():
    r = run(Config(policy="dawid_skene", adversary_type=peers.COLLUDE,
                   adversary_fraction=0.3, **FAST))
    assert r["accuracy"] < 0.2


def test_tada_resists_collusion_and_needs_audits():
    full = run(Config(policy="tada", adversary_type=peers.COLLUDE,
                      adversary_fraction=0.3, **FAST))
    blind = run(Config(policy="tada", adversary_type=peers.COLLUDE,
                       adversary_fraction=0.3, p_audit=0.0, **FAST))
    assert full["accuracy"] > 0.65
    assert full["accuracy"] > blind["accuracy"] + 0.15


def test_tada_uses_fewer_probes_than_a_full_committee():
    r = run(Config(policy="tada", adversary_fraction=0.0, **FAST))
    assert r["probes_per_query"] < 5.0
