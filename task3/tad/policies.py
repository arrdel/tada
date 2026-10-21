"""Registry: policy name -> (routing rule, trust provider)."""
from . import profiles, routing, trust


def make_policy(cfg, fleet):
    n, K = fleet.n, profiles.N_CLUSTERS
    name = cfg.policy
    m = cfg.committee_size
    if name == "ppai":
        return routing.PPAIArgmax(cfg, trust.NoTrust(n, K))
    if name == "majority":
        return routing.Committee(cfg, trust.NoTrust(n, K), "Majority vote (m=%d)" % m)
    if name == "beta":
        return routing.Committee(cfg, trust.BetaAgreement(n, K),
                                 "Agreement trust (m=%d)" % m)
    raise ValueError("unknown policy %r" % name)
