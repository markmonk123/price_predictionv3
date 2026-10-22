"""Sample-weight routing for voting ensembles containing sklearn pipelines."""
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline


class SampleWeightedPipeline(Pipeline):
    """Accept VotingClassifier's top-level weights at the final pipeline step."""

    def fit(self, X, y=None, sample_weight=None, **params):
        if sample_weight is not None:
            params[f"{self.steps[-1][0]}__sample_weight"] = sample_weight
        return super().fit(X, y, **params)


class SampleWeightedVotingClassifier(VotingClassifier):
    """Expose weights explicitly so calibration forwards them to training folds."""

    def fit(self, X, y, sample_weight=None, **params):
        if sample_weight is not None:
            params["sample_weight"] = sample_weight
        return super().fit(X, y, **params)


def make_weighted_voting_classifier(estimators, **kwargs):
    """Clone members and use sample weights as the single class-balancing source."""
    members = []
    for name, estimator in estimators:
        member = clone(estimator)
        # Constructor class weights multiply sample weights; avoid double balancing.
        weight_params = {
            key: None for key in member.get_params(deep=True)
            if key == "class_weight" or key.endswith("__class_weight")
        }
        member.set_params(**weight_params)
        if isinstance(member, Pipeline):
            member = SampleWeightedPipeline(
                member.steps, memory=member.memory, verbose=member.verbose,
            )
        members.append((name, member))
    return SampleWeightedVotingClassifier(members, **kwargs)
