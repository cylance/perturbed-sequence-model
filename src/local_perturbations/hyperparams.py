class LocalPerturbationsHyperParams:
    def __init__(self, K, W, Kstars):
        self.K = K
        self.W = W
        self.Kstars = Kstars

    @classmethod
    def from_model(cls, local_perturbations_model):
        return cls(
            local_perturbations_model.K,
            local_perturbations_model.W,
            local_perturbations_model.Kstars,
        )

    @property
    def n_Kstars(self):
        return len(self.Kstars)
