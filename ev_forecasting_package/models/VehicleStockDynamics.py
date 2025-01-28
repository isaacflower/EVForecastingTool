import pymc as pm

class VehicleStockDynamicsModel:
    """
    A class to encapsulate the Vehicle Stock Dynamics Bayesian Model in PyMC.

    Parameters
    ----------
    A_v : np.array or float
        Number of vehicle additions (National level).
    R_v : np.array or float
        Number of vehicle removals (National level).
    total_vehicles : np.array or float
        Total number of vehicles in the population (National level).
    V_l : np.array or float
        Total number of vehicles in the population (Local level).
    delta_V_l_obs : np.array
        Observed change in vehicle stock (Local level).
    delta_V_ev_l_obs : np.array
        Observed change in EV stock (Local level).
    delta_V_bev_l_obs : np.array
        Observed change in BEV stock (Local level).
    delta_V_phev_l_obs : np.array
        Observed change in PHEV stock (Local level).
    T : int
        Number of historical timesteps.
    L : int
        Number of local areas
    V_ev_l : np.array
        Number of EVs in the population (Local level).
    V_bev_l : np.array
        Number of BEVs in the population (Local level).
    V_phev_l : np.array
        Number of PHEVs in the population (Local level).
    """

    def __init__(
        self,
        A_v,
        R_v,
        total_vehicles,
        V_l,
        delta_V_l_obs,
        delta_V_ev_l_obs,
        delta_V_bev_l_obs,
        delta_V_phev_l_obs,
        V_ev_l,
        V_bev_l,
        V_phev_l,
        T,
        L
    ):
        # Store the data and hyperparams.
        self.A_v = A_v
        self.R_v = R_v
        self.total_vehicles = total_vehicles
        self.V_l = V_l
        self.delta_V_l_obs = delta_V_l_obs
        self.delta_V_ev_l_obs = delta_V_ev_l_obs
        self.delta_V_bev_l_obs = delta_V_bev_l_obs
        self.delta_V_phev_l_obs = delta_V_phev_l_obs
        self.V_ev_l = V_ev_l
        self.V_bev_l = V_bev_l
        self.V_phev_l = V_phev_l
        self.T = T
        self.L = L

        # Internal placeholders
        self.model = None
        self.trace = None

    def build_model(self):
        """
        Define the PyMC model using the stored data.
        """
        with pm.Model() as model:
            # Hyperparameters for Beta distributions - General vehicles
            p_A = pm.Beta(
                'p_A',
                alpha=self.A_v + 1, 
                beta=self.total_vehicles - self.A_v + 1,
                shape=self.T
            )

            p_R = pm.Beta(
                'p_R',
                alpha=self.R_v + 1, 
                beta=self.total_vehicles - self.R_v + 1,
                shape=self.T
            )

            # Latent counts for general vehicles
            A_v_l = pm.Binomial(
                'A_vehicles', 
                n=self.V_l, 
                p=p_A, 
                shape=(self.L, self.T)
            )
            R_v_l = pm.Binomial(
                'R_vehicles', 
                n=self.V_l, 
                p=p_R, 
                shape=(self.L, self.T)
            )

            # Calculate EV Share of Vehicle Additions
            a = pm.Beta(
                'a', 
                alpha=1, 
                beta=1, 
                shape=(self.L, self.T)
            )

            A_ev_l = pm.Deterministic('A_ev', a * A_v_l)
            r = pm.Deterministic('r', R_v_l / self.V_l)
            R_ev_l = pm.Deterministic('R_ev', r * self.V_ev_l)

            # Calculate BEV and PHEV Share of Vehicle Additions
            bev_ev_a_share = pm.Beta(
                'b', 
                alpha=1, 
                beta=1, 
                shape=(self.L, self.T)
            )
            phev_ev_a_share = pm.Deterministic(
                '1 - b', 
                1 - bev_ev_a_share
            )

            A_bev_l = pm.Deterministic('A_bev', bev_ev_a_share * A_ev_l)
            A_phev_l = pm.Deterministic('A_phev', phev_ev_a_share * A_ev_l)

            R_bev_l = pm.Deterministic('R_bev', r * self.V_bev_l)
            R_phev_l = pm.Deterministic('R_phev', r * self.V_phev_l)
            
            # Incorporate observed net changes - General vehicles
            delta_V_l_mean = pm.Deterministic('A_vehicles - R_vehicles', A_v_l - R_v_l)
            pm.Normal(
                'delta_N_vehicles',
                mu=delta_V_l_mean,
                sigma=1e-1,
                observed=self.delta_V_l_obs
            )

            # Incorporate observed net changes - EVs
            delta_V_ev_l_mean = pm.Deterministic('A_ev - R_ev', A_ev_l - R_ev_l)
            pm.Normal(
                'delta_N_ev',
                mu=delta_V_ev_l_mean,
                sigma=1e-1,
                observed=self.delta_V_ev_l_obs
            )

            # Incorporate observed net changes - BEVs
            delta_V_bev_l_mean = pm.Deterministic('A_bev - R_bev', A_bev_l - R_bev_l)
            pm.Normal(
                'delta_N_bev',
                mu=delta_V_bev_l_mean,
                sigma=1e-1,
                observed=self.delta_V_bev_l_obs
            )

            # Incorporate observed net changes - PHEVs
            delta_V_phev_l_mean = pm.Deterministic('A_phev - R_phev', A_phev_l - R_phev_l)
            pm.Normal(
                'delta_N_phev',
                mu=delta_V_phev_l_mean,
                sigma=1e-1,
                observed=self.delta_V_phev_l_obs
            )
            
        self.model = model
        return model

    def fit(self, draws=1000, tune=500, chains=4, cores=4, random_seed=42):
        """
        Sample from the posterior using MCMC. 
        """
        if self.model is None:
            self.build_model()
        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                random_seed=random_seed,
                return_inferencedata=True
            )
        return self.trace
    

def main():
    print("Running Vehicle Stock Dynamics Model")

if __name__ == "__main__":
    main()