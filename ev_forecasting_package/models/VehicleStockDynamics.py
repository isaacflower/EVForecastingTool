import pandas as pd
import sys
import os
import pymc as pm
import arviz as az

class VehicleStockDynamicsInferenceModel:
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
    V_ev_l : np.array
        Number of EVs in the population (Local level).
    V_bev_l : np.array
        Number of BEVs in the population (Local level).
    V_phev_l : np.array
        Number of PHEVs in the population (Local level).
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
    """
    def __init__(self, model_variables_dict: dict, annual_data_dict: dict, lsoa_idxs: list) -> None:
        required_keys = [
            'A_v', 'R_v', 'total_vehicles', 'V_l', 'V_ev_l', 'V_bev_l', 'V_phev_l', 
            'delta_V_l_obs', 'delta_V_ev_l_obs', 'delta_V_bev_l_obs', 'delta_V_phev_l_obs', 
            'L', 'T'
        ]
        
        for key in required_keys:
            if key not in model_variables_dict:
                raise ValueError(f"Missing required model variable: {key}")
        
        self.A_v = model_variables_dict['A_v']
        self.R_v = model_variables_dict['R_v']
        self.total_vehicles = model_variables_dict['total_vehicles']
        self.V_l = model_variables_dict['V_l']
        self.V_ev_l = model_variables_dict['V_ev_l']
        self.V_bev_l = model_variables_dict['V_bev_l']
        self.V_phev_l = model_variables_dict['V_phev_l']
        self.delta_V_l_obs = model_variables_dict['delta_V_l_obs']
        self.delta_V_ev_l_obs = model_variables_dict['delta_V_ev_l_obs']
        self.delta_V_bev_l_obs = model_variables_dict['delta_V_bev_l_obs']
        self.delta_V_phev_l_obs = model_variables_dict['delta_V_phev_l_obs']
        self.L = model_variables_dict['L']
        self.T = model_variables_dict['T']
        self.annual_data_dict = annual_data_dict
        self.lsoa_idxs = lsoa_idxs
        self._create_lsoa_data_dict()

    def build_model(self) -> None:
        with pm.Model() as self.model:
            # Hyperparameters for Beta distributions - General vehicles
            p_A = pm.Beta('p_A', alpha=self.A_v + 1, beta=self.total_vehicles - self.A_v + 1, shape=self.T)
            p_R = pm.Beta('p_R', alpha=self.R_v + 1, beta=self.total_vehicles - self.R_v + 1, shape=self.T)

            # Latent counts for general vehicle additions and removals
            A_vehicles = pm.Binomial('A_vehicles', n=self.V_l, p=p_A, shape=(self.L, self.T))
            R_vehicles = pm.Binomial('R_vehicles', n=self.V_l, p=p_R, shape=(self.L, self.T))

            # Calculate EV Share of Vehicle Additions
            a = pm.Beta('a', alpha=1, beta=1, shape=(self.L, self.T))

            # Calculate EV Additions
            A_ev_l = pm.Deterministic('A_ev', a * A_vehicles)

            # Calculate removal rate
            r = pm.Deterministic('r', R_vehicles / self.V_l)

            # Calculate EV Removals
            R_ev_l = pm.Deterministic('R_ev', r * self.V_ev_l)

            # Calculate BEV and PHEV Share of Vehicle Additions
            bev_ev_a_share = pm.Beta('b', alpha=1, beta=1, shape=(self.L, self.T))
            phev_ev_a_share = pm.Deterministic('1 - b', 1 - bev_ev_a_share)

            # Calculate BEV and PHEV Additions
            A_bev_l = pm.Deterministic('A_bev', bev_ev_a_share * A_ev_l)
            A_phev_l = pm.Deterministic('A_phev', phev_ev_a_share * A_ev_l)

            # Calculate BEV and PHEV Removals
            R_bev_l = pm.Deterministic('R_bev', r * self.V_bev_l)
            R_phev_l = pm.Deterministic('R_phev', r * self.V_phev_l)

            # Incorporate observed net changes - General vehicles
            delta_V_l_mean = pm.Deterministic('A_vehicles - R_vehicles', A_vehicles - R_vehicles)
            delta_V_l = pm.Normal('delta_N_vehicles', mu=delta_V_l_mean, sigma=1e-1, observed=self.delta_V_l_obs)

            # Incorporate observed net changes - Electric vehicles
            delta_V_ev_l_mean = pm.Deterministic('A_ev - R_ev', A_ev_l - R_ev_l)
            delta_V_ev_l = pm.Normal('delta_N_ev', mu=delta_V_ev_l_mean, sigma=1e-1, observed=self.delta_V_ev_l_obs)

            # Incorporate observed net changes - BEVs
            delta_V_bev_l_mean = pm.Deterministic('A_bev - R_bev', A_bev_l - R_bev_l)
            delta_V_bev_l = pm.Normal('delta_N_bev', mu=delta_V_bev_l_mean, sigma=1e-1, observed=self.delta_V_bev_l_obs)

            # Incorporate observed net changes - PHEVs
            delta_V_phev_l_mean = pm.Deterministic('A_phev - R_phev', A_phev_l - R_phev_l)
            delta_V_phev_l = pm.Normal('delta_N_phev', mu=delta_V_phev_l_mean, sigma=1e-1, observed=self.delta_V_phev_l_obs)

        print("Model built successfully")

    def sample(self, samples=1000, tune=500, target_accept=0.99, **kwargs) -> None:
        with self.model:
            self.trace = pm.sample(samples, tune=tune, target_accept=target_accept, **kwargs)
        return self.trace
    
    def save_trace(self, trace_path: str) -> None:
        if os.path.exists(trace_path):
            os.remove(trace_path)
        self.trace.to_netcdf(trace_path)
        print(f"Trace saved to {trace_path}")

    def load_trace(self, trace_path: str) -> None:
        self.trace = az.from_netcdf(trace_path)
        print(f"Trace loaded from {trace_path}")
    
    def _create_lsoa_data_dict(self):
        lsoa_data_dict = {}
        for vehicle_type in [
            'v_lsoa', 'icev_lsoa', 'ev_lsoa', 'bev_lsoa', 'phev_lsoa', 
            'ev_market_share', 'bev_market_share', 'phev_market_share'
        ]:
            for lsoa in self.lsoa_idxs:
                if lsoa not in lsoa_data_dict:
                    lsoa_data_dict[lsoa] = pd.DataFrame()
                lsoa_data_dict[lsoa][vehicle_type] = self.annual_data_dict[vehicle_type][lsoa]
        self.lsoa_data_dict = lsoa_data_dict
        return self.lsoa_data_dict
    
    def calculate_posterior_means(self) -> None:
        variables = [
            'A_vehicles', 'R_vehicles', 'A_ev', 'R_ev', 'A_bev', 'R_bev', 
            'A_phev', 'R_phev', 'a', 'b', '1 - b'
        ]
        
        # Initialise dictionaries to hold the DataFrames
        years = self.annual_data_dict['v_lsoa'].iloc[1:].index.to_list() # from t_0 not t_0_raw
        lsoa_idxs = self.lsoa_idxs
        posterior_means = {var: pd.DataFrame(index=years, columns=lsoa_idxs) for var in variables}
        
        for lsoa_idx, l in enumerate(lsoa_idxs):
            for var in variables:
                dim_name = f"{var}_dim_0"
                mean_values = self.trace.posterior[var].sel({dim_name: lsoa_idx}).mean(axis=(0, 1))
                posterior_means[var].loc[:, l] = mean_values

        # Convert DataFrames to float and round
        for var in posterior_means:
            posterior_means[var] = posterior_means[var].astype(float).round(6)
        self.update_lsoa_data_dict(posterior_means)
        self.posterior_means = posterior_means
        print("Posterior means calculated successfully")
        return self.posterior_means
    
    def update_lsoa_data_dict(self, posterior_means: dict) -> dict:
        for key in posterior_means.keys():
            for lsoa in self.lsoa_idxs:
                self.lsoa_data_dict[lsoa][key] = posterior_means[key][lsoa]
        return self.lsoa_data_dict

def main():
    print("Running Vehicle Stock Dynamics Model")
    # Add the parent directory to the Python path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from ev_forecasting_package.data_processing.VehicleRegistrationDataProcessing import VehicleStockModelDataPrepper

    vehicle_stock_model_data_prepper = VehicleStockModelDataPrepper()
    year_quarter = '2023_Q4'
    file_names = {
        'stock_eng': f'stock_df_{year_quarter}.csv',
        'additions_eng': f'sales_df_{year_quarter}.csv',
        'v_lsoa': f'v_lsoa_{year_quarter}.csv',
        'icev_lsoa': f'icev_lsoa_{year_quarter}.csv',
        'ev_lsoa': f'ev_lsoa_{year_quarter}.csv',
        'bev_lsoa': f'bev_lsoa_{year_quarter}.csv',
        'phev_lsoa': f'phev_lsoa_{year_quarter}.csv'
    }
    vehicle_stock_model_data_prepper.prepare_data(
        data_path='../../data/large_datasets/vehicle_registrations/processed_data', 
        file_names=file_names, 
        lsoa_subset=20, 
        t_0=2022,
        t_0_raw=2021, 
        t_n=2023
    )
    model = VehicleStockDynamicsInferenceModel(
        model_variables_dict=vehicle_stock_model_data_prepper.model_variables_dict, 
        annual_data_dict=vehicle_stock_model_data_prepper.annual_data_dict, 
        lsoa_idxs=vehicle_stock_model_data_prepper.lsoa_subset
    )
    model.build_model()
    trace = model.sample()
    model.save_trace('vehicle_stock_dynamics_model_trace.nc')
    posterior_means = model.calculate_posterior_means()

if __name__ == "__main__":
    main()