import pandas as pd
import geopandas as gpd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

class DataMapper:
    def __init__(
            self,
            source_geometries: gpd.GeoSeries,
            target_geometries: gpd.GeoSeries,
            target_customer_counts: pd.Series,
            probabilisitic_rho: bool
    ):
        self.source_geometries = source_geometries
        self.target_geometries = target_geometries
        self.target_customer_counts = target_customer_counts
        self.area_crs = 25832
        self.relative_intersectional_areas_df = pd.DataFrame(index=self.source_geometries.index, columns=self.target_geometries.index, data=0.0)
        self.rho_mean_df = pd.DataFrame(index=self.source_geometries.index, columns=self.target_geometries.index, data=0.0)
        self.rho_std_dev_df = pd.DataFrame(index=self.source_geometries.index, columns=self.target_geometries.index, data=0.0)
        self.mapped_data_params = None
        self.quantiles_dict = None
        self.calculate_rhos(probabilisitic_rho)

    def map_data(self, data_dict: dict) -> dict:
        self.mapped_data_params = {}
        self.quantiles_dict = {}
        for key, data in data_dict.items():
            means, std_devs = self.estimate_binomial_moments(data)
            mapped_data_params = self.calculate_mapped_data_moments(means, std_devs)
            self.mapped_data_params[key] = mapped_data_params
            self.quantiles_dict[key] = self.compute_quantiles(mapped_data_params)
        return self.mapped_data_params

    def detect_intersections(self, source_idx: str) -> tuple[np.array, gpd.GeoSeries]:
        intersections = self.target_geometries.intersection(self.source_geometries.loc[source_idx])
        pip_mask = ~intersections.is_empty
        intersections_idxs = self.target_geometries[pip_mask].index.values
        return intersections_idxs, intersections

    def calculate_relative_intersectional_areas(self, intersections_idxs: np.array, intersections: gpd.GeoSeries) -> pd.Series:
        intersection_areas = intersections.loc[intersections_idxs].to_crs(self.area_crs).area
        target_areas = self.target_geometries.loc[intersections_idxs].to_crs(self.area_crs).area
        relative_intersectional_areas = intersection_areas / target_areas
        return relative_intersectional_areas

    def estimate_target_customers_in_source_geography(self, relative_intersectional_areas: pd.Series, intersections_idxs: np.array) -> pd.Series:
        target_customers_in_source_geography = relative_intersectional_areas * self.target_customer_counts.loc[intersections_idxs]
        return target_customers_in_source_geography
    
    def compute_beta_moments(self, a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
        mean = a/(a+b)
        std_dev = np.sqrt(a*b/(((a+b)**2)*(a+b+1)))
        return mean, std_dev

    def calculate_rhos(self, probabilistic: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        for source_idx in self.source_geometries.index:
            intersections_idxs, intersections = self.detect_intersections(source_idx)
            relative_intersectional_areas = self.calculate_relative_intersectional_areas(intersections_idxs, intersections)
            target_customers_in_source_geography = self.estimate_target_customers_in_source_geography(relative_intersectional_areas, intersections_idxs)
            self.relative_intersectional_areas_df.loc[source_idx, intersections_idxs] = relative_intersectional_areas
            a = target_customers_in_source_geography
            b = target_customers_in_source_geography.sum(axis=0) - target_customers_in_source_geography
            if probabilistic:
                rho_mean, rho_std_dev = self.compute_beta_moments(a, b)
            else:
                rho_mean = target_customers_in_source_geography / target_customers_in_source_geography.sum(axis=0)
                rho_std_dev = 0
            self.rho_mean_df.loc[source_idx, intersections_idxs] = rho_mean
            self.rho_std_dev_df.loc[source_idx, intersections_idxs] = rho_std_dev
        self.rho_mean_df.fillna(0, inplace=True)
        self.rho_std_dev_df.fillna(0, inplace=True)
        return self.rho_mean_df, self.rho_std_dev_df
    
    def estimate_binomial_moments(self, data: pd.DataFrame) -> tuple[np.array, np.array]:
        """
        Compute mean and standard deviation for multiple target geographies, 
        accounting for uncertainty in both `n` and `p`. This takes a normal approximation
        based on the mean and standard deviation.

        Returns:
        - Tuple (mean_array, std_dev_array) where:
            - mean_array: Mean values for each target geography.
            - std_dev_array: Standard deviation values for each target geography.
        """
        mu_n = data['mean'].values[:, None]
        sigma_n = data['std_dev'].values[:, None]
        mu_p = self.rho_mean_df.values
        sigma_p = self.rho_std_dev_df.values
        
        mean = mu_n * mu_p 
        variance = mu_n*(mu_p - mu_p**2 - sigma_p**2) + (mu_n**2 + sigma_n**2)*(mu_p**2 + sigma_p**2) - (mu_n**2)*(mu_p**2)
        return mean.sum(axis=0), np.sqrt(variance.sum(axis=0))  # Aggregate across sources
    
    def calculate_mapped_data_moments(self, means: np.array, std_devs: np.array) -> pd.DataFrame:
        mapped_data_params = pd.DataFrame({'mean': means, 'std_dev': std_devs}, index=self.rho_mean_df.columns)
        return mapped_data_params
    
    def compute_quantiles(self, mapped_data_params: pd.DataFrame) -> pd.DataFrame:
        """
        Compute quantiles (0 to 1) for each row in a DataFrame containing mean and std_dev.

        Parameters:
        - mapped_data_params_df: Pandas DataFrame with 'mean' and 'std_dev' columns.

        Returns:
        - Pandas DataFrame where each row corresponds to an entry in `mapped_data_params_df`,
        and columns represent quantiles from 0 to 1.
        """
        quantiles = np.linspace(0, 1.0, 101) # 0 to 1 in steps of 0.01
        
        means = np.nan_to_num(mapped_data_params["mean"].values[:, None], nan=0.0) # Reshape for broadcasting
        std_devs = np.maximum(mapped_data_params["std_dev"].values[:, None], 1e-6)

        quantile_values = stats.norm.ppf(quantiles, loc=means, scale=std_devs)

        # Replace extreme quantiles manually
        quantile_values[:, 0] = means[:, 0] - 4 * std_devs[:, 0] # Approximate 0th percentile
        quantile_values[:, -1] = means[:, 0] + 4 * std_devs[:, 0] # Approximate 100th percentile
        quantile_values = np.clip(quantile_values, 0, np.inf)

        # Create DataFrame with same index as input
        quantiles_df = pd.DataFrame(
            quantile_values, 
            index=mapped_data_params.index, 
            columns=[p for p in quantiles]
        )

        return quantiles_df.T
    
    def plot_mapped_data_mean(self, data_name: str):
        mean_data = self.mapped_data_params[data_name]['mean']
        mapped_mean_data_gdf = gpd.GeoDataFrame(data=mean_data, geometry=self.target_geometries)

        fig, ax = plt.subplots(figsize=(12, 6))

        mapped_mean_data_gdf.to_crs(25832).plot(
            ax=ax,
            column='mean',
            edgecolor='none',
            missing_kwds= dict(color = "lightgrey",),
            linewidth=0.1,
            cmap='Blues',
            legend=True
        )

        gpd.GeoSeries(data=self.target_geometries.to_crs(25832).union_all(), index=[0]).plot(
            ax=ax,
            edgecolor='black',
            linewidth=0.5,
            color='none'
        )

        ax.tick_params(left = False, labelleft = False, labelbottom = False, bottom = False)
        ax.add_artist(ScaleBar(1, length_fraction=0.3, location='lower left', font_properties={"size": 12}))
        colorbar = ax.get_figure().axes[-1]  # Get the colorbar axis
        colorbar.tick_params(labelsize=12)  # Adjust tick label size
        colorbar.set_ylabel(data_name, fontsize=16, labelpad=15)  # Set label and size
        plt.show()

def main():
    pass

if __name__ == "__main__":
    main()