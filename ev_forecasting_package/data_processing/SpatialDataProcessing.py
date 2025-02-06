import pandas as pd
import geopandas as gpd
import numpy as np
import os
import sys
import osmnx as ox

class DistributionSubstationDataProcessor:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.ds_data = None
        self.ds_gdf = None
        self.place_gdf = None
        self.ds_gdf_filtered = None
    
    def load_data(self, ds_data_filename: str) -> pd.DataFrame:
        self.ds_data = pd.read_csv(os.path.join(self.data_dir, ds_data_filename))
        return self.ds_data
    
    def process_data(self) -> pd.DataFrame:
        self.drop_columns()
        self.replace_values()
        self.change_dtypes()
        return self.ds_data

    def drop_columns(
            self, 
            columns: list[str] = [
                'LCT Count Total', 'Energy Storage', 'Heat Pumps', 
                'Total LCT Capacity', 'Total Generation Capacity', 'Solar', 
                'Wind', 'Bio Fuels', 'Water Generation', 'Waste Generation',
                'Storage Generation', 'Fossil Fuels', 'Other Generation'
            ]
        ) -> pd.DataFrame:
        self.ds_data = self.ds_data.drop(columns=columns)
        return self.ds_data
    
    def replace_values(self) -> pd.DataFrame:
        self.ds_data = self.ds_data.replace('Hidden', np.nan)
        return self.ds_data
    
    def change_dtypes(self) -> pd.DataFrame:
        self.ds_data = self.ds_data.astype({'Customers': 'float64', 'Substation Number': 'Int64'})
        self.ds_data = self.ds_data.astype({'Substation Number': str})
        return self.ds_data
    
    def load_geo_data(self, ds_geo_filename: str) -> gpd.GeoDataFrame:
        self.ds_gdf = gpd.read_file(os.path.join(self.data_dir, ds_geo_filename))
        return self.ds_gdf
    
    def process_geo_data(self) -> gpd.GeoDataFrame:
        self.ds_gdf = self.ds_gdf.rename(columns={'NR': 'Substation Number'})
        self.ds_gdf = self.ds_gdf.dissolve('Substation Number').reset_index()
        self.ds_gdf = self.ds_gdf.merge(self.ds_data, how='left', on='Substation Number')
        self.ds_gdf = self.ds_gdf.rename(columns={'Substation Name': 'Name'})
        self.ds_gdf = self.ds_gdf.fillna(value={'Discount': 'Unknown'})  # For the "key_on" part of the choropleth map
        self.ds_gdf = self.ds_gdf.to_crs('EPSG:4326')
        self.ds_gdf['Location'] = gpd.GeoSeries(gpd.points_from_xy(self.ds_gdf.Longitude, self.ds_gdf.Latitude, crs="EPSG:4326"))
        self.ds_gdf = self.ds_gdf.set_index('Substation Number', drop=False)  # Set Index to Substation Number
        self.ds_gdf = self.ds_gdf.loc[~self.ds_gdf.index.duplicated(keep='first')]  # Drop Duplicates
        return self.ds_gdf
    
    def filter_data_by_place(self, place: str) -> gpd.GeoDataFrame:
        self.place_gdf = ox.geocoder.geocode_to_gdf(place, which_result=1)
        self.ds_gdf_filtered = self.ds_gdf.loc[self.ds_gdf.geometry.intersects(self.place_gdf.geometry.values[0])] 
        # Changed from checking if Location was within place_gdf to checking if geometry intersects place_gdf.
        # Future versions could add both options and use this to flag inconsistencies.
        # Some of the substation locations are missing
        return self.ds_gdf_filtered
    
    def run_pipeline(self, ds_data_filename: str, ds_geo_filename: str, place: str) -> gpd.GeoDataFrame:
        self.load_data(ds_data_filename)
        self.process_data()
        self.load_geo_data(ds_geo_filename)
        self.process_geo_data()
        self.filter_data_by_place(place)
        return self.ds_gdf_filtered

class LSOABoundariesDataProcessor:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.lsoa_gdf = None
        self.lsoa_gdf_filtered = None
    
    def load_data(self, lsoa_geo_filename: str) -> gpd.GeoDataFrame:
        self.lsoa_gdf = gpd.read_file(os.path.join(self.data_dir, lsoa_geo_filename))
        return self.lsoa_gdf
    
    def convert_crs(self, crs: str = 'EPSG:4326') -> gpd.GeoDataFrame:
        self.lsoa_gdf = self.lsoa_gdf.to_crs(crs)
        return self.lsoa_gdf
    
    def set_index(self, index_col: str = 'LSOA11CD') -> gpd.GeoDataFrame:
        self.lsoa_gdf.set_index(index_col, inplace=True)
        return self.lsoa_gdf
    
    def filter_data_by_lad(self, lad: str) -> gpd.GeoDataFrame:
        self.lsoa_gdf_filtered = self.lsoa_gdf.loc[self.lsoa_gdf['LSOA11NM'].str[:-5] == lad]
        return self.lsoa_gdf_filtered
    
    def run_pipeline(self, lsoa_geo_filename: str, lad: str) -> gpd.GeoDataFrame:
        self.load_data(lsoa_geo_filename)
        self.convert_crs()
        self.set_index()
        self.filter_data_by_lad(lad) # Could add option to filter by place (similar to distribution substations)
        return self.lsoa_gdf