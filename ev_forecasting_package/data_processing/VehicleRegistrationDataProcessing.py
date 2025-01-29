import pandas as pd
import numpy as np
import os

class NationalVehicleStockDataProcessor:
    def __init__(self):
        self.vehicle_stock_df_raw = None
        self.piv_stock_df_raw = None
        self.vehicle_stock_df = None
        self.piv_stock_df = None
        self.stock_df = None

    def read_excel_file(self, file_path, details: dict) -> pd.DataFrame:
        return pd.read_excel(
            file_path,
            sheet_name=details['sheet_name'], 
            header=details['header'], 
            engine=details['engine']
        )
    
    def load_raw_data(self, raw_data_path: str, meta_data: dict) -> pd.DataFrame:
        for vehicle_type, details in meta_data.items():
            file_path = os.path.join(raw_data_path, details['file_name'])
            print(f"Loading {vehicle_type} data from {file_path}")
            if vehicle_type == 'vehicle_stock':
                self.vehicle_stock_df_raw = self.read_excel_file(file_path, details)
            elif vehicle_type == 'piv_stock':
                self.piv_stock_df_raw = self.read_excel_file(file_path, details)
        return self.vehicle_stock_df_raw, self.piv_stock_df_raw
    
    def filter_dataframe(self, df_raw: pd.DataFrame, filters: dict) -> pd.DataFrame:
        df = (
            df_raw
            .rename(columns=filters['new_cols'])
            .query(filters['query'])
            .drop(filters['dropped_cols'], axis=1)
            .reset_index(drop=True)
        )
        return df
    
    def process_raw_data(self, filters_dict: dict) -> pd.DataFrame:
        if self.vehicle_stock_df_raw is None or self.piv_stock_df_raw is None:
            raise ValueError('Raw data not loaded. Please load raw data first.')
        
        for vehicle_type, filters in filters_dict.items():
            if vehicle_type == 'vehicle_stock':
                self.vehicle_stock_df = self.filter_dataframe(self.vehicle_stock_df_raw, filters)
                self.vehicle_stock_df.iloc[:, 1:] *= 1000
            elif vehicle_type == 'piv_stock':
                self.piv_stock_df = self.filter_dataframe(self.piv_stock_df_raw, filters)
        
        return self.vehicle_stock_df, self.piv_stock_df
    
    def calculate_stock(self) -> pd.DataFrame:
        total_stock = self.vehicle_stock_df[self.vehicle_stock_df['Fuel'] == 'Total'].drop(['Fuel'], axis=1).iloc[0]
        total_stock.rename('Total', inplace=True)

        icev_stock = self.vehicle_stock_df[self.vehicle_stock_df['Fuel'] == 'Total'].drop(['Fuel'], axis=1).iloc[0] - self.vehicle_stock_df[self.vehicle_stock_df['Fuel'] == 'Other fuels'].drop(['Fuel'], axis=1).iloc[0]
        icev_stock.rename('ICEV', inplace=True)

        ev_stock = self.piv_stock_df[self.piv_stock_df['Fuel'] == 'Total'].drop(['Fuel'], axis=1).iloc[0]
        ev_stock.rename('EV', inplace=True)

        bev_stock = self.piv_stock_df[(self.piv_stock_df['Fuel'] == 'Battery electric') | (self.piv_stock_df['Fuel'] == 'Range extended electric')].drop(['Fuel'], axis=1).sum()
        bev_stock.rename('BEV', inplace=True)

        phev_stock = self.piv_stock_df[(self.piv_stock_df['Fuel'] == 'Plug-in hybrid electric (diesel)') | (self.piv_stock_df['Fuel'] == 'Plug-in hybrid electric (petrol)')].drop(['Fuel'], axis=1).sum()
        phev_stock.rename('PHEV', inplace=True)

        self.stock_df = pd.concat([total_stock, icev_stock, ev_stock, bev_stock, phev_stock], axis=1)[::-1]

        return self.stock_df
    
    def run_pipeline(self, raw_data_path: str, meta_data: dict, filters_dict: dict):
        self.load_raw_data(raw_data_path, meta_data)
        self.process_raw_data(filters_dict)
        self.calculate_stock()
        print('Pipeline run successfully')

    def save_data(self, save_path: str, year_quarter: str):
        self.stock_df.to_csv(os.path.join(save_path, f'stock_df_{year_quarter}.csv'))
        print('Data saved successfully')

class NationalVehicleSalesDataProcessor:
    def __init__(self):
        self.vehicle_sales_df_raw = None
        self.vehicle_sales_df = None
        self.sales_df = None

    def read_excel_file(self, file_path, details: dict) -> pd.DataFrame:
        return pd.read_excel(
            file_path,
            sheet_name=details['sheet_name'], 
            header=details['header'], 
            engine=details['engine']
        )
    
    def load_raw_data(self,raw_data_path: str, meta_data: dict ) -> pd.DataFrame:
        file_path = os.path.join(raw_data_path, meta_data['file_name'])
        print(f"Loading data from {file_path}")
        self.vehicle_sales_df_raw = self.read_excel_file(file_path, meta_data)
        return self.vehicle_sales_df_raw
    
    def process_raw_data(self, filters_dict: dict) -> pd.DataFrame:
        if self.vehicle_sales_df_raw is None:
            raise ValueError('Raw data not loaded. Please load raw data first.')
        vehicle_sales_df = (
            self.vehicle_sales_df_raw.rename(columns=filters_dict['new_cols'])
            .query(filters_dict['query'])
            .drop(filters_dict['dropped_cols'], axis=1)
            .reset_index(drop=True)
        )
        vehicle_sales_df['Date'] = vehicle_sales_df['Date'].str.replace(r"\(.*\)","", regex=True).str.strip()
        vehicle_sales_df.set_index('Date', inplace=True)
        vehicle_sales_df.iloc[:, :] = vehicle_sales_df.iloc[:, :]*1000
        vehicle_sales_df['ICEV'] = (
            vehicle_sales_df['Petrol'] + 
            vehicle_sales_df['Diesel'] + 
            vehicle_sales_df['Hybrid electric (petrol)'] + 
            vehicle_sales_df['Hybrid electric (diesel)']
        )
        self.vehicle_sales_df = vehicle_sales_df
        return self.vehicle_sales_df
    
    def calculate_sales(self) -> pd.DataFrame:
        total_sales = (self.vehicle_sales_df['Total'])
        total_sales.rename('Total', inplace=True)

        icev_sales = self.vehicle_sales_df['ICEV']
        icev_sales.rename('ICEV', inplace=True)

        ev_sales = self.vehicle_sales_df['EV']
        ev_sales.rename('EV', inplace=True)

        bev_sales = self.vehicle_sales_df['BEV'] + self.vehicle_sales_df['REX']
        bev_sales.rename('BEV', inplace=True)

        phev_sales = (self.vehicle_sales_df['PHEV_P']+self.vehicle_sales_df['PHEV_D'] )
        phev_sales.rename('PHEV', inplace=True)

        self.sales_df = pd.concat([total_sales, icev_sales, ev_sales, bev_sales, phev_sales], axis=1)

        return self.sales_df
    
    def run_pipeline(self, raw_data_path: str, meta_data: dict, filters_dict: dict):
        self.load_raw_data(raw_data_path, meta_data)
        self.process_raw_data(filters_dict)
        self.calculate_sales()
        print('Pipeline run successfully')

    def save_data(self, save_path: str, year_quarter: str):
        self.sales_df.to_csv(os.path.join(save_path, f'sales_df_{year_quarter}.csv'))
        print('Data saved successfully')

class LSOAVehicleRegistrationDataProcessor:
    def __init__(self, lsoa_lookup_path: str):
        self.v_reg_df_raw = None
        self.v_reg_df = None
        self.piv_reg_df_raw = None
        self.piv_reg_df = None
        self.lsoa_lookup = pd.read_csv(lsoa_lookup_path)
        self.ev_reg_df = None
        self.bev_reg_df = None
        self.phev_reg_df = None
        self.icev_reg_df = None

    def apply_dtypes(self, first: int, last: int) -> dict:
        dtypes = {i: str for i in range(first)}
        dtypes.update({i: float for i in range(first, last)})
        return dtypes
    
    def load_csv(self, filepath: str, details: dict) -> pd.DataFrame:
        csv_df = pd.read_csv(
            filepath, 
            dtype=self.apply_dtypes(details['first'], details['last']),
            na_values=details['na_values']
        )
        return csv_df
    
    def load_raw_data(self, raw_data_path: str, meta_data: dict):
        for vehicle_type, details in meta_data.items():
            file_path = os.path.join(raw_data_path, details['file_name'])
            print(f"Loading {vehicle_type} data from {file_path}")
            if vehicle_type == 'v_reg':
                self.v_reg_df_raw = self.load_csv(file_path, details)
            elif vehicle_type == 'piv_reg':
                self.piv_reg_df_raw = self.load_csv(file_path, details)
        return self.v_reg_df_raw, self.piv_reg_df_raw
    
    def filter_dataframe(self, df_raw: pd.DataFrame, filters: dict, LAD: str) -> pd.DataFrame:
        df = (
            df_raw
            .query(filters['query'])
            .drop(filters['dropped_cols'], axis=1)
            .merge(self.lsoa_lookup.loc[:, ['LSOA11CD', 'LAD22NM']], on='LSOA11CD', how='outer')
            .loc[lambda x: x['LAD22NM'] == LAD] # Filter for target Local Authority
            .drop(columns = ['LAD22NM'])
            .set_index('LSOA11CD')
        )
        return df
    
    def process_raw_data(self, filters_dict: dict, LAD: str) -> pd.DataFrame:
        if self.v_reg_df_raw is None or self.piv_reg_df_raw is None:
            raise ValueError('Raw data not loaded. Please load raw data first.')
        for vehicle_type, filters in filters_dict.items():
            if vehicle_type == 'v_reg':
                v_reg_df = self.filter_dataframe(self.v_reg_df_raw, filters, LAD)
                self.v_reg_df = v_reg_df[~v_reg_df.index.duplicated(keep='first')].T.iloc[::-1]
            elif vehicle_type == 'piv_reg':
                self.piv_reg_df = self.filter_dataframe(self.piv_reg_df_raw, filters, LAD)
        
        return self.v_reg_df, self.piv_reg_df
    
    def disaggregate_data(self) -> pd.DataFrame:
        template_piv_reg_df = pd.DataFrame(index=self.piv_reg_df.index.unique(), columns=self.piv_reg_df.columns[1:], data=0)

        ev_reg_df = template_piv_reg_df + self.piv_reg_df.query("Fuel == 'Total'").drop(columns = ['Fuel']).astype(float)
        ev_reg_df = ev_reg_df[~ev_reg_df.index.duplicated(keep='first')].T.iloc[::-1]
        self.ev_reg_df = ev_reg_df

        bev_reg_df = template_piv_reg_df + self.piv_reg_df.query("Fuel == 'Battery electric'").drop(columns = ['Fuel']).astype(float)
        bev_reg_df = bev_reg_df[~bev_reg_df.index.duplicated(keep='first')].T.iloc[::-1]
        self.bev_reg_df = bev_reg_df

        phev_reg_df = template_piv_reg_df + self.piv_reg_df.query("Fuel == 'Plug-in hybrid electric (petrol)'").drop(columns = ['Fuel']).astype(float)
        phev_reg_df = phev_reg_df[~phev_reg_df.index.duplicated(keep='first')].T.iloc[::-1]
        self.phev_reg_df = phev_reg_df

        return self.ev_reg_df, self.bev_reg_df, self.phev_reg_df
    
    def fill_missing_bev_and_phev_data(self):
        # Fill instances where total and PHEV are known but BEV is unknown
        BOOL = self.ev_reg_df.notna() & self.bev_reg_df.isna() & self.phev_reg_df.notna()
        self.bev_reg_df[BOOL] = self.ev_reg_df[BOOL] - self.phev_reg_df[BOOL]

        # Fill instances where total and BEV are known but PHEV is unknown
        BOOL = self.ev_reg_df.notna() & self.bev_reg_df.notna() & self.phev_reg_df.isna()
        self.phev_reg_df[BOOL] = self.ev_reg_df[BOOL] - self.bev_reg_df[BOOL]

        # Fill instances where total is known but BEV and PHEV are unknown
        BOOL = self.ev_reg_df.notna() & self.bev_reg_df.isna() & self.phev_reg_df.isna()
        bev_split = (self.bev_reg_df.sum(axis=1)/(self.bev_reg_df.sum(axis=1) + self.phev_reg_df.sum(axis=1))).fillna(0.5)
        phev_split = (self.phev_reg_df.sum(axis=1)/(self.bev_reg_df.sum(axis=1) + self.phev_reg_df.sum(axis=1))).fillna(0.5)
        self.bev_reg_df[BOOL] = round(bev_split*self.ev_reg_df[BOOL])
        self.phev_reg_df[BOOL] = round(phev_split*self.ev_reg_df[BOOL])

        return self.bev_reg_df, self.phev_reg_df
    
    def calculate_t0(self, data: pd.Series | pd.DataFrame) -> float:
        """
        Returns the first (and oldest) date (assuming date is the index) 
        of a Pandas Series or DataFrame
        """
        year = int(data.head(1).index[0][:4])
        quarter = data.head(1).index[0][-2:]
        if quarter == 'Q1':
            t0 = year + 0
        elif quarter == 'Q2':
            t0 = year + 0.25
        elif quarter == 'Q3':
            t0 = year + 0.5
        elif quarter == 'Q4':
            t0 = year + 0.75
        return t0
    
    def calculate_t_present(self, data: pd.Series | pd.DataFrame) -> float:
        """
        Returns the last (and most recent) date (assuming date is the index) 
        of a Pandas Series or DataFrame
        """
        year = int(data.tail(1).index[0][:4])
        quarter = data.tail(1).index[0][-2:]
        if quarter == 'Q1':
            t_present = year + 0
        elif quarter == 'Q2':
            t_present = year + 0.25
        elif quarter == 'Q3':
            t_present = year + 0.5
        elif quarter == 'Q4':
            t_present = year + 0.75    
        return t_present
    
    def calculate_date_range(self, t0: float, t1: float, sample_rate=4) -> np.array:
        # sample_rate = 4 implies quarterly data
        return np.linspace(t0, t1, int((t1-t0)*sample_rate) + 1)
    
    def interpolate_col(self, col: pd.Series) -> pd.Series:
        dates = self.calculate_date_range(self.calculate_t0(col), self.calculate_t_present(col))
        mask = ~col.isna().values
        if mask.any():
            xp = dates[mask]
            fp = col[mask]
            x = dates
            interpolated = np.round(np.interp(x, xp, fp))
            return pd.Series(data=interpolated, index=col.index)
        else:
            return pd.Series(data=np.nan, index=col.index)
    
    def interpolate_df(self, df: pd.DataFrame) -> pd.DataFrame:
        interpolated_df = df.apply(self.interpolate_col, axis=0)
        interpolated_df = interpolated_df.fillna(0)
        return interpolated_df
    
    def interpolate_piv_reg_data(self):
        # Set first data point to 1 if NaN
        self.ev_reg_df.iloc[0, self.ev_reg_df.iloc[0].isna()] = 1
        self.bev_reg_df.iloc[0, self.bev_reg_df.iloc[0].isna()] = 1
        self.phev_reg_df.iloc[0, self.phev_reg_df.iloc[0].isna()] = 1

        # Interpolate missing values
        self.ev_reg_df = self.interpolate_df(self.ev_reg_df)
        self.bev_reg_df = self.interpolate_df(self.bev_reg_df)
        self.phev_reg_df = self.interpolate_df(self.phev_reg_df)
        return self.ev_reg_df, self.bev_reg_df, self.phev_reg_df
    
    def infer_icev_reg_data(self):
        self.icev_reg_df = self.v_reg_df - self.ev_reg_df
        self.icev_reg_df[self.icev_reg_df.isna()] = self.v_reg_df[self.icev_reg_df.isna()]
        return self.icev_reg_df
    
    def run_pipeline(self, raw_data_path: str, meta_data: dict, filters_dict: dict, LAD: str):
        self.load_raw_data(raw_data_path, meta_data)
        self.process_raw_data(filters_dict, LAD)
        self.disaggregate_data()
        self.fill_missing_bev_and_phev_data()
        self.interpolate_piv_reg_data()
        self.infer_icev_reg_data()
        print('Pipeline run successfully')

    def save_data(self, save_path: str, year_quarter: str):
        self.v_reg_df.to_csv(os.path.join(save_path, f'v_lsoa_{year_quarter}.csv'))
        self.ev_reg_df.to_csv(os.path.join(save_path, f'ev_lsoa_{year_quarter}.csv'))
        self.bev_reg_df.to_csv(os.path.join(save_path, f'bev_lsoa_{year_quarter}.csv'))
        self.phev_reg_df.to_csv(os.path.join(save_path, f'phev_lsoa_{year_quarter}.csv'))
        self.icev_reg_df.to_csv(os.path.join(save_path, f'icev_lsoa_{year_quarter}.csv'))
        print('Data saved successfully')

def main():
    # National Vehicle Stock Data Processing
    national_vehicle_stock_data_processor = NationalVehicleStockDataProcessor()
    raw_data_path = '../../data/large_datasets/vehicle_registrations/raw_data'
    meta_data = {
        'vehicle_stock': {
            'file_name': 'veh0105_2023_Q4.ods',
            'sheet_name': 'VEH0105',
            'header': 4,
            'engine': 'odf'
        },
        'piv_stock': {
            'file_name': 'veh0142_2023_Q4.ods',
            'sheet_name': 'VEH0142',
            'header': 4,
            'engine': 'odf'
        }
    }
    filters_dict = {
        'vehicle_stock': {
            'new_cols': {
                'Fuel [note 2]': 'Fuel',
                'Keepership [note 3]': 'Keepership',
                'ONS Geography [note 6]': 'Geography',
            },
            'query': "Geography == 'England' & Units == 'Thousands' & BodyType == 'Cars' & Keepership == 'Total'",
            'dropped_cols': [
                'Units', 
                'Geography', 
                'BodyType', 
                'Keepership',
                'ONS Sort [note 6]',
                'ONS Code [note 6]'
            ]
        },
        'piv_stock': {
            'new_cols': {
                'Fuel [note 2]': 'Fuel',
                'Keepership [note 3]': 'Keepership',
                'ONS Geography [note 6]': 'Geography',
            },
            'query': "Geography == 'England' & Units == 'Number' & BodyType == 'Cars' & Keepership == 'Total'",
            'dropped_cols': [
                'Units', 
                'Geography', 
                'BodyType', 
                'Keepership',
                'ONS Sort [note 6]',
                'ONS Code [note 6]'
            ]
        }
    }
    national_vehicle_stock_data_processor.run_pipeline(raw_data_path, meta_data,  filters_dict)
    national_vehicle_stock_data_processor.save_data('../../data/large_datasets/vehicle_registrations/processed_data', '2023_Q4')

    # National Vehicle Sales Data Processing
    national_vehicle_sales_data_processor = NationalVehicleSalesDataProcessor()
    raw_data_path = '../../data/large_datasets/vehicle_registrations/raw_data'
    meta_data = {
        'file_name': 'veh1153_2023_Q4.ods',
        'sheet_name': 'VEH1153a_RoadUsing',
        'header': 4,
        'engine': 'odf'
    }
    filters_dict = {
        'new_cols': {
            'Geography [note 4]': 'Geography',
            'Date Interval [note 5]': 'DateInterval',
            'Date [note 5]': 'Date',
            'Battery electric': 'BEV',
            'Plug-in hybrid electric (petrol)': 'PHEV_P',
            'Plug-in hybrid electric (diesel)': 'PHEV_D',
            'Range extended electric': 'REX',
            'Fuel cell electric': 'FCEV',
            'Plugin [note 8]': 'EV',
        },
        'query': "Geography == 'England' & DateInterval == 'Quarterly' & Units == 'Thousands' & BodyType == 'Cars' & Keepership == 'Total'",
        'dropped_cols': [
            'Geography',
            'DateInterval',
            'Units',
            'BodyType',
            'Keepership'
        ]
    }
    national_vehicle_sales_data_processor.run_pipeline(raw_data_path, meta_data, filters_dict)
    national_vehicle_sales_data_processor.save_data('../../data/large_datasets/vehicle_registrations/processed_data', '2023_Q4')
    
    # LSOA Vehicle Registration Data Processing
    lsoa_vehicle_registration_data_processor = LSOAVehicleRegistrationDataProcessor(
        lsoa_lookup_path='../../data/large_datasets/spatial/LSOA_(2011)_to_LSOA_(2021)_to_Local_Authority_District_(2022)_Lookup_for_England_and_Wales_(Version_2).csv'
        )
    raw_data_path = '../../data/large_datasets/vehicle_registrations/raw_data'
    meta_data = {
        'v_reg': {
            'file_name': 'df_VEH0125_2023_Q4.csv',
            'first': 5,
            'last': 57,
            'na_values': ['[c]', '[x]'],
        },
        'piv_reg': {
            'file_name': 'df_VEH0145_2023_Q4.csv',
            'first': 5,
            'last': 56,
            'na_values': ['[c]', '[x]'],
        }
    }
    filters_dict = {
        'v_reg': {
            'query': "BodyType == 'Cars' and Keepership == 'Private' and LicenceStatus == 'Licensed'",
            'dropped_cols': ['BodyType', 'Keepership', 'LicenceStatus', 'LSOA11NM']
        },
        'piv_reg': {
            'query': "Keepership == 'Private'",
            'dropped_cols': ['Keepership', 'LSOA11NM'] 
        }
    }
    LAD = 'Bath and North East Somerset'
    lsoa_vehicle_registration_data_processor.run_pipeline(raw_data_path, meta_data, filters_dict, LAD)
    lsoa_vehicle_registration_data_processor.save_data('../../data/large_datasets/vehicle_registrations/processed_data', '2023_Q4')

if __name__ == '__main__':
    main()