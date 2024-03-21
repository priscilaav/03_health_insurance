import pickle
import numpy  as np
import pandas as pd

class HealthInsurance:
    
    def __init__( self ):
        self.home_path = 'C:/Users/prisc/OneDrive/Documentos/Dados/DS_prod/3_seguradora'
        self.annual_premium_robust_scaler =     pickle.load( open( self.home_path + 'parameter/annual_premium_robust_scaler.pkl' ) )
        self.age_robust_scaler =                pickle.load( open( self.home_path + 'parameter/age_robust_scaler.pkl' ) ) 
        self.vintage_minmax_scaler =            pickle.load( open( self.home_path + 'parameter/vintage_minmax_scaler.pkl' ) ) 
        self.target_encode_gender =             pickle.load( open( self.home_path + 'parameter/target_encode_gender.pkl' ) )
        self.fe_region_code =                   pickle.load( open( self.home_path + 'parameter/fe_region_code.pkl' ) )
        self.fe_policy_sales_channel =          pickle.load( open( self.home_path + 'parameter/fe_policy_sales_channel.pkl' ) )
        self.ordinal_mapping_vehicle_age =      pickle.load( open( self.home_path + 'parameter/ordinal_mapping_vehicle_age.pkl' ) )
        
    def data_cleaning( self, df1 ):
        # 1.1. Rename Columns
        new_cols = ['id', 'gender', 'age', 'driving_license', 'region_code', 'previously_insured', 'vehicle_age', 'vehicle_damage', 'annual_premium',
               'policy_sales_channel', 'vintage', 'response']
        df1.columns = new_cols

        return df1 

    
    def feature_engineering( self, df2 ):
        # 2.0. Feature Engineering

        # Vehicle Damage Number
        df2['vehicle_damage'] = df2['vehicle_damage'].apply( lambda x: 1 if x == 'Yes' else 0 )

        return df2
    
    
    def data_preparation( self, df5 ):
        # anual premium - StandarScaler
        df5['annual_premium'] = self.annual_premium_robust_scaler.transform( df5[['annual_premium']].values )

        # Age - MinMaxScaler
        df5['age'] = self.age_robust_scaler.transform( df5[['age']].values )

        # Vintage - MinMaxScaler
        df5['vintage'] = self.vintage_minmax_scaler.transform( df5[['vintage']].values )

        # gender - One Hot Encoding / Target Encoding
        df5.loc[:, 'gender'] = df5['gender'].map( self.target_encode_gender )

        # region_code - Target Encoding / Frequency Encoding
        df5.loc[:, 'region_code'] = df5['region_code'].map( self.fe_region_code )

        # policy_sales_channel - Target Encoding / Frequency Encoding
        df5.loc[:, 'policy_sales_channel'] = df5['policy_sales_channel'].map( self.fe_policy_sales_channel )
        
        # vehicle_age
        df5.loc[:, 'vehicle_age'] = df5['vehicle_age'].map( self.ordinal_mapping_vehicle_age )
        
        # Feature Selection
        cols_selected = ['vintage','annual_premium','age','region_code','vehicle_damage', 'policy_sales_channel', 'previously_insured', 'response']
        
        return df5[ cols_selected ]
    
    
    def get_prediction( self, model, original_data, test_data ):
        # model prediction
        pred = model.predict_proba( test_data )
        
        # join prediction into original data
        original_data['prediction'] = pred
        
        return original_data.to_json( orient='records', date_format='iso' )