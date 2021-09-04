import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class Transform():
    def __init__(self):
        pass

    def read_data(self, data):
        '''
        Converts a numpy array into a DataFrame.

        Parameters
        ----------
        data : string, takes in a string where a dataset or array is stored,
        eg:'df','df1'
        where df is a variable which contains either a dataframe or numpy array

        Returns
        -------
        DataFrame
        
        '''
        try:
            df = pd.DataFrame(data)
            return df
        except Exception as e:
            print(e)

    def drop_col(self, data, dropcol='None'):
        '''
        Drops the feature columns

        Parameters
        ----------
        data : string, takes in a string where a dataset or array is stored,
        eg:'df','df1'(where df is a variable which contains either a dataframe or numpy array)
        
        dropcol(default = 'None') : Add the list of feature columns to be deleted from the given dataset.
        
        Returns
        -------
        DataFrame
        
        '''
        try:
            data = data.copy()
            df = self.read_data(data)
            if dropcol == 'None':
                raise Exception(
                    'Column to be deleted is not defined, to delete a column please pass a column name as a parameter')
            else:
                df = df.drop(columns=dropcol)
                return df
        except Exception as e:
            print(e)

    def drop_cat_col(self, data):
        '''
        Drops all the feature columns which are not either 'int or float' dtypes from the given dataset

        Parameters
        ----------
        data : string, takes in a string where a dataset or array is stored,
        eg:'df','df1'(where df is a variable which contains either a dataframe or numpy array)
        
        Returns
        -------
        DataFrame
        
        '''
        try:
            data = data.copy()
            df = self.read_data(data)
            for i in df:
                if df[i].dtypes != int and df[i].dtypes != float:
                    df = df.drop(columns=i)
            return df
        except Exception as e:
            print(e)

    def count_missingval(self, data):
        '''
        Calculates and displays the number of NaN values in each feature column for the given dataset.

        Parameters
        ----------
        data : string, takes in a string where a dataset or array is stored,
        eg:'df','df1'(where df is a variable which contains either a dataframe or numpy array)

        Returns
        -------
        DataFrame
        
        '''
        try:
            df = self.read_data(data)
            count = df.isnull().sum()
            return count
        except Exception as e:
            print(e)

    def fill_mean(self, data):
        '''
        Calculates mean of each feature column(only for the features with dtypes int or float) for the given dataset,
        and replaces all the NaN values with calculated mean

        Parameters
        ----------
        data : string, takes in a string where a dataset or array is stored,
        eg:'df','df1'(where df is a variable which contains either a dataframe or numpy array)

        Returns
        -------
        Replaces all NaN values with 'mean' and returns a DataFrame with no NaN values
        
        '''
        try:
            data = data.copy()
            df = self.drop_cat_col(data)
            for i in df:
                if df[i].isnull().sum() != 0:
                    df[i] = df[i].fillna(value=df[i].mean())
            return df
        except Exception as e:
            print(e)

    def fill_median(self, data):
        '''
        Calculates median of each feature column(only for the features with dtypes int or float) for the given dataset,
        and replaces all the NaN values with calculated median

        Parameters
        ----------
        data : string, takes in a string where a dataset or array is stored,
        eg:'df','df1'(where df is a variable which contains either a dataframe or numpy array)

        Returns
        -------
        Replaces all NaN values with 'median' and returns a DataFrame with no NaN values
        
        '''
        try:
            data = data.copy()
            df = self.drop_cat_col(data)
            for i in df:
                if df[i].isnull().sum() != 0:
                    df[i] = df[i].fillna(value=df[i].median())
            return df
        except Exception as e:
            print(e)

    def fill_mode(self, data):
        '''
        Calculates mode of each feature column (only for the features with dtypes int or float) and replaces the NaN values with calculated mode

        Parameters
        ----------
        data : string, takes in a string where a dataset or array is stored,
        eg:'df','df1'(where df is a variable which contains either a dataframe or numpy array)

        Returns
        -------
        Replaces all NaN values with 'mode' and returns a DataFrame with no NaN values

        '''
        try:
            data = data.copy()
            df = self.drop_cat_col(data)
            for i in df:
                if df[i].isnull().sum() != 0:
                    df[i] = df[i].fillna(value=df[i].mode()[0])
            return df
        except Exception as e:
            print(e)

    def scaling(self, data, dropcol='None'):
        '''
        Standardize features(which are either of int or float dtypes) by removing the mean and scaling to unit variance

        Parameters
        ----------
        1. data : string, takes in a string where a dataset or array is stored,
        eg:'df','df1'(where df is a variable which contains either a dataframe or numpy array)
        2. dropcol : Enter list of all feature columns to be deleted.
        
        Returns
        -------
        Replaces all NaN values with 'median' and returns a DataFrame with no NaN values
        
        '''
        try:
            data = data.copy()
            df = self.drop_cat_col(data)
            if dropcol == 'None':
                scaler = StandardScaler()
                transformed_arr = scaler.fit_transform(df)
                standardized_df = pd.DataFrame(
                    transformed_arr, columns=df.columns)
                return standardized_df
            else:
                transform_cols = df.drop(columns=dropcol)
                scaler = StandardScaler()
                transformed_arr = scaler.fit_transform(transform_cols)
                standardized_df = pd.DataFrame(
                    transformed_arr, columns=transform_cols.columns)
                return standardized_df
        except Exception as e:
            print(e)

    def std_transform(self, data, dropcol='None', fillnan='None'):
        '''
        Standardize features(which are either of int or float dtypes) by removing the mean and scaling to unit variance

        Parameters
        ----------
        1. data : string, takes in a string where a dataset or array is stored,
        eg:'df','df1'
        where df is a variable which contains either a dataframe or numpy array
        2. dropcol : Enter list of all feature columns to be deleted before standarding the dataset
        3. fillnan : Enter a value with which the NaN values are to be replaced with
            options:
            a. 'mean'  ---> Replaces all the NaN values with mean
            b. 'median'---> Replaces all the NaN values with median
            c. 'mode'  ---> Replaces all the NaN values with mode

        Returns
        -------
        Replaces all NaN values with either mean or median or mode and returns a DataFrame with no NaN values

        '''
        try:
            data = data.copy()
            df = self.drop_cat_col(data)
            if fillnan == 'None':
                return self.scaling(df, dropcol)
            elif fillnan == 'mean':
                df = self.fill_mean(data)
                return self.scaling(df, dropcol)
            elif fillnan == 'median':
                df = self.fill_median(data)
                return self.scaling(df, dropcol)
            elif fillnan == 'mode':
                df = self.fill_mode(data)
                return self.scaling(df, dropcol)
            else:
                raise NameError(f'{fillnan} is not defined')
        except Exception as e:
            print(e)
