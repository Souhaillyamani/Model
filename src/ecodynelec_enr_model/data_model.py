"""
This module contains functions to train and use a power prediction model.

Two default models are provided:
    - a model for wind power prediction
    - a model for solar power prediction
These two models use ecd_enr_model from 2020 to 2022 for training.
They use:
    - a plants mapping file to get the installed capacities
    - a weather ecd_enr_model file to get the weather ecd_enr_model (from ERA5/Copernicus)
    - a power ecd_enr_model file to get the real power ecd_enr_model (from Pronovo and EnergyCharts), see data_loading module

This module contains the following functions:
    - train_model: trains a model with the given parameters
    - save_model: saves a model
    - load_model: loads a model from the given file
    - predict_power_production: predicts the power for the given years
    - load_expected_result : loads the expected prediction result for the given years,
    in the format as the predict_power result
    - train_predict_pipeline : trains a model and predicts the power for the given years
    - generate_production_files : generates the production files for the given years : the result will contain the real
    production for the training years, and the predicted production for the other years
"""

from os.path import exists

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesRegressor

from ecodynelec_enr_model.data_loading import load_model_data, generate_train_data, load_all_capacities, load_all_pronovo_files, \
    save_power_data, root_dir


wind_plants_model = {
    'file': 'CH_ElectricityProductionPlant_final.csv',
    'date_col': 'BeginningOfOperation',
    'category_col': 'SubCategory',
    'category': 'Wind_Onshore_CH',
    'name_prefixs': ['wind_speed'],
    'gen_path_x': ['generated_wind/'],
    '.cdf_value_vars': [['var_100_metre_wind_speed', 'ws100']],
    'gen_path_y': 'generated_wind/',
    'type': 'Wind',
    'add_hour': False,
    'div_by_capa': True,

    'add_0_capa': True,
    'train_years': [2020, 2021, 2022],
    'plant_mapping_year': 2022,
    'n_nearests': 4,
    'regressor': ExtraTreesRegressor(random_state=0, n_jobs=-1, n_estimators=100, criterion="poisson")
}
"""
Default model for wind power prediction, hour by hour, using an ExtraTreesRegressor.
It uses ecd_enr_model from 2020 to 2022 for training.
Input ecd_enr_model:
    - a plants mapping file to get the installed capacities
    - wind speeds (from ERA5/Copernicus)
Output ecd_enr_model:
    - real wind power production (from Pronovo and EnergyCharts)
See data_loading module for more details.
"""

solar_plants_model = {
    'file': 'CH_ElectricityProductionPlant_final.csv',
    'date_col': 'BeginningOfOperation',
    'category_col': 'SubCategory',
    'category': 'Solar_CH',
    'name_prefixs': ['solar_radiation', '2m_temperature'],
    'gen_path_x': ['generated_solar/SR_', 'generated_solar/T2m_'],
    '.cdf_value_vars': [['rsds', 'rsdsAdjust'], ['t2m', 'tasAdjust']],
    'gen_path_y': 'generated_solar/',
    'type': 'Solar',
    'add_hour': True,
    'div_by_capa': True,

    'add_0_capa': True,
    'train_years': [2020, 2021, 2022],
    'plant_mapping_year': 2022,
    'n_nearests': 1,
    'regressor': ExtraTreesRegressor(random_state=0, verbose=1, n_estimators=1000, n_jobs=-1, criterion="friedman_mse")
}
"""
Default model for solar power prediction, hour by hour, using an ExtraTreesRegressor.
It uses ecd_enr_model from 2020 to 2022 for training.
Input ecd_enr_model:
    - a plants mapping file to get the installed capacities
    - solar radiation and temperature (from ERA5/Copernicus)
Output ecd_enr_model:
    - real solar power production (from Pronovo and EnergyCharts)
See data_loading module for more details.
"""


def train_model(model_params: dict, generate_input: bool=False, verbose: bool=False) -> BaseEstimator:
    """
    Trains the given model

    :param model_params: the model parameters, see the default models for more details
    :param generate_input: if True, the input/output ecd_enr_model of the model will be generated
    (else, it will be loaded from local files)
    :param verbose: if True, the function will print more debug information
    :return: the trained model
    """

    if generate_input:
        years = sorted(model_params['train_years'], reverse=True)
        print('Generating train ecd_enr_model for years: ', years)
        generate_train_data(years, model_params=model_params, load_X=True, load_Y=True, verbose=verbose)
    full_X, X, full_Y, Y = load_model_data(model_params['train_years'], model_params, load_X=True, load_Y=True,
                                           add_0_capa=model_params['add_0_capa'], verbose=verbose)
    regr = model_params['regressor']
    print('Training model...', end='\r')
    regr.fit(X, Y)
    print('Training model - done                   ')
    return regr


def save_model(model_params: dict, regressor: BaseEstimator, save_name: str="model"):
    """
    Saves the given model

    :param model_params:  the model parameters, see the default models for more details
    :param regressor:  the trained model to save
    :param save_name:  the name of the file to save the model
    """
    import joblib
    joblib.dump(regressor, f'{root_dir}{model_params["gen_path_y"]}{save_name}.joblib')
    print(f'Model saved to {root_dir}{model_params["gen_path_y"]}{save_name}.joblib')


def load_model(model_params: dict, save_name: str="model") -> BaseEstimator:
    """
    Loads the given model

    :param model_params: the model parameters, see the default models for more details
    :param save_name:  the name of the file to load the model
    :return: the loaded model
    """
    import joblib
    return joblib.load(f'{root_dir}{model_params["gen_path_y"]}{save_name}.joblib')


def predict_power_production(model_params: dict, regressor: BaseEstimator, years: [int], generate_input: bool = False, verbose: bool=False) -> pd.Series:
    """
    Predicts the power production for the given years

    :param model_params:  the model parameters, see the default models for more details
    :param regressor: the trained model to use for prediction
    :param years:  the years to predict
    :param generate_input:  if True, the input ecd_enr_model of the model will be generated (else, it will be loaded from local files)
    :param verbose: if True, the function will print more debug information
    :return:  the predicted power production
    """
    mapping_path = f'{root_dir}{model_params["gen_path_x"][0]}plants_{model_params["plant_mapping_year"]}.csv'
    if not exists(mapping_path):
        raise Exception(f'File {mapping_path} not found')
    plants_mapping = pd.read_csv(mapping_path, parse_dates=[model_params['date_col']])
    if generate_input:
        print('Generating predict ecd_enr_model for years: ', years)
        generate_train_data(years, plants_map=plants_mapping, model_params=model_params, load_X=True,
                            load_Y=False, verbose=verbose)
    full_X, X, _, _ = load_model_data(years, model_params, load_X=True, load_Y=False, add_0_capa=False, verbose=verbose)
    # X_0 = mult_capacities(X.copy(), 0)
    # bias = regr.predict(X_0)
    # print('Total bias:', bias,'//', bias.mean())
    print(f'Predicting years {years}...', end='\r')
    y_pred = regressor.predict(X)
    # worsens the results y_pred = y_pred - bias
    date_axis = full_X.index
    y_pred_series = pd.Series(index=pd.DatetimeIndex(date_axis), data=y_pred)
    if model_params['div_by_capa']:
        reindex = [d for d in date_axis if type(d) != int]
        full_capas = load_all_capacities(plants_mapping, reindex)
        full_capas.index = pd.DatetimeIndex(full_capas.index)
        y_pred_series = y_pred_series.multiply(full_capas, axis=0)
    print(f'Predicting years {years} - done                   ')
    return y_pred_series


def load_expected_result(model_params: dict, years: [int], generate_output: bool = False, verbose: bool=False) -> pd.Series:
    """
    Loads the read power production for the given years, in the same format as the predict_power_production output
    Useful to compare the results of the prediction (see data_utils module for other functions)

    :param model_params: the model parameters, see the default models for more details
    :param years:  the years to load
    :param generate_output:  if True, the output ecd_enr_model will be generated (else, it will be loaded from local files)
    :param verbose: if True, the function will print more debug information
    :return: the read power production
    """
    mapping_path = f'{root_dir}{model_params["gen_path_x"][0]}plants_{model_params["plant_mapping_year"]}.csv'
    if not exists(mapping_path):
        raise Exception(f'File {mapping_path} not found')
    plants_mapping = pd.read_csv(mapping_path, parse_dates=[model_params['date_col']])
    if generate_output:
        print('Generating output ecd_enr_model for years: ', years)
        generate_train_data(years, plants_map=plants_mapping, model_params=model_params, load_X=False,
                            load_Y=True, verbose=verbose)
    _, _, full_Y, Y = load_model_data(years, model_params, load_X=False, load_Y=True, add_0_capa=False, verbose=verbose)
    y_series = pd.Series(index=pd.DatetimeIndex(full_Y.index), data=Y)
    if model_params['div_by_capa']:
        reindex = [d for d in y_series.index if type(d) != int]
        full_capas = load_all_capacities(plants_mapping, reindex)
        full_capas.index = pd.DatetimeIndex(full_capas.index)
        y_series = y_series.multiply(full_capas, axis=0)
    return y_series


def train_predict_pipeline(model_params: dict, predict_years: [int], generate_input: bool = False, verbose: bool=False) -> pd.Series:
    """
    Trains the model and predicts the power production for the given years

    :param model_params: the model parameters, see the default models for more details
    :param predict_years:  the years to predict
    :param generate_input:  if True, the input/output ecd_enr_model of the model will be generated (else, it will be loaded from local files)
    :param verbose: if True, the function will print more debug information
    :return: the predicted power production
    """
    regressor = train_model(model_params, generate_input=generate_input, verbose=verbose)
    save_model(model_params, regressor)
    y_pred = predict_power_production(model_params, regressor, predict_years, generate_input=generate_input, verbose=verbose)
    return y_pred


def generate_production_files(models: [dict], train_years: [int], predict_years: [int], save_to: str = None,
                              generate_learning_model_input: bool = False, verbose: bool=False) -> pd.DataFrame:
    """
    Generates the production files for the given models and years

    :param models:  the models to use for prediction (see the default models for more details)
    :param train_years: the years to add to the output power production, concatenated with the predicted years
    <strong>This doesn't affect the training of the model (model.train_years parameter)</strong>
    :param predict_years: the years to predict
    :param save_to:  the name of the file to save the output to (if None, the output will not be saved)
    :param generate_learning_model_input: if True, the input/output ecd_enr_model of the model will be generated (else, it will be loaded from local files)
    :param verbose: if True, the function will print more debug information
    :return: the predicted power production for train_years and predict_years
    """
    types = [model['type'] for model in models]
    final_data = []
    for model in models:
        print(f'>>> Predicting production ecd_enr_model for {model["type"]} model...')
        if generate_learning_model_input:
            print('!!! Model input will be generated, this may take a while.')
        y_pred = train_predict_pipeline(model, predict_years, generate_input=generate_learning_model_input, verbose=verbose)
        final_data.append(y_pred)
    final_data = {types[i]: final_data[i] for i in range(len(types))}
    final_data = pd.concat(final_data, axis=1)
    print('Loading training years ecd_enr_model...')
    years = [f'prod_{year}' for year in train_years]
    pronovo_data = load_all_pronovo_files(years, types=types, verbose=verbose)
    final_data = pd.concat([pronovo_data, final_data], axis=0).sort_index()
    if save_to is not None:
        save_power_data(final_data, path=f'{root_dir}export/{save_to}')
    print('Done!')
    return final_data
