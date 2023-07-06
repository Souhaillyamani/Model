"""
This module contains functions to load ecd_enr_model for data_model module.

Functions
---------
load_pronovo_file: load pronovo ecd_enr_model from csv files
load_all_pronovo_files: load all pronovo files in a given directory, applying daily scaling with energy charts ecd_enr_model
save_power_data: saves the results of load_all_pronovo_files in a csv file
load_plants: load the plants ecd_enr_model from a csv file
"""

from math import isnan
from os import listdir, mkdir
from os.path import exists

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.spatial.distance import cdist

transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326")  # MN95 -> WGS84
""" Transformer from MN95 to WGS84 (CH to long/lat coordinate system) """

root_dir = './ecd_enr_model/'
""" The root directory where the ecd_enr_model is stored. './ecd_enr_model/' by default. """

def load_pronovo_file(file: str, types: [str] = ['Wind', 'Solar'], verbose: bool=False) -> pd.DataFrame:
    """
    Load pronovo ecd_enr_model from a csv file.
    Supports years from 2020 to 2022 (historically the format of the files changes every semester).

    :param file: the path of the file to load
    :param types:  The types of plants to extract (in ['Wind', 'Solar'])
    :param verbose:  Whether to print debug information
    :return:  A dataframe containing the pronovo ecd_enr_model for all 'types', indexed by date
    """

    if file.endswith('2020.csv'):
        format = 5 if int(file[-11:-9]) < 5 else 6
    elif file.endswith('2021.csv'):
        format = 3 if int(file[-11:-9]) < 8 else 4
    elif file.endswith('2022.csv'):
        format = 1
    else:
        format = 2
    pronovo_types = ['Wind (kWh)' if tpe == 'Wind' else 'Photovoltaik (kWh)' for tpe in types]
    if format == 5:
        pronovo_data = pd.read_csv(f'{root_dir}{file}', index_col=0, skiprows=2,
                                   encoding='windows-1252', sep=';')
        if verbose:
            print('Load fmt 5', file, end='\r')
        pronovo_types = ['Wind' if tpe == 'Wind' else 'Photovoltaic' for tpe in types]
    elif format == 6:
        pronovo_data = pd.read_csv(f'{root_dir}{file}', index_col=0, skiprows=10,
                                   encoding='windows-1252', sep=';')
        if verbose: print('Load fmt 6', file, end='\r')
    elif format == 3:
        pronovo_data = pd.read_csv(f'{root_dir}{file}', index_col=0, skiprows=16,
                                   encoding='windows-1252', sep=';')
        if verbose: print('Load fmt 3', file, end='\r')
    elif format == 4:
        pronovo_data = pd.read_csv(f'{root_dir}{file}', index_col=0, skiprows=18,
                                   encoding='windows-1252', sep=';')
        if verbose: print('Load fmt 4', file, end='\r')
    elif format == 1:
        pronovo_data = pd.read_csv(f'{root_dir}{file}', index_col=0, skiprows=17,
                                   encoding='windows-1252', sep=';')
        if verbose: print('Load fmt 1', file, end='\r')
    elif format == 2:
        pronovo_data = pd.read_csv(f'{root_dir}{file}', index_col=1, skiprows=1,
                                   encoding='windows-1252', sep=';')
        if verbose: print('Load fmt 2', file, end='\r')
        pronovo_types = ['-A.Windturbine [kWh]' if tpe == 'Wind' else '-A.Photovoltaik [kWh]' for tpe in types]
    else:
        raise Exception('Unknown format')
    pronovo_data.index = pd.to_datetime(pronovo_data.index, format='%d.%m.%Y %H:%M')
    pronovo_data = pronovo_data[pronovo_types]
    for i in range(len(types)):
        pronovo_data.rename(columns={pronovo_types[i]: types[i]}, inplace=True)
    pronovo_data = pronovo_data.applymap(
        lambda x: x if type(x) == float else float(x.replace('\'', '').replace('’', '')))
    pronovo_data = pronovo_data.resample('H').sum()
    pronovo_data = pronovo_data.iloc[:-1]  # last value is first hour if the next month
    return pronovo_data


def load_all_pronovo_files(dirs: [str], types: [str] = ['Wind', 'Solar'], verbose: bool=False) -> pd.DataFrame:
    """
    Loads all pronovo files in a given directory, applying daily scaling with energy charts ecd_enr_model (the hourly variation
    comes from the pronovo ecd_enr_model, and the daily total from energy charts ecd_enr_model, if available).
    The scaling is done with csv files starting by "EC". All other csv files are considered as pronovo files.

    :param dirs: The directories to load the ecd_enr_model from
    :param types:  The types of plants to extract (in ['Wind', 'Solar'])
    :param verbose:  Whether to print debug information
    :return:  A dataframe containing the pronovo ecd_enr_model for all 'types', indexed by date
    """

    final_data = []
    for dir in dirs:
        Ys = []
        scalers = []
        for f in listdir(f'{root_dir}{dir}'):
            if f.endswith('.csv'):
                if f.startswith('EC'):
                    if any([tpe in f for tpe in types]):
                        if verbose: print('Found scaler', f)
                        scalers.append(f)
                    continue
                f_y = load_pronovo_file(f'{dir}/{f}', types, verbose=verbose)
                Ys.append(f_y)
        pronovo_data = pd.concat(Ys).sort_index()
        for i in range(len(scalers)):
            scaler = scalers[i]
            col = [tpe for tpe in types if tpe in scaler][0]
            if verbose: print('Applying scaler', scaler)
            ec_data = pd.read_csv(f'{root_dir}{dir}/{scaler}', skiprows=1, index_col=0)['Énergie (GWh)']
            daily_y = pronovo_data[col].resample('D').sum()
            ec_data.index = daily_y.index
            ec_data = ec_data * 1000000
            factors = ec_data / daily_y
            # Ajuster l'index pour inclure l'heure 23:00 du dernier jour
            adjusted_dates = pd.date_range(start=factors.index[0], end=factors.index[-1] + pd.Timedelta(hours=23),
                                           freq='H')
            # Réindexer la série resamplée avec l'index ajusté
            resampled_dates = factors.reindex(adjusted_dates, method='ffill')
            pronovo_data[col] = pronovo_data[col].multiply(resampled_dates)
        final_data.append(pronovo_data)
    if verbose: print('Done!                   ')
    return pd.concat(final_data).sort_index()


def save_power_data(power_data: pd.DataFrame, path: str = '{root_dir}export/enr_prod_2017-2022.csv'):
    """
    Saves the power production ecd_enr_model to a csv file

    :param power_data:  the power production ecd_enr_model
    :param path:  the path to save the ecd_enr_model to (creates the parent directory if it does not exist)
    """
    print(f'Saving to {path}...')
    if not exists(path[:path.rfind('/')]):
        mkdir(path[:path.rfind('/')])
    power_data.to_csv(path)


def load_plants(model_params: dict) -> pd.DataFrame:
    """
    Loads the plants map from the given file, and filters it by the given category

    :param model_params:  the model parameters (indicating the file to load and the category to filter)
    :return: the plants map
    """
    plants_map = pd.read_csv(model_params['file'], parse_dates=[model_params['date_col']])
    plants_map = plants_map[plants_map[model_params['category_col']] == model_params['category']]
    return plants_map


def load_all_capacities(plants_map: pd.DataFrame, time_index: pd.DatetimeIndex) -> pd.Series:
    """
    Loads the cumulated capacities timeline of all plants in the given map

    :param plants_map: the plants map (as returned by load_plants)
    :param time_index: the time index to load the capacities for
    :return: the cumulated capacities timeline
    """

    initial_capacity = plants_map[plants_map['BeginningOfOperation'] < time_index[0]]['TotalPower'].sum()
    prod_sources = plants_map.groupby(['SubCategory', 'BeginningOfOperation'])['TotalPower'].sum()
    prod_sources = prod_sources.reset_index().set_index('BeginningOfOperation')
    prod_sources.index = pd.DatetimeIndex(prod_sources.index)
    capacities = prod_sources.loc[time_index[0]:time_index[-1], 'TotalPower'].cumsum()
    capacities = capacities.reindex(time_index).ffill()
    capacities = capacities.fillna(0) + initial_capacity
    return capacities


def get_weather_grid(file: str, year: int, verbose: bool=False) -> pd.DataFrame:
    """
    Loads the given .cdf file and filters it by the given year and region of interest

    :param file: the .cdf file to load, containing the weather ecd_enr_model
    :param year: the year to filter the ecd_enr_model by
    :param verbose: whether to print debug information
    :return: the weather ecd_enr_model grid
    """

    if verbose: print('Read:', file)
    #else: print('Process:', file, end='\r')
    ds2 = xr.open_dataset(f'{root_dir}{file}')
    factors = ds2.to_dataframe().dropna()
    del ds2
    if verbose: print('Filter ecd_enr_model...', factors.shape, end='\r')
    # Retain only the values in the region of interest (Switzerland), and in the year 2021
    mask = (factors.index.get_level_values('longitude') > 5.9865) & (
            factors.index.get_level_values('longitude') < 10.4921) & (
                   factors.index.get_level_values('latitude') > 45.8175) & (
                   factors.index.get_level_values('latitude') < 47.8085) & (
                   factors.index.get_level_values('time').year == year)
    if verbose: print('Keeping', mask.sum(), '/', len(mask), 'values', end='\r')
    factors = factors.loc[factors.index[mask]].copy()
    return factors


def map_plants_to_weather_grid(weather_grid: pd.DataFrame, plants_map : pd.DataFrame, n_nearests: int=4, verbose: bool=False) -> pd.DataFrame:
    """
    Finds the n_nearests positions of the weather grid for each plant in the given plants map, and updates it with the found positions

    :param weather_grid: the weather grid to use as reference position map (as returned by get_weather_grid)
    :param plants_map: the plants map (as returned by load_plants)
    :param n_nearests: the number of nearest positions to find for each plant (has an impact on the computation time)
    :param verbose: whether to print debug information
    :return: the updated plants map, with 'pos' (nearest position) and 'all_pos' (n_nearests positions) columns, mapped to the weather grid
    """

    if verbose: print('Mapping plant poses... In', weather_grid.shape, 'values                                                     ')
    else: print('Mapping plant poses...                                                                                            ', end='\r')
    wd40 = weather_grid.reset_index()
    grid = wd40[['longitude', 'latitude']].drop_duplicates(subset=['longitude', 'latitude'])
    del wd40

    global k
    k = 0
    szh = len(plants_map)

    def get_avg_load_factor(row, n_nearests):
        global k
        if verbose: print('Progress:', k, '/', szh, end='\r')
        if isnan(row['_x']) or isnan(row['_y']):
            # return center of switzerland
            lon, lat = 8.231974, 46.798562
        else:
            # Convertir les coordonnées en degrés décimaux
            lat, lon = transformer.transform(row['_x'], row['_y'])
        centrale_position = np.array([lon, lat])
        distances = cdist([centrale_position], grid[['longitude', 'latitude']])
        distances = distances[0]
        indices_plus_proches = np.argpartition(distances, n_nearests)[:n_nearests]
        positions_plus_proches = grid.iloc[indices_plus_proches]
        k += 1
        return positions_plus_proches

    # Attribution des positions les plus proches aux centrales
    pos_matrix = plants_map.apply(get_avg_load_factor, axis=1, n_nearests=n_nearests)
    del grid
    plants_map['pos'] = [str((p.iloc[0]['longitude'], p.iloc[0]['latitude'])) for p in pos_matrix.values]
    plants_map['all_pos'] = [tuple([(p.iloc[i]['longitude'], p.iloc[i]['latitude']) for i in range(len(p))]) for p in
                             pos_matrix.values]
    del pos_matrix
    print('Mapping plant poses - done                   ')
    return plants_map


def process_file(f, year, plants_map, verbose):
    """
    Proxy function to load_X, used for multiprocessing
    """
    f_X, X = load_X(plants_map, f'{f}', year, verbose=verbose)
    del X
    return f_X


def load_all_X(plants_map: pd.DataFrame, dir: str, year: int, verbose: bool=False) -> (pd.DataFrame, np.ndarray):
    """
    Loads weather ecd_enr_model from all files in the given directory, and maps it to the given plants_map

    :param plants_map: the plants map (as returned by map_plants_to_weather_grid)
    :param dir: the directory containing the .cdf files to load
    :param year: the year to load
    :param verbose: whether to print debug information
    :return: the mapped weather ecd_enr_model, with n_nearests columns for each plant
    """
    import multiprocessing

    # Define the number of processes to use
    num_processes = multiprocessing.cpu_count() - 2

    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=num_processes)

    # Create a list of filenames
    files = [(f'{root_dir}{dir}/{f}', year, plants_map, verbose) for f in listdir(f'{root_dir}{dir}') if
             f.endswith('.nc')]

    if verbose: print('Using thread pool with', num_processes, 'processes', end='\r')
    # Process the files in parallel
    Xs = pool.starmap(process_file, files)
    # Xs = []
    # for f in listdir(f'{root_dir}{dir}'):
    #    if f.endswith('.nc'):
    #        Xs.append(process_file(f'{root_dir}{dir}/{f}', year, wind_plants))
    #        print('Done', f)
    if verbose: print('Done!                   ')

    # Close the pool to free up resources
    pool.close()
    pool.join()

    full_X = pd.concat(Xs)
    X = full_X.values
    return full_X, X


def load_X(plants_map: pd.DataFrame, file: str, year: int, verbose: bool=False) -> (pd.DataFrame, np.ndarray):
    """
    Loads the weather ecd_enr_model for the given year, from the given .cdf file, and maps it to the given plants_map

    :param plants_map: the plants map (as returned by map_plants_to_weather_grid)
    :param file: the .cdf file to load
    :param year: the year to load
    :param verbose: whether to print debug information
    :return: the mapped weather ecd_enr_model, with n_nearest columns for each plant
    """

    # Load weather grid
    weather_grid = get_weather_grid(file, year, verbose=verbose)
    if verbose: print(f'Processing weather ecd_enr_model from {file}...')
    weather_grid = weather_grid.reset_index().set_index(['longitude', 'latitude', 'time'])
    weather_grid = weather_grid.sort_index()
    time_index = weather_grid.index.levels[2]
    load_factors = {}
    dtrange = pd.date_range(start=time_index[0], end=time_index[-1] + pd.Timedelta(hours=24), freq='D')

    agg_power_plants = plants_map.groupby(['pos', 'all_pos']).agg({'TotalPower': 'sum'}).reset_index()
    le = len(agg_power_plants)
    for k in agg_power_plants.index:
        # print('Remaster X for plant group', k, '/', le, end='\r')
        # Prod capacities at pos
        pos = agg_power_plants.loc[k]['pos']
        debug = False  # (pos == "(8.5, 46.5)")
        pos_str = str(pos)
        import ast
        allp = ast.literal_eval(str(agg_power_plants.loc[k]['all_pos']))
        for pos in allp:
            a, b = pos
            df = weather_grid.loc[a, b].copy()
            df = df[df.columns[0]]
            load_factors[str(k) + '_' + str(pos)] = df
    full_X = pd.DataFrame(load_factors, index=time_index)
    del load_factors
    X = full_X.values
    return full_X, X


def generate_train_data(years, model_params, load_X=True,
                        load_Y=True, plants_map=None, do_map_plants_to_weather_grid=False, verbose: bool=False):
    """
    Generates the input and/or output ecd_enr_model for the given years, and the given model parameters
    The ecd_enr_model is saved in the path defined in the model parameters

    :param years: the years to generate the ecd_enr_model for
    :param model_params:  the model parameters
    :param load_X:  whether to load the input ecd_enr_model
    :param load_Y:  whether to load the output ecd_enr_model
    :param plants_map: the plants map, if None it will be loaded according to the 'plant_mapping_year' parameter, and mapped to the weather grid of this year
    Note that this operation can be quite long, so it is recommended to load the plants map once, and then pass it to this function
    :param do_map_plants_to_weather_grid: whether to map the plants map to the weather grid. If plants_map is None, this parameter is ignored.
    Note that this operation can be quite long.
    :param verbose: whether to print debug information
    """
    if plants_map is None:
        if verbose: print('Load plants...')
        plants_map = load_plants(model_params)
        do_map_plants_to_weather_grid = True
    if do_map_plants_to_weather_grid:
        if verbose: print('Mapping plant poses...')
        else: print('Mapping plant poses...', end='\r')
        year = model_params["plant_mapping_year"]
        dir = f'{model_params["name_prefixs"][0]}_{year}'
        found = False
        for f in listdir(f'{root_dir}{dir}'):
            if f.endswith('.nc'):
                weather_grid = get_weather_grid(f'{dir}/{f}', year, verbose=verbose)
                plants_map = map_plants_to_weather_grid(weather_grid, plants_map, n_nearests=model_params['n_nearests'], verbose=verbose)
                del weather_grid
                if verbose: print(f'Save to ecd_enr_model/{model_params["gen_path_y"]}plants_{year}.csv...')
                plants_map.to_csv(f'{root_dir}{model_params["gen_path_y"]}plants_{year}.csv...')
                found = True
                break
        if not found:
            raise Exception('No weather file found in ' + dir)

    for year in years:
        if not verbose: print(f'Loading ecd_enr_model for {year}...', end='\r')
        if load_X:
            for i in range(len(model_params['name_prefixs'])):
                name_prefix = model_params['name_prefixs'][i]
                if verbose: print(f'Load X for {name_prefix}_{year}...')
                full_X, _ = load_all_X(plants_map, f'{name_prefix}_{year}', year, verbose=verbose)
                if verbose: print(f'Save to full_X_{year}.csv...')
                full_X.to_csv(f'{root_dir}{model_params["gen_path_x"][i]}full_X_{year}.csv')
        if load_Y:
            if verbose: print(f'Load Y for {year}...')
            full_y = load_all_pronovo_files([f'prod_{year}'], types=[model_params['type']], verbose=verbose)
            if model_params['div_by_capa']:
                full_capas = load_all_capacities(plants_map, full_y.index)
                full_y = full_y.divide(full_capas, axis=0)
            if verbose: print(f'Save to ecd_enr_model/{model_params["gen_path_y"]}full_y_prod_{year}.csv...')
            full_y.to_csv(f'{root_dir}{model_params["gen_path_y"]}full_y_prod_{year}.csv...')


def load_model_data(years: [int], model_params: dict, load_X: bool=True, load_Y: bool=True, add_0_capa: bool=False, verbose: bool=False) -> (pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray):
    """
    Loads the model input and/or output ecd_enr_model for the given years, and the given model parameters
    The ecd_enr_model is loaded from the files generated by generate_train_data, from the paths defined in model_params

    :param years:  the years to load
    :param model_params: the model parameters
    :param load_X: whether to load the input ecd_enr_model
    :param load_Y: whether to load the output ecd_enr_model
    :param add_0_capa: whether to add fake zeros in both input and output ecd_enr_model ([0 0 ... 0 0] -> 0), to ensure no wind/solar radiation gives no production
    :param verbose: whether to print debug info
    :return: a tuple (full_X, X, full_Y, Y), where full_X and full_Y are the full input and output dataframes, and X and Y are the corresponding numpy arrays
    """
    full_X = None
    full_Y = None
    steps = []
    for year in years:
        print(f'Load {year}...')
        if load_X:
            size = len(model_params['gen_path_x'])
            fXs = []
            for i in range(size):
                name_prefix = model_params['gen_path_x'][i]
                spc = pd.read_csv(f'{root_dir}{name_prefix}full_X_{year}.csv', index_col=0)
                spc.rename(columns=(lambda s: str(i) + '_' + s), inplace=True)
                fXs.append(spc)
            fX = pd.concat(fXs, axis=1)
            if add_0_capa:
                steps = np.arange(0, len(fX.index), 10)
                if verbose: print('Add', len(steps), 'zero capas to X')
                for j in steps:
                    tarpaing = fX.iloc[j].copy()
                    tarpaing[fXs[0].columns] = 0
                    fX.loc[len(fX.index)] = tarpaing
            full_X = fX if full_X is None else pd.concat([full_X, fX], axis=0)
        if load_Y:
            fY = pd.read_csv(f'{root_dir}{model_params["gen_path_y"]}full_y_prod_{year}.csv', index_col=0)
            if add_0_capa:
                if verbose: print('Add', len(steps), 'zero capas to Y')
                zeros = pd.DataFrame(index=steps, data=0, columns=fY.columns)
                fY = pd.concat([fY, zeros], axis=0)
            fY.columns = ['prod']
            full_Y = fY if full_Y is None else pd.concat([full_Y, fY], axis=0)
    if load_X:
        idx = pd.DatetimeIndex(full_X.index)
        if verbose: print('Using model params', model_params)
        if model_params['add_hour']:
            full_X['hour'] = np.sin(np.array([d.hour for d in idx]) * np.pi / 24 - np.pi / 2) * 10
            # full_X['hour'] = [d.hour for d in idx]
        # full_X['week'] = [d.week for d in idx]
        full_X['month'] = [d.month for d in idx]
        # seq = [d.week for d in idx]
        # print(seq)
        # full_X['week'] = np.sin(np.array(seq)*np.pi/52 - np.pi/2)*10
        X = full_X.values
        X = np.nan_to_num(X, copy=False)
        # from sklearn.preprocessing import StandardScaler
        # X = StandardScaler().fit_transform(X)
        #X[:, -1] = full_X['month'].values
        #if model_params['add_hour']:
        #    X[:, -2] = full_X['hour'].values
    else:
        X = None
    if load_Y:
        Y = full_Y.values
        Y = Y.reshape(-1)
    else:
        Y = None
    return full_X, X, full_Y, Y
