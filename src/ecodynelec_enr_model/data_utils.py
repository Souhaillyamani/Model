"""
This module contains utilities for ecd_enr_model manipulation and visualization
"""

from os import listdir

import pandas as pd
from matplotlib import pyplot as plt


def get_correlation_results(y_series: pd.Series, y_pred_series: pd.Series) -> pd.Series:
    """
    Gets the correlation results between y_series and y_pred_series

    :param y_series:  the actual series
    :param y_pred_series:  the predicted series
    :return: A series containing:
        - The actual sum, the predicted sum, the difference between the two in percentage
        - Monthly, weekly, daily and hourly correlation
    """
    stats = pd.Series()
    stats['Actual sum'] = y_series.sum()
    stats['Predicted'] = y_pred_series.sum()
    stats['Diff (%)'] = (y_pred_series.sum() - y_series.sum()) * 100 / y_series.sum()
    stats['Monthly correlation'] = y_series.resample('M').sum().corr(y_pred_series.resample('M').sum())
    stats['Weekly correlation'] = y_series.resample('W').sum().corr(y_pred_series.resample('W').sum())
    stats['Daily correlation'] = y_series.resample('D').sum().corr(y_pred_series.resample('D').sum())
    stats['Hourly correlation'] = y_series.corr(y_pred_series)
    return stats


def plot_predictions(y_series: pd.Series, y_pred_series: pd.Series, date_label: str, title: str, scale: int=4, actual_series_label: str= 'Actual'):
    """
    Plots the predictions and the actual series
    :param y_series: the actual series
    :param y_pred_series:  the predicted series
    :param date_label: the date label for the figures
    :param title: the name of the series (e.g 'Wind')
    :param scale: the scale of the plot (6 for monthly to daily, 5 for monthly, 4 for monthly to hourly, 3 for weekly to hourly, 2 for daily to hourly, 1 for hourly)
    :param actual_series_label: the label for the actual series (can indicate the ecd_enr_model source name)
    """
    if y_series is not None:
        compare_plot_predictions({
            actual_series_label: y_series
        }, y_pred_series, date_label, title=title, scale=scale)
    else:
        compare_plot_predictions({}, y_pred_series, date_label, title=title, scale=scale)


def compare_plot_predictions(compare_to_series: dict, y_pred_series: pd.Series, date_label: str, title: str, scale: int=4):
    """
    Plots the predicted series and the series contained in compare_to_series

    :param compare_to_series: a dict containing the series to compare to and their plot label (e.g. {'Actual': y_series})
    :param y_pred_series: the predicted series
    :param date_label: the date label for the figures
    :param title: the name of the series (e.g 'Wind')
    :param scale: the scale of the plot (6 for monthly to daily, 5 for monthly, 4 for monthly to hourly, 3 for weekly to hourly, 2 for daily to hourly, 1 for hourly)
    """
    legend = list(compare_to_series.keys()) + ['Predicted']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["tab:blue", "tab:green", "tab:red"])
    if scale == 5:
        fig, axes = plt.subplots(1, 1, figsize=(16, 4))
        axes = [axes]
    else:
        fig, axes = plt.subplots(scale, 1, figsize=(16, 4 * scale))
    for i, ax in enumerate(axes):
        if scale > i:
            if i == 0:
                freq = 'M'
                xlabel = 'Month'
            elif i == 1:
                freq = 'W'
                xlabel = 'Week'
            elif i == 2:
                freq = 'D'
                xlabel = 'Day'
            else:
                freq = 'H'
                xlabel = 'Hour'
            ax.set_title(f'{title} electricity production - {date_label} {xlabel.lower()} prediction')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('kWh')
            for label, y_series in compare_to_series.items():
                y_series.resample(freq).sum().plot(ax=ax)
            y_pred_series.resample(freq).sum().plot(ax=ax, color='tab:orange')
            ax.legend(legend)
            if (scale == 6 and i == 2) or scale == 5:
                break
    plt.tight_layout()
    plt.show()

    # """
    # Plots the predicted series and the series contained in compare_to_series
    # TODO CLEAN
    #
    # :param compare_to_series: a dict containing the series to compare to and their plot label (e.g. {'Actual': y_series})
    # :param y_pred_series: the predicted series
    # :param date_label: the date label for the figures
    # :param title: the name of the series (e.g 'Wind')
    # :param scale: the scale of the plot (6 for monthly to daily, 5 for monthly, 4 for monthly to hourly, 3 for weekly to hourly, 2 for daily to hourly, 1 for hourly)
    # """
    # import matplotlib as mpl
    # mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["tab:blue", "tab:green", "tab:red"])
    # if scale == 5:
    #     plt.figure(figsize=(16, 4))
    # else:
    #     plt.figure(figsize=(16, 4 * 4))
    # i = scale * 100 + 11
    # # Monthly
    # if scale > 3:
    #     if scale != 5:
    #         plt.subplot(i)
    #     plt.title(f'{title} electricity production - {date_label} monthly prediction')
    #     plt.xlabel('Month')
    #     plt.ylabel('kWh')
    #     for y_series in compare_to_series.values():
    #         y_series.resample('M').sum().plot()
    #     y_pred_series.resample('M').sum().plot(color='tab:orange')
    #     legend = list(compare_to_series.keys())
    #     legend.append('Predicted')
    #     plt.legend(legend)
    #     plt.ylim(bottom=0)
    #     i += 1
    #     if scale == 5:
    #         return
    # # Weekly
    # if scale > 2:
    #     plt.subplot(i)
    #     plt.title(f'{title} electricity production - {date_label} weekly prediction')
    #     plt.xlabel('Week')
    #     plt.ylabel('kWh')
    #     for y_series in compare_to_series.values():
    #         y_series.resample('W').sum().plot()
    #     y_pred_series.resample('W').sum().plot(color='tab:orange')
    #     legend = list(compare_to_series.keys())
    #     legend.append('Predicted')
    #     plt.legend(legend)
    #     i += 1
    # # Daily
    # if scale > 1:
    #     plt.subplot(i)
    #     plt.title(f'{title} electricity production - {date_label} daily prediction')
    #     plt.xlabel('Day')
    #     plt.ylabel('kWh')
    #     for y_series in compare_to_series.values():
    #         y_series.resample('D').sum().plot()
    #     y_pred_series.resample('D').sum().plot(color='tab:orange')
    #     legend = list(compare_to_series.keys())
    #     legend.append('Predicted')
    #     plt.legend(legend)
    #     i += 1
    #     if scale == 6:
    #         return
    # # Hourly
    # plt.subplot(i)
    # plt.title(f'{title} electricity production - {date_label} hourly prediction')
    # plt.xlabel('Hour')
    # plt.ylabel('kWh')
    # for y_series in compare_to_series.values():
    #     y_series.plot()
    # y_pred_series.plot(color='tab:orange')
    # legend = list(compare_to_series.keys())
    # legend.append('Predicted')
    # plt.legend(legend)
    # plt.show()


def mult_input_matrix(X, factor, n_time_indic):
    """
    Multiplies the input matrix by a factor, but leaves the time indicators unchanged
    :param X:  the input matrix
    :param factor:  the factor to multiply by
    :param n_time_indic: the number of time indicators to keep (at the end of the matrix)
    :return: the multiplied matrix
    """
    prev_hour = X[:, -n_time_indic:]
    X = X * factor
    X[:, -n_time_indic:] = prev_hour
    return X


def mult_output_vector(y, factor):
    # todo
    return y * factor


def load_EC_time_series_from_dir(from_dir):
    """
    Loads all the .csv time series from the given directory
    This function is made to load the time series from the EC dataset
    :param from_dir: the directory to load the time series from
    :return: the concatenated time series
    """
    result = pd.Series()
    for f in listdir(from_dir):
        print('Reading', f, '...')
        df = pd.read_csv(f'{from_dir}{f}', index_col=0)
        df = df.iloc[1:]
        series = df[df.columns[0]]
        series.index = pd.DatetimeIndex(series.index)
        series = series.astype(float)
        result = pd.concat([result, series], axis=0)
    return result
