'''
RUBICS (RUB/CS): Random Uncertainty Benefit / Cost Simulator
RUBICS is intended to be an Open Source program for economists to undertake Monte Carlo & Distributional Benefit-Cost
Analysis, enabling the sharing and peer review of inputs and outputs.
Copyright (C) 2023
Author: Samuel Miller
Version: Star (1.0.0)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

Author: Samuel Miller
Email: smiller.econ@gmail.com
In Memoriam: Timothy Miller (1996-2012) & James Miller (1994-2022)

The File is organised into sections:
    - Import Packages
    - def functions and variables for individual sections
    - Layouts for individual sections
    - Layout for the main window
    - While Loop, divided into individual sections

sicpy random number tutorial:
https://stackoverflow.com/questions/16016959/scipy-stats-seed

graphing in PSGUI using matplotlib
https://github.com/PySimpleGUI/PySimpleGUI/issues/5410
https://matplotlib.org/stable/plot_types/index.html
'''

######################################################################################################################
# SECTION 1: IMPORT PACKAGES
######################################################################################################################

import PySimpleGUI as sg
import numpy as np
from scipy.stats import uniform
from scipy.stats import triang
from scipy.stats import beta
from scipy.stats import fisk
from scipy.stats import truncnorm
from scipy.stats import foldnorm
from scipy.stats import expon
from scipy.stats import lognorm
from scipy.stats import rv_continuous
from scipy import interpolate
from json import (load as jsonload, dumps as jsondumps, dump as jsondump)
import os
import datetime as dt
import re
import ast
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.pyplot import savefig
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from openpyxl import load_workbook
#from sklearn import linear_model
import requests


tbar_icon = os.getcwd() + '/RUBICS ICON.ico'

######################################################################################################################
# SECTION 2: DEF FUNCTIONS
######################################################################################################################

### LOADING/EXPORTING SECTION ----------------------------------------------------------------------------------------

# Setting file
# First create the Default settings dictionary, and the settings key that links between the default and modified settings dictionaries and the GUI's values dictionary.

# Define user_settings as a blank dictionary to incorporate the relevant settings supplied in a JSON file.
user_settings = {}
new_values = {}
event_list = []
MC_update_display = []

def apply_user_settings():
    global DA_table
    global pop_mean_income
    global income_table
    global income_weight_parameter
    global n_pr_groups
    global n_scenarios
    global n_params
    global n_functions
    global MC_update_display

    # Add the discount rate data from the user_settings_file to the layout.
    if 'discount_rate_settings' in user_settings.keys():
        for k_pair in [('discounting_method', '-DR_TYPE-'), ('discount_rate', '-DR_MAIN-'),
                       ('discount_rate_sigma', '-DR_SIG-'), ('dr_step_thereafter', '-DR_STEP_BASE-'),
                       ('MC_PDF', '-DR_MC_TYPE-')]:
            if k_pair[0] in user_settings['discount_rate_settings'].keys():
                window[f'{k_pair[1]}'].update(user_settings['discount_rate_settings'][f'{k_pair[0]}'])
        # Multiline inputs require special translation from lists to strings.
        for k_pair in [('dr_step_range', '-DR_STEP_RANGE-'), ('dr_step_rates', '-DR_STEP_RATE-')]:
            if k_pair[0] in user_settings['discount_rate_settings'].keys():
                window[f'{k_pair[1]}'].update(
                    '\n'.join(map(str, user_settings['discount_rate_settings'][f'{k_pair[0]}'])))
        # Updating the PDF setting requires the MC_holding_dictionary to be updated, and the get_pdf_settings window to be opened & closed in order for the main window layout to display.
        if 'MC_PDF' in user_settings['discount_rate_settings'].keys():
            for k_pair in [('PDF_min', '-DR_PDFMIN-'), ('PDF_max', '-DR_PDFMAX-'), ('PDF_mean', '-DR_PDFMEA-'),
                           ('PDF_stdev', '-DR_PDFSIG-'), ('PDF_mode', '-DR_PDFMOD-'), ('PDF_median', '-DR_PDFMED-'),
                           ('PDF_shape', '-DR_PDFSHA-'), ('PDF_scale', '-DR_PDFSCA-'),
                           ('PDF_lambda', '-DR_PDFRAT-'), ('PDF_rand_each_t', '-DR_PDF_REACHT-'),
                           ('PDF_Xs', '-DR_PDFX-'), ('PDF_Ys', '-DR_PDFY-')]:
                if k_pair[0] in user_settings['discount_rate_settings'].keys():
                    MC_holding_dict.update(
                        {f'{k_pair[1]}': user_settings['discount_rate_settings'][f'{k_pair[0]}']})
            if user_settings['discount_rate_settings']['MC_PDF'] != 'None':
                MC_update_display.append('-DR_MC_TYPE-')

    # Add the distribution analysis data from the user_settings_file to the layout
    if 'distribution_analysis_settings' in user_settings.keys():
        if 'simple_weight_matrix' in user_settings['distribution_analysis_settings'].keys():
            DA_table = user_settings['distribution_analysis_settings']['simple_weight_matrix']
            DA_table[0][0] = 'DA Table'
            window['-DA_GEOGRP-'].update('\n'.join(DA_table[0][1:len(DA_table)]))
            window['-DA_STKGRP-'].update('\n'.join(list(DA_table[n][0] for n in range(1, len(DA_table)))))

        if 'population_average_income' in user_settings['distribution_analysis_settings'].keys():
            pop_mean_income = user_settings['distribution_analysis_settings']['population_average_income']
        if 'income_weighting_parameter' in user_settings['distribution_analysis_settings'].keys():
            income_weight_parameter = user_settings['distribution_analysis_settings']['income_weighting_parameter']
        if 'subgroup_average_income' in user_settings['distribution_analysis_settings'].keys():
            income_table = user_settings['distribution_analysis_settings']['subgroup_average_income']
            income_table[0][0] = 'Income Table'
            window['-DA_GEOGRP-'].update('\n'.join(income_table[0][1:len(income_table)]))
            window['-DA_STKGRP-'].update('\n'.join(list(income_table[n][0] for n in range(1, len(income_table)))))
            # Update the DA_table accordingly.
            DA_table = income_table
            for i in range(1, len(income_table)):
                for j in range(1, len(income_table[1])):
                    try:
                        DA_table[i][j] = (pop_mean_income / income_table[i][j]) ** income_weight_parameter
                    except (ZeroDivisionError, ValueError):
                        DA_table[i][j] = np.nan

    # Add the function from the user_settings file to the layout
    if 'functions' in user_settings.keys():
        # Get a list of functions given in the user_settings_file
        fnc_list_user_settings = [k for k in list(user_settings['functions'].keys()) if 'function_' in k]
        fnc_list_user_settings_ints = list(map(int, [k.replace('function_', '') for k in fnc_list_user_settings]))

        for fnc in fnc_list_user_settings_ints:
            # Add missing function fields to the layout if needed
            while n_functions < fnc:
                n_functions += 1
                add_function()
                if window['-FNC_COLS-'].get_size()[1] < 700:
                    resize = 280 + window['-FNC_COLS-'].get_size()[1]
                    set_size(window['-FNC_COLS-'], (None, resize))
                window.refresh()
                window['-FNC_COLS-'].contents_changed()

            # Add function data from the user_settings_file to the layout
            for k_pair in [('ID', f'-FNC{fnc}_ID-'),
                           ('interpolation_method', f'-FNC{fnc}_INTERP-'),
                           ('smoothing_parameter', f'-FNC{fnc}_SMOOTHSET-'), ('comments', f'-FNC{fnc}_CM-')]:
                if k_pair[0] in user_settings['functions'][f'function_{fnc}'].keys():
                    window[f'{k_pair[1]}'].update(user_settings['functions'][f'function_{fnc}'][f'{k_pair[0]}'])
            for k_pair in [('x_values', f'-FNC{fnc}_XVALS-'), ('y_values', f'-FNC{fnc}_YVALS-')]:
                if k_pair[0] in user_settings['functions'][f'function_{fnc}'].keys():
                    fnc_str = list(map(str, user_settings['functions'][f'function_{fnc}'][f'{k_pair[0]}']))
                    window[f'{k_pair[1]}'].update('\n'.join(fnc_str))

    # Add the parameters from the user_settings file to the layout
    if 'parameters' in user_settings.keys():
        # Get a list of parameters given in the user_settings_file
        par_list_user_settings = [k for k in list(user_settings['parameters'].keys()) if 'parameter_' in k]
        par_list_user_settings_ints = list(map(int, [k.replace('parameter_', '') for k in par_list_user_settings]))

        for par in par_list_user_settings_ints:
            # Add missing parameter fields to the layout if needed
            while n_params < par:
                n_params += 1
                add_parameter()
                if window['-PAR_COLS-'].get_size()[1] < 700:
                    resize = 280 + window['-PAR_COLS-'].get_size()[1]
                    set_size(window['-PAR_COLS-'], (None, resize))
                window.refresh()
                window['-PAR_COLS-'].contents_changed()

            # Add parameter data from the user_settings_file to the layout
            for k_pair in [('ID', f'-PAR{par}_ID-'),  ("MC_PDF", f'-PAR{par}_MC_TYPE-'),
                           ('comments', f'-PAR{par}_CM-')]:
                if k_pair[0] in user_settings['parameters'][f'parameter_{par}'].keys():
                    window[f'{k_pair[1]}'].update(user_settings['parameters'][f'parameter_{par}'][f'{k_pair[0]}'])
            for k_pair in [('period_range', f'-PAR{par}_PRANGE-'),
                           ('value', f'-PAR{par}_PVALUE-')]:
                if k_pair[0] in user_settings['parameters'][f'parameter_{par}'].keys():
                    par_str = list(map(str, user_settings['parameters'][f'parameter_{par}'][f'{k_pair[0]}']))
                    window[f'{k_pair[1]}'].update('\n'.join(par_str))
            if 'MC_PDF' in user_settings['parameters'][f'parameter_{par}'].keys():
                for k_pair in [('PDF_min', f'-PAR{par}_PDFMIN-'), ('PDF_max', f'-PAR{par}_PDFMAX-'),
                               ('PDF_mean', f'-PAR{par}_PDFMEA-'), ('PDF_stdev', f'-PAR{par}_PDFSIG-'),
                               ('PDF_mode', f'-PAR{par}_PDFMOD-'), ('PDF_median', f'-PAR{par}_PDFMED-'),
                               ('PDF_shape', f'-PAR{par}_PDFSHA-'), ('PDF_scale', f'-PAR{par}_PDFSCA-'),
                               ('PDF_lambda', f'-PAR{par}_PDFRAT-'), ('PDF_rand_each_t', f'-PAR{par}_PDF_REACHT-'),
                               ('PDF_Xs', f'-PAR{par}_PDFX-'), ('PDF_Ys', f'-PAR{par}_PDFY-')]:
                    if k_pair[0] in user_settings['parameters'][f'parameter_{par}'].keys():
                        MC_holding_dict.update({f'{k_pair[1]}':
                                                    user_settings['parameters'][f'parameter_{par}'][
                                                        f'{k_pair[0]}']})
                if user_settings['parameters'][f'parameter_{par}']['MC_PDF'] != 'None':
                    MC_update_display.append(f'-PAR{par}_MC_TYPE-')

    # Add the global reference prices data from the user_settings_file to the layout.
    if 'reference_prices' in user_settings.keys():
        for k_pair in [('country', '-PR_COUNTRY-'), ('year', '-PR_YEAR-'),
                       ('currency_conversion_range', '-PR_CUR_RANGE-'),
                       ('currency_conversion_measure', '-PR_CUR_POINT-')]:
            if k_pair[0] in user_settings['reference_prices'].keys():
                window[f'{k_pair[1]}'].update(user_settings['reference_prices'][f'{k_pair[0]}'])

        # Get a list of price groups given in the user_settings_file
        pg_list_user_settings = [k for k in list(user_settings['reference_prices'].keys()) if 'price_group_' in k]
        pg_list_user_settings_ints = list(map(int, [k.replace('price_group_', '') for k in pg_list_user_settings]))
        # Get the number of price groups needed from the highest number price group given
        n_pr_groups_user_settings = max(pg_list_user_settings_ints)

        for grp in pg_list_user_settings_ints:
            # Add any missing groups to the layout if needed.
            while grp not in pr_group_dict.keys():
                pr_group_dict[n_pr_groups + 1] = 0
                n_pr_groups += 1
                add_price_group(n_pr_groups)
                add_price_line(n_pr_groups, values['-PR_COUNTRY-'], values['-PR_YEAR-'])
                pr_group_dict[n_pr_groups] += 1
                # Limit the height of the column to 700 pixels
                if window['-PG_COLS-'].get_size()[1] < 700:
                    resize = 242 + window['-PG_COLS-'].get_size()[1]
                    set_size(window['-PG_COLS-'], (None, resize))
                window.refresh()
                window['-PG_COLS-'].contents_changed()

            # Add the price group data from the user_settings_file to the layout.
            for k_pair in [('ID', f'-PG{grp}_ID-'), ('MC_PDF', f'-PG{grp}_MC_TYPE-')]:
                if k_pair[0] in user_settings['reference_prices'][f'price_group_{grp}'].keys():
                    window[f'{k_pair[1]}'].update(
                        user_settings['reference_prices'][f'price_group_{grp}'][f'{k_pair[0]}'])
            if 'MC_PDF' in user_settings['reference_prices'][f'price_group_{grp}'].keys():
                for k_pair in [('PDF_min', f'-PG{grp}_PDFMIN-'), ('PDF_max', f'-PG{grp}_PDFMAX-'),
                               ('PDF_mean', f'-PG{grp}_PDFMEA-'), ('PDF_stdev', f'-PG{grp}_PDFSIG-'),
                               ('PDF_mode', f'-PG{grp}_PDFMOD-'), ('PDF_median', f'-PG{grp}_PDFMED-'),
                               ('PDF_shape', f'-PG{grp}_PDFSHA-'), ('PDF_scale', f'-PG{grp}_PDFSCA-'),
                               ('PDF_lambda', f'-PG{grp}_PDFRAT-'), ('PDF_rand_each_t', f'-PG{grp}_PDF_REACHT-'),
                               ('PDF_Xs', f'-PG{grp}_PDFX-'), ('PDF_Ys', f'-PG{grp}_PDFY-')]:
                    if k_pair[0] in user_settings['reference_prices'][f'price_group_{grp}'].keys():
                        MC_holding_dict.update({f'{k_pair[1]}':
                                                    user_settings['reference_prices'][f'price_group_{grp}'][
                                                        f'{k_pair[0]}']})
                if user_settings['reference_prices'][f'price_group_{grp}']['MC_PDF'] != 'None':
                    MC_update_display.append(f'-PG{grp}_MC_TYPE-')

            # Get a list of price lines in the price groups given in the user_settings_file
            pg_lines_list_user_settings = [k for k in
                                           list(user_settings['reference_prices'][f'price_group_{grp}'].keys()) if
                                           'price_line_' in k]
            pg_lines_list_user_settings_ints = list(
                map(int, [k.replace('price_line_', '') for k in pg_lines_list_user_settings]))
            # Get the number of price lines needed from the highest number of price groups given.
            n_pg_lines_user_settings = max(pg_lines_list_user_settings_ints)

            # Add any missing groups to the layout if needed.
            while n_pg_lines_user_settings > pr_group_dict[grp]:
                add_price_line(grp, values['-PR_COUNTRY-'], values['-PR_YEAR-'])
                pr_group_dict[grp] += 1
                if window['-PG_COLS-'].get_size()[1] < 700:
                    resize = 100 + window['-PG_COLS-'].get_size()[1]
                    set_size(window['-PG_COLS-'], (None, resize))
                window.refresh()
                window['-PG_COLS-'].contents_changed()

            # Add price line data from the user_settings_file to the layout.
            for lin in pg_lines_list_user_settings_ints:
                for k_pair in [('ID', f'-PG{grp}_LIN{lin}_ID-'), ('units', f'-PG{grp}_LIN{lin}_UN-'),
                               ('nominal_value', f'-PG{grp}_LIN{lin}_NV-'), ('currency', f'-PG{grp}_LIN{lin}_CUR-'),
                               ('value_year', f'-PG{grp}_LIN{lin}_CURYR-'),
                               ('adjustment_factor', f'-PG{grp}_LIN{lin}_AF-'),
                               ('MC_PDF', f'-PG{grp}_LIN{lin}_MC_TYPE-'), ('comments', f'-PG{grp}_LIN{lin}_CM-')]:
                    if k_pair[0] in user_settings['reference_prices'][f'price_group_{grp}'][
                        f'price_line_{lin}'].keys():
                        window[f'{k_pair[1]}'].update(
                            user_settings['reference_prices'][f'price_group_{grp}'][f'price_line_{lin}'][
                                f'{k_pair[0]}'])
                if 'MC_PDF' in user_settings['reference_prices'][f'price_group_{grp}'][f'price_line_{lin}'].keys():
                    for k_pair in [('PDF_min', f'-PG{grp}_LIN{lin}_PDFMIN-'),
                                   ('PDF_max', f'-PG{grp}_LIN{lin}_PDFMAX-'),
                                   ('PDF_mean', f'-PG{grp}_LIN{lin}_PDFMEA-'),
                                   ('PDF_stdev', f'-PG{grp}_LIN{lin}_PDFSIG-'),
                                   ('PDF_mode', f'-PG{grp}_LIN{lin}_PDFMOD-'),
                                   ('PDF_median', f'-PG{grp}_LIN{lin}_PDFMED-'),
                                   ('PDF_shape', f'-PG{grp}_LIN{lin}_PDFSHA-'),
                                   ('PDF_scale', f'-PG{grp}_LIN{lin}_PDFSCA-'),
                                   ('PDF_lambda', f'-PG{grp}_LIN{lin}_PDFRAT-'),
                                   ('PDF_rand_each_t', f'-PG{grp}_LIN{lin}_PDF_REACHT-'),
                                   ('PDF_Xs', f'-PG{grp}_LIN{lin}_PDFX-'), ('PDF_Ys', f'-PG{grp}_LIN{lin}_PDFY-')]:
                        if k_pair[0] in user_settings['reference_prices'][f'price_group_{grp}'][
                            f'price_line_{lin}'].keys():
                            MC_holding_dict.update({f'{k_pair[1]}':
                                                        user_settings['reference_prices'][f'price_group_{grp}'][
                                                            f'price_line_{lin}'][f'{k_pair[0]}']})
                    if user_settings['reference_prices'][f'price_group_{grp}'][f'price_line_{lin}']['MC_PDF'] != 'None':
                        MC_update_display.append(f'-PG{grp}_LIN{lin}_MC_TYPE-')

    # Event Model Settings
    if 'event_model' in user_settings.keys():
        # Get lit of events in the event_model dictionary of the uder_settings_file
        event_model_list_user_settings = [k for k in list(user_settings['event_model'].keys())]
        event_model_list_user_settings_ints = list(
            map(int, [k.replace('event_', '') for k in event_model_list_user_settings]))
        # Add the events to the model events dictionary if missing.
        model_events_dict.update(
            {e: 0 for e in event_model_list_user_settings_ints if e not in list(model_events_dict.keys())})
        for evt in event_model_list_user_settings_ints:
            if 'ID' in list(user_settings['event_model'][f'event_{evt}'].keys()):
                model_events_user_settings[f'-EVENT{evt}_ID-'] = user_settings['event_model'][f'event_{evt}']['ID']
            if 'event_depends' in list(user_settings['event_model'][f'event_{evt}'].keys()):
                model_events_user_settings[f'-EVENT{evt}_DEPENDS-'] = user_settings['event_model'][f'event_{evt}'][
                    'event_depends']
            else:
                model_events_user_settings[f'-EVENT{evt}_DEPENDS-'] = []
            # Get list of outcomes.
            outcomes_list_user_settings = [k for k in list(user_settings['event_model'][f'event_{evt}'].keys()) if
                                           'outcome_' in k]
            outcomes_list_user_settings_ints = list(
                map(int, [k.replace('outcome_', '') for k in outcomes_list_user_settings]))
            # Add the outcomes to the model events dictionary is missing
            if max(outcomes_list_user_settings_ints) > model_events_dict[evt]:
                model_events_dict.update({evt: max(outcomes_list_user_settings_ints)})
            for out in outcomes_list_user_settings_ints:
                if 'ID' in list(user_settings['event_model'][f'event_{evt}'][f'outcome_{out}'].keys()):
                    model_events_user_settings[f'-EVENT{evt}_OUTCOME{out}_ID-'] = \
                        user_settings['event_model'][f'event_{evt}'][f'outcome_{out}']['ID']
                    model_outcome_list.append(model_events_user_settings[f'-EVENT{evt}_OUTCOME{out}_ID-'])
                # Get list of scenarios
                outcome_scenarios_list_user_settings = [k for k in list(
                    user_settings['event_model'][f'event_{evt}'][f'outcome_{out}'].keys()) if 'scenario_' in k]
                outcome_scenarios_list_user_settings_ints = list(
                    map(int, [k.replace('scenario_', '') for k in outcome_scenarios_list_user_settings]))
                for scn in outcome_scenarios_list_user_settings_ints:
                    for k_pair in [('periods_without_repeat', f'-EVENT{evt}_OUTCOME{out}_SCN{scn}_NOREPS-'),
                                   ('max_repeats', f'-EVENT{evt}_OUTCOME{out}_SCN{scn}_MAXREPS-')]:
                        if k_pair[0] in user_settings['event_model'][f'event_{evt}'][f'outcome_{out}'][
                            f'scenario_{scn}'].keys():
                            model_events_user_settings.update({f'{k_pair[1]}':
                                                                   str(user_settings['event_model'][f'event_{evt}'][
                                                                           f'outcome_{out}'][f'scenario_{scn}'][
                                                                           f'{k_pair[0]}'])})
                    for k_pair in [('period_range', f'-EVENT{evt}_OUTCOME{out}_SCN{scn}_RANGE-'),
                                   ('outcome_weight', f'-EVENT{evt}_OUTCOME{out}_SCN{scn}_WEIGHT-')]:
                        if k_pair[0] in user_settings['event_model'][f'event_{evt}'][f'outcome_{out}'][
                            f'scenario_{scn}'].keys():
                            model_events_user_settings.update(
                                {f'{k_pair[1]}': '\n'.join(map(str, user_settings['event_model'][f'event_{evt}'][
                                    f'outcome_{out}'][f'scenario_{scn}'][
                                    f'{k_pair[0]}']))}
                            )

    # Quantity Scenario Settings
    if 'quantity_scenarios' in user_settings.keys():
        # Get a list of the scenarios in the quantity scenarios dictionary of the user_settings_file
        qs_scenarios_list_user_settings = [k for k in list(user_settings['quantity_scenarios'].keys()) if
                                           'scenario_' in k]
        qs_scenarios_list_user_settings_ints = list(
            map(int, [k.replace('scenario_', '') for k in qs_scenarios_list_user_settings]))
        # Get the number of price lines needed from the highest number of price groups given.
        n_qs_scenarios_user_settings = max(qs_scenarios_list_user_settings_ints)

        for scn in qs_scenarios_list_user_settings_ints:
            # Add missing scenarios if needed
            while scn > n_scenarios:
                n_scenarios += 1
                add_scenario_tab()

            # Add the scenario description as required.
            if 'scenario_description' in user_settings['quantity_scenarios'][f'scenario_{scn}'].keys():
                window[f'-QNT_SCN{scn}_DESC-'].update(
                    user_settings['quantity_scenarios'][f'scenario_{scn}']['scenario_description'])

            # Get list of groups in the scenario.
            qs_groups_list_user_settings = [k for k in
                                            list(user_settings['quantity_scenarios'][f'scenario_{scn}'].keys()) if
                                            'quantity_group_' in k]
            qs_groups_list_user_settings_ints = list(
                map(int, [k.replace('quantity_group_', '') for k in qs_groups_list_user_settings]))
            # Get the number of price groups needed from the highest number price group given
            n_qs_groups_user_settings = max(qs_groups_list_user_settings_ints)

            for grp in qs_groups_list_user_settings_ints:
                # Add missing groups to the layout if needed.
                while grp not in qs_group_dict.keys():
                    qs_group_dict[len(qs_group_dict) + 1] = 0
                    for s in range(0, n_scenarios + 1):
                        add_quantity_group(s, len(qs_group_dict))

                # Add the quantity group data from the user_settings_file to the layout. Note that value type can only be assigned in scenario_0.
                for k_pair in [('ID', f'-SCN{scn}_QG{grp}_ID-'), ('group_value_type', f'-SCN0_QG{grp}_GRP_TYPE-'),
                               ('MC_PDF', f'-SCN{scn}_QG{grp}_MC_TYPE-')]:
                    if k_pair[0] in user_settings['quantity_scenarios'][f'scenario_{scn}'][
                        f'quantity_group_{grp}'].keys():
                        window[f'{k_pair[1]}'].update(
                            user_settings['quantity_scenarios'][f'scenario_{scn}'][f'quantity_group_{grp}'][
                                f'{k_pair[0]}'])
                if 'MC_PDF' in user_settings['quantity_scenarios'][f'scenario_{scn}'][
                    f'quantity_group_{grp}'].keys():
                    for k_pair in [('PDF_min', f'-SCN{scn}_QG{grp}_PDFMIN-'),
                                   ('PDF_max', f'-SCN{scn}_QG{grp}_PDFMAX-'),
                                   ('PDF_mean', f'-SCN{scn}_QG{grp}_PDFMEA-'),
                                   ('PDF_stdev', f'-SCN{scn}_QG{grp}_PDFSIG-'),
                                   ('PDF_mode', f'-SCN{scn}_QG{grp}_PDFMOD-'),
                                   ('PDF_median', f'-SCN{scn}_QG{grp}_PDFMED-'),
                                   ('PDF_shape', f'-SCN{scn}_QG{grp}_PDFSHA-'),
                                   ('PDF_scale', f'-SCN{scn}_QG{grp}_PDFSCA-'),
                                   ('PDF_lambda', f'-SCN{scn}_QG{grp}_PDFRAT-'),
                                   ('PDF_rand_each_t', f'-SCN{scn}_QG{grp}_PDF_REACHT-'),
                                   ('PDF_Xs', f'-SCN{scn}_QG{grp}_PDFX-'), ('PDF_Ys', f'-SCN{scn}_QG{grp}_PDFY-')]:
                        if k_pair[0] in user_settings['quantity_scenarios'][f'scenario_{scn}'][
                            f'quantity_group_{grp}'].keys():
                            MC_holding_dict.update({f'{k_pair[1]}':
                                                        user_settings['quantity_scenarios'][f'scenario_{scn}'][
                                                            f'quantity_group_{grp}'][f'{k_pair[0]}']})
                            # MC_holding_dict = get_pdf_settings(f'-SCN{scn}_QG{grp}_MC_TYPE-', True)
                    if user_settings['quantity_scenarios'][f'scenario_{scn}'][
                            f'quantity_group_{grp}']['MC_PDF'] != 'None':
                        MC_update_display.append(f'-SCN{scn}_QG{grp}_MC_TYPE-')

                # Get a list of price lines in the price groups given in the user_settings_file
                qs_lines_list_user_settings = [k for k in list(
                    user_settings['quantity_scenarios'][f'scenario_{scn}'][f'quantity_group_{grp}'].keys()) if
                                               'quantity_line_' in k]
                qs_lines_list_user_settings_ints = list(
                    map(int, [k.replace('quantity_line_', '') for k in qs_lines_list_user_settings]))
                # Get the number of price lines needed from the highest number of price groups given.
                n_qs_lines_user_settings = max(qs_lines_list_user_settings_ints)

                while n_qs_lines_user_settings > qs_group_dict[grp]:
                    qs_group_dict[grp] += 1
                    for s in range(0, n_scenarios + 1):
                        add_quantity_line(s, grp)

                # Add quantity line data from the user_settings_file to the layout.
                for lin in qs_lines_list_user_settings_ints:
                    for k_pair in [('ID', f'-SCN{scn}_QG{grp}_LIN{lin}_ID-'),
                                   ('value', f'-SCN{scn}_QG{grp}_LIN{lin}_PRICE-'),
                                   ('stakeholder_group', f'-SCN{scn}_QG{grp}_LIN{lin}_STKE-'),
                                   ('geographic_zone', f'-SCN{scn}_QG{grp}_LIN{lin}_GEOZN-'),
                                   ('outcome_dependency', f'-SCN{scn}_QG{grp}_LIN{lin}_DEPENDS-'),
                                   ('comments', f'-SCN{scn}_QG{grp}_LIN{lin}_CM-'),
                                   ('MC_PDF', f'-SCN{scn}_QG{grp}_LIN{lin}_MC_TYPE-')]:
                        if k_pair[0] in \
                                user_settings['quantity_scenarios'][f'scenario_{scn}'][f'quantity_group_{grp}'][
                                    f'quantity_line_{lin}'].keys():
                            window[f'{k_pair[1]}'].update(
                                user_settings['quantity_scenarios'][f'scenario_{scn}'][f'quantity_group_{grp}'][
                                    f'quantity_line_{lin}'][f'{k_pair[0]}'])
                    for k_pair in [('period_range', f'-SCN{scn}_QG{grp}_LIN{lin}_PRANGE-'),
                                   ('quantity', f'-SCN{scn}_QG{grp}_LIN{lin}_PQUANT-')]:
                        if k_pair[0] in \
                                user_settings['quantity_scenarios'][f'scenario_{scn}'][f'quantity_group_{grp}'][
                                    f'quantity_line_{lin}'].keys():
                            window[f'{k_pair[1]}'].update('\n'.join(
                                user_settings['quantity_scenarios'][f'scenario_{scn}'][f'quantity_group_{grp}'][
                                    f'quantity_line_{lin}'][f'{k_pair[0]}']))
                    if 'MC_PDF' in user_settings['quantity_scenarios'][f'scenario_{scn}'][f'quantity_group_{grp}'][
                        f'quantity_line_{lin}'].keys():
                        for k_pair in [('PDF_min', f'-SCN{scn}_QG{grp}_LIN{lin}_PDFMIN-'),
                                       ('PDF_max', f'-SCN{scn}_QG{grp}_LIN{lin}_PDFMAX-'),
                                       ('PDF_mean', f'-SCN{scn}_QG{grp}_LIN{lin}_PDFMEA-'),
                                       ('PDF_stdev', f'-SCN{scn}_QG{grp}_LIN{lin}_PDFSIG-'),
                                       ('PDF_mode', f'-SCN{scn}_QG{grp}_LIN{lin}_PDFMOD-'),
                                       ('PDF_median', f'-SCN{scn}_QG{grp}_LIN{lin}_PDFMED-'),
                                       ('PDF_shape', f'-SCN{scn}_QG{grp}_LIN{lin}_PDFSHA-'),
                                       ('PDF_scale', f'-SCN{scn}_QG{grp}_LIN{lin}_PDFSCA-'),
                                       ('PDF_lambda', f'-SCN{scn}_QG{grp}_LIN{lin}_PDFRAT-'),
                                       ('PDF_rand_each_t', f'-SCN{scn}_QG{grp}_LIN{lin}_PDF_REACHT-'),
                                       ('PDF_Xs', f'-SCN{scn}_QG{grp}_LIN{lin}_PDFX-'),
                                       ('PDF_Ys', f'-SCN{scn}_QG{grp}_LIN{lin}_PDFY-')]:
                            if k_pair[0] in \
                                    user_settings['quantity_scenarios'][f'scenario_{scn}'][f'quantity_group_{grp}'][
                                        f'quantity_line_{lin}'].keys():
                                MC_holding_dict.update({f'{k_pair[1]}':
                                                            user_settings['quantity_scenarios'][f'scenario_{scn}'][
                                                                f'quantity_group_{grp}'][f'quantity_line_{lin}'][
                                                                f'{k_pair[0]}']})
                        if user_settings['quantity_scenarios'][f'scenario_{scn}'][f'quantity_group_{grp}'][
                                        f'quantity_line_{lin}']['MC_PDF'] != 'None':
                            MC_update_display.append(f'-SCN{scn}_QG{grp}_LIN{lin}_MC_TYPE-')

    if "monte_carlo_settings" in user_settings.keys():
        for k_pair in [('seed', '-RES_SET_SEED-'),
                       ('n_simulations', '-RES_NUM_SIMS-')]:
            if k_pair[0] in user_settings['monte_carlo_settings']:
                window[f'{k_pair[1]}'].update(user_settings['monte_carlo_settings'][f'{k_pair[0]}'])

    # Change windows for size and push changes to display
    set_size(window['-PG_COLS-'], (None, 700))
    set_size(window['-QS_COLS-'], (None, 950))
    window.refresh()
    window['-PG_COLS-'].contents_changed()
    window['-QS_COLS-'].contents_changed()
    window['Trigger Refresh'].click()


# Load Settings to a dictionary called settings
def load_settings_json(user_settings_file):
    global user_settings

    try:
        with open(user_settings_file, 'r') as f:
            user_settings = jsonload(f)
    except Exception as e:
        sg.popup_quick_message(f'exception {e}', 'No settings file found', keep_on_top=True,
                               background_color='red', text_color='white')

    apply_user_settings()


def xlsx_mcpdf_read(source_table, target, row, col):
    global user_settings

    pdf_regex = []
    if target['MC_PDF'] != 'None':
        pdf_regex = re.findall(r'\d+[.]?\d*|\w+', source_table.loc[row, col])
    if len(pdf_regex) > 0:
        if pdf_regex[-1] == 'true':
            pdf_regex[-1] = True
        elif pdf_regex[-1] == 'false':
            pdf_regex[-1] = False

    if target['MC_PDF'] in ['Uniform', 'Bounded normal-like']:
        target['PDF_min'], target['PDF_max'], target['PDF_rand_each_t'] = pdf_regex
    elif target['MC_PDF'] in ['Triangular', 'PERT']:
        target['PDF_min'], target['PDF_mode'], target['PDF_max'], target['PDF_rand_each_t'] = pdf_regex
    elif target['MC_PDF'] in ['Folded normal', 'Truncated normal']:
        target['PDF_mean'], target['PDF_sigma'], target['PDF_rand_each_t'] = pdf_regex
    elif target['MC_PDF'] == 'Exponential':
        target['PDF_lambda'], target['PDF_rand_each_t'] = pdf_regex
    elif target['MC_PDF'] == 'Log-logistic':
        target['PDF_median'], target['PDF_shape'], target['PDF_rand_each_t'] = pdf_regex
    elif target['MC_PDF'] == 'Log-normal':
        target['PDF_median'], target['PDF_sigma'], target['PDF_rand_each_t'] = pdf_regex


def load_xlsx_tables(workbook, sheet):
    table_dict = {}

    worksheet = workbook[sheet]

    for entry, data_boundary in worksheet.tables.items():
        # parse the data within the ref boundary
        data = worksheet[data_boundary]
        # extract the data
        # the inner list comprehension gets the values for each cell in the table
        content = [[cell.value for cell in ent]
                   for ent in data
                   ]

        header = content[0]

        # the contents ... excluding the header
        rest = content[1:]

        # create dataframe with the column names
        # and pair table name with dataframe
        df = pd.DataFrame(rest, columns=header)
        table_dict[entry] = df

    return table_dict


def load_settings_xlsx(user_settings_file):
    global user_settings
    wb = load_workbook(user_settings_file, data_only=True)
    '''
    Note that the xlsx workbooks and tables contain data validation formatting which cannot be translated with openpyxl,
    the user will see the following warnings:
    UserWarning: Data Validation extension is not supported and will be removed warn(msg)
    These warnings can be safely ignored. 
    '''

    mc_settings_xlsx = load_xlsx_tables(wb, 'Monte Carlo Settings')

    user_settings['monte_carlo_settings'] = {}
    user_settings['monte_carlo_settings']['seed'] = mc_settings_xlsx['MC_settings'].loc[0, 'Input']
    user_settings['monte_carlo_settings']['n_simulations'] = mc_settings_xlsx['MC_settings'].loc[1, 'Input']

    dr_settings_xlsx = load_xlsx_tables(wb, 'Discount Rate')

    user_settings['discount_rate_settings'] = {}
    user_settings['discount_rate_settings']['discounting_method'] = dr_settings_xlsx['DR_settings'].loc[0, 'Input']
    if user_settings['discount_rate_settings']['discounting_method'] == 'Constant':
        user_settings['discount_rate_settings']['discount_rate'] = dr_settings_xlsx['DR_settings'].loc[1, 'Input']
    elif user_settings['discount_rate_settings']['discounting_method'] == 'Gamma time declining':
        user_settings['discount_rate_settings']['discount_rate'] = dr_settings_xlsx['DR_settings'].loc[1, 'Input']
        user_settings['discount_rate_settings']['discount_sigma'] = dr_settings_xlsx['DR_settings'].loc[2, 'Input']
    elif user_settings['discount_rate_settings']['discounting_method'] == 'Stepped':
        user_settings['discount_rate_settings']['dr_step_range'] = dr_settings_xlsx['DR_steps'].dr_step_range.tolist()
        user_settings['discount_rate_settings']['dr_step_rates'] = dr_settings_xlsx['DR_steps'].dr_step_rates.tolist()
        user_settings['discount_rate_settings']['discount_step_thereafter'] = dr_settings_xlsx['DR_settings'].loc[
            3, 'Input']

    user_settings['discount_rate_settings']['MC_PDF'] = dr_settings_xlsx['DR_settings'].loc[4, 'Input']

    xlsx_mcpdf_read(dr_settings_xlsx['DR_settings'],
                    user_settings['discount_rate_settings'],
                    5,
                    'Input')

    user_settings['distribution_analysis_settings'] = {}
    da_settings_xlsx = load_xlsx_tables(wb, 'Distribution Analysis')

    if da_settings_xlsx['DA_settings'].loc[0, 'Input'] == 'Simple weight matrix':
        user_settings['distribution_analysis_settings']['simple_weight_matrix'] = \
            [da_settings_xlsx['DA_matrix'].columns.values.tolist()] + da_settings_xlsx['DA_matrix'].values.tolist()
    elif da_settings_xlsx['DA_settings'].loc[0, 'Input'] == 'Income weight matrix':
        user_settings['distribution_analysis_settings']['subgroup_average_income'] = \
            [da_settings_xlsx['DA_matrix'].columns.values.tolist()] + da_settings_xlsx['DA_matrix'].values.tolist()
        user_settings['distribution_analysis_settings']['income_weighting_parameter'] = \
            da_settings_xlsx['DA_settings'].loc[1, 'Input']
        user_settings['distribution_analysis_settings']['population_average_income'] = \
            da_settings_xlsx['DA_settings'].loc[2, 'Input']

    user_settings['parameters'] = {}
    par_settings_xlsx = load_xlsx_tables(wb, 'Parameters')

    for par in range(1, par_settings_xlsx['Parameter_settings'].index.size +1):
        user_settings['parameters'][f'parameter_{par}'] = {}
        user_settings['parameters'][f'parameter_{par}']['ID'] = par_settings_xlsx['Parameter_settings'].loc[par-1, 'parameter_id']
        user_settings['parameters'][f'parameter_{par}']['period_range'] = ast.literal_eval(par_settings_xlsx['Parameter_settings'].loc[par-1, 'period_range'])
        user_settings['parameters'][f'parameter_{par}']['value'] = ast.literal_eval(par_settings_xlsx['Parameter_settings'].loc[par-1, 'value'])
        user_settings['parameters'][f'parameter_{par}']['MC_PDF'] = par_settings_xlsx['Parameter_settings'].loc[
            par - 1, 'MC_PDF']
        user_settings['parameters'][f'parameter_{par}']['comments'] = par_settings_xlsx['Parameter_settings'].loc[par-1, 'comments']
        xlsx_mcpdf_read(par_settings_xlsx['Parameter_settings'],
                        user_settings['parameters'][f'parameter_{par}'],
                        par - 1,
                        'PDF_parameters')

    user_settings['reference_prices'] = {}
    pr_settings_xlsx = load_xlsx_tables(wb, 'Reference Prices')

    user_settings['reference_prices']['country'] = pr_settings_xlsx['PR_settings'].loc[0, 'Input']
    user_settings['reference_prices']['year'] = pr_settings_xlsx['PR_settings'].loc[1, 'Input']
    user_settings['reference_prices']['currency_conversion_range'] = pr_settings_xlsx['PR_settings'].loc[2, 'Input']
    user_settings['reference_prices']['currency_conversion_measure'] = pr_settings_xlsx['PR_settings'].loc[3, 'Input']

    for grp in range(1, pr_settings_xlsx['PR_group'].index.size + 1):
        user_settings['reference_prices'][f'price_group_{grp}'] = {}
        user_settings['reference_prices'][f'price_group_{grp}']['ID'] = pr_settings_xlsx['PR_group'].loc[
            grp - 1, 'group_id']
        user_settings['reference_prices'][f'price_group_{grp}']['MC_PDF'] = pr_settings_xlsx['PR_group'].loc[
            grp - 1, 'MC_PDF']
        xlsx_mcpdf_read(pr_settings_xlsx['PR_group'],
                        user_settings['reference_prices'][f'price_group_{grp}'],
                        grp - 1,
                        'PDF_parameters')
        pr_lines = pr_settings_xlsx['PR_line'].loc[(pr_settings_xlsx['PR_line']['group_under'] ==
                                                    user_settings['reference_prices'][f'price_group_{grp}'][
                                                        'ID'])].reset_index()
        for lin in range(1, pr_lines.index.size + 1):
            user_settings['reference_prices'][f'price_group_{grp}'][f'price_line_{lin}'] = {}
            user_settings['reference_prices'][f'price_group_{grp}'][f'price_line_{lin}']['ID'] = pr_lines.loc[
                lin - 1, 'line_id']
            user_settings['reference_prices'][f'price_group_{grp}'][f'price_line_{lin}']['units'] = pr_lines.loc[
                lin - 1, 'units']
            user_settings['reference_prices'][f'price_group_{grp}'][f'price_line_{lin}']['nominal_value'] = \
                pr_lines.loc[
                    lin - 1, 'nominal_value']
            user_settings['reference_prices'][f'price_group_{grp}'][f'price_line_{lin}']['currency'] = pr_lines.loc[
                lin - 1, 'currency']
            user_settings['reference_prices'][f'price_group_{grp}'][f'price_line_{lin}']['value_year'] = pr_lines.loc[
                lin - 1, 'value_year']
            user_settings['reference_prices'][f'price_group_{grp}'][f'price_line_{lin}']['adjustment_factor'] = \
                pr_lines.loc[
                    lin - 1, 'adjustment_factor']
            user_settings['reference_prices'][f'price_group_{grp}'][f'price_line_{lin}']['MC_PDF'] = pr_lines.loc[
                lin - 1, 'MC_PDF']
            xlsx_mcpdf_read(pr_lines,
                            user_settings['reference_prices'][f'price_group_{grp}'][f'price_line_{lin}'],
                            lin - 1,
                            'PDF_parameters')
            user_settings['reference_prices'][f'price_group_{grp}'][f'price_line_{lin}']['comments'] = pr_lines.loc[
                lin - 1, 'comments']

    user_settings['event_model'] = {}
    em_settings_xlsx = load_xlsx_tables(wb, 'Event Model')

    for event in range(1, em_settings_xlsx['Event_settings'].index.size + 1):
        user_settings['event_model'][f'event_{event}'] = {}
        user_settings['event_model'][f'event_{event}']['ID'] = em_settings_xlsx['Event_settings'].loc[
            event - 1, 'event_id']
        if em_settings_xlsx['Event_settings'].loc[event - 1, 'event_depends'] is not None:
            user_settings['event_model'][f'event_{event}']['event_depends'] = [em_settings_xlsx['Event_settings'].loc[
                                                                                   event - 1, 'event_depends']]
        else:
            user_settings['event_model'][f'event_{event}']['event_depends'] = []

        out_table = em_settings_xlsx['Outcome_settings0'].loc[(
                em_settings_xlsx['Outcome_settings0']['event_under'] ==
                user_settings['event_model'][f'event_{event}']['ID'])].reset_index()

        for out in range(1, out_table.index.size + 1):
            user_settings['event_model'][f'event_{event}'][f'outcome_{out}'] = {}
            user_settings['event_model'][f'event_{event}'][f'outcome_{out}']['ID'] = \
                out_table.loc[out - 1, 'outcome_id']
            out_scn = 0

            while f'Outcome_settings{out_scn}' in em_settings_xlsx:
                out_table = em_settings_xlsx[f'Outcome_settings{out_scn}'].loc[(
                        em_settings_xlsx[f'Outcome_settings{out_scn}']['event_under'] ==
                        user_settings['event_model'][f'event_{event}']['ID'])].reset_index()

                user_settings['event_model'][f'event_{event}'][f'outcome_{out}'][f'scenario_{out_scn}'] = {}
                user_settings['event_model'][f'event_{event}'][f'outcome_{out}'][f'scenario_{out_scn}'][
                    'period_range'] = \
                    ast.literal_eval(out_table.loc[out - 1, 'period_range'])
                user_settings['event_model'][f'event_{event}'][f'outcome_{out}'][f'scenario_{out_scn}'][
                    'outcome_weight'] = ast.literal_eval(out_table.loc[out - 1, 'outcome_weight'])
                user_settings['event_model'][f'event_{event}'][f'outcome_{out}'][f'scenario_{out_scn}'][
                    'periods_without_repeat'] = out_table.loc[out - 1, 'periods_without_repeat']
                user_settings['event_model'][f'event_{event}'][f'outcome_{out}'][f'scenario_{out_scn}'][
                    'max_repeats'] = out_table.loc[out - 1, 'max_repeats']
                out_scn += 1

    user_settings['quantity_scenarios'] = {}
    qs_settings_xlsx = load_xlsx_tables(wb, 'Quantity Scenarios')

    qnt_scn = 0
    while f'QS_settings{qnt_scn}' in qs_settings_xlsx:
        user_settings['quantity_scenarios'][f'scenario_{qnt_scn}'] = {}
        user_settings['quantity_scenarios'][f'scenario_{qnt_scn}']['scenario_description'] = \
            qs_settings_xlsx[f'QS_settings{qnt_scn}'].loc[1, 'Input']

        for qnt_grp in range(1, qs_settings_xlsx[f'QS_group{qnt_scn}'].index.size + 1):
            user_settings['quantity_scenarios'][f'scenario_{qnt_scn}'][f'quantity_group_{qnt_grp}'] = {}
            user_settings['quantity_scenarios'][f'scenario_{qnt_scn}'][f'quantity_group_{qnt_grp}']['ID'] = \
                qs_settings_xlsx[f'QS_group{qnt_scn}'].loc[qnt_grp - 1, 'group_id']
            user_settings['quantity_scenarios'][f'scenario_{qnt_scn}'][f'quantity_group_{qnt_grp}'][
                'group_value_type'] = \
                qs_settings_xlsx[f'QS_group{qnt_scn}'].loc[qnt_grp - 1, 'group_type']
            user_settings['quantity_scenarios'][f'scenario_{qnt_scn}'][f'quantity_group_{qnt_grp}'][
                'MC_PDF'] = qs_settings_xlsx[f'QS_group{qnt_scn}'].loc[qnt_grp - 1, 'MC_PDF']
            xlsx_mcpdf_read(qs_settings_xlsx[f'QS_group{qnt_scn}'],
                            user_settings['quantity_scenarios'][f'scenario_{qnt_scn}'][f'quantity_group_{qnt_grp}'],
                            qnt_grp - 1,
                            'PDF_parameters')
            qnt_lin_table = qs_settings_xlsx[f'QS_line{qnt_scn}'].loc[(
                    qs_settings_xlsx[f'QS_line{qnt_scn}']['group_under'] ==
                    user_settings['quantity_scenarios'][f'scenario_{qnt_scn}'][f'quantity_group_{qnt_grp}'][
                        'ID'])].reset_index()
            for qnt_lin in range(1, qnt_lin_table.index.size + 1):
                user_settings['quantity_scenarios'][f'scenario_{qnt_scn}'][f'quantity_group_{qnt_grp}'][
                    f'quantity_line_{qnt_lin}'] = {}
                user_settings['quantity_scenarios'][f'scenario_{qnt_scn}'][f'quantity_group_{qnt_grp}'][
                    f'quantity_line_{qnt_lin}']['ID'] = qnt_lin_table.loc[qnt_lin - 1, 'line_id']
                user_settings['quantity_scenarios'][f'scenario_{qnt_scn}'][f'quantity_group_{qnt_grp}'][
                    f'quantity_line_{qnt_lin}']['value'] = qnt_lin_table.loc[qnt_lin - 1, 'value']
                user_settings['quantity_scenarios'][f'scenario_{qnt_scn}'][f'quantity_group_{qnt_grp}'][
                    f'quantity_line_{qnt_lin}']['geographic_zone'] = qnt_lin_table.loc[qnt_lin - 1, 'geographic_zone']
                user_settings['quantity_scenarios'][f'scenario_{qnt_scn}'][f'quantity_group_{qnt_grp}'][
                    f'quantity_line_{qnt_lin}']['stakeholder_group'] = qnt_lin_table.loc[
                    qnt_lin - 1, 'stakeholder_group']
                user_settings['quantity_scenarios'][f'scenario_{qnt_scn}'][f'quantity_group_{qnt_grp}'][
                    f'quantity_line_{qnt_lin}']['period_range'] = ast.literal_eval(
                    qnt_lin_table.loc[qnt_lin - 1, 'period_range'])
                user_settings['quantity_scenarios'][f'scenario_{qnt_scn}'][f'quantity_group_{qnt_grp}'][
                    f'quantity_line_{qnt_lin}']['quantity'] = ast.literal_eval(
                    qnt_lin_table.loc[qnt_lin - 1, 'quantity'])
                user_settings['quantity_scenarios'][f'scenario_{qnt_scn}'][f'quantity_group_{qnt_grp}'][
                    f'quantity_line_{qnt_lin}']['MC_PDF'] = qnt_lin_table.loc[qnt_lin - 1, 'MC_PDF']
                xlsx_mcpdf_read(qnt_lin_table,
                                user_settings['quantity_scenarios'][f'scenario_{qnt_scn}'][f'quantity_group_{qnt_grp}'][
                                    f'quantity_line_{qnt_lin}'],
                                qnt_lin - 1,
                                'PDF_parameters')
                user_settings['quantity_scenarios'][f'scenario_{qnt_scn}'][f'quantity_group_{qnt_grp}'][
                    f'quantity_line_{qnt_lin}']['outcome_dependency'] = qnt_lin_table.loc[
                    qnt_lin - 1, 'outcome_dependency']

        qnt_scn += 1

    apply_user_settings()


def extract_pdf_settings(setting_dict, prefix):
    pdf_pairs = [('PDF_min', f'{prefix}_PDFMIN-'), ('PDF_max', f'{prefix}_PDFMAX-'), ('PDF_mean', f'{prefix}_PDFMEA-'),
                 ('PDF_stdev', f'{prefix}_PDFSIG-'), ('PDF_mode', f'{prefix}_PDFMOD-'),
                 ('PDF_median', f'{prefix}_PDFMED-'),
                 ('PDF_shape', f'{prefix}_PDFSHA-'), ('PDF_scale', f'{prefix}_PDFSCA-'),
                 ('PDF_lambda', f'{prefix}_PDFRAT-'), ('PDF_rand_each_t', f'{prefix}_PDF_REACHT-'),
                 ('PDF_Xs', f'{prefix}_PDFX-'), ('PDF_Ys', f'{prefix}_PDFY-')]

    for pair in pdf_pairs:
        if pair[1] in MC_holding_dict:
            setting_dict[pair[0]] = MC_holding_dict[pair[1]]


def json_settings():
    monte_carlo_dict = {}
    discount_rate_dict = {}
    distribution_dict = {}
    function_dict_j = {}
    parameter_dict_j = {}
    reference_prices_dict = {}
    event_model_dict = {}
    quantity_scenarios_dict = {}
    # parameter_dict is used elsewhere and is a very important variable, parameter_dict_j is used here to maintain hygiene.

    monte_carlo_dict['seed'] = int(values['-RES_SET_SEED-'])
    monte_carlo_dict['n_simulations'] = int(values['-RES_NUM_SIMS-'])

    if values['-DR_TYPE-'] == 'Constant':
        discount_rate_dict['discounting_method'] = 'Constant'
        discount_rate_dict['discount_rate'] = float(values['-DR_MAIN-'])
    elif values['-DR_TYPE-'] == 'Stepped':
        discount_rate_dict['dr_step_range'] = values['-DR_STEP_RANGE-'].split('\n')
        discount_rate_dict['dr_step_rates'] = map(float, values['-DR_STEP_RATE-'].split('\n'))
        discount_rate_dict['dr_step_thereafter'] = float(values['-DR_STEP_BASE-'])
    elif values['-DR_TYPE-'] == 'Gamma time declining':
        discount_rate_dict['discount_rate'] = float(values['-DR_MAIN-'])
    discount_rate_dict['MC_PDF'] = values['-DR_MC_TYPE-']
    extract_pdf_settings('-DR', discount_rate_dict)

    if len(DA_table) > 1:
        if simpe_DA is True:
            distribution_dict['simple_weight_matrix'] = DA_table
            distribution_dict['simple_weight_matrix'][0][0] = ''
        else:
            distribution_dict['population_average_income'] = pop_mean_income
            distribution_dict['income_weighting_parameter'] = income_weight_parameter
            distribution_dict['subgroup_average_income'] = income_table
            distribution_dict['subgroup_average_income'][0][0] = ''

    if n_functions >0:
        for fnc in range(1, n_functions+1):
            function_dict_j[f'function_{fnc}'] = {}
            function_dict_j[f'function_{fnc}']['ID'] = values[f'-FNC{fnc}_ID-']
            function_dict_j[f'function_{fnc}']['x_values'] = values[f'-FNC{fnc}_XVALS-']
            function_dict_j[f'function_{fnc}']['y_values'] = values[f'-FNC{fnc}_YVALS-']
            function_dict_j[f'function_{fnc}']['interpolation_method'] = values[f'-FNC{fnc}_INTERP-']
            function_dict_j[f'function_{fnc}']['smoothing_factor'] = values[f'-FNC{fnc}_SMOOTHSET-']
            function_dict_j[f'function_{fnc}']['comments'] = values[f'-FNC{fnc}_CM-']

    if n_params > 0:
        for par in range(1, n_params+1):
            parameter_dict_j[f'parameter_{par}'] = {}
            parameter_dict_j[f'parameter_{par}']['ID'] = values[f'-PAR{par}_ID-']
            parameter_dict_j[f'parameter_{par}']['period_range'] = values[f'-PAR{par}_PRANGE-']
            parameter_dict_j[f'parameter_{par}']['values'] = values[f'-PAR{par}_PVALUE-']
            parameter_dict_j[f'parameter_{par}']['comments'] = values[f'-PAR{par}_CM-']
            extract_pdf_settings(parameter_dict_j[f'parameter_{par}'], f'-PAR{par}')

    reference_prices_dict['country'] = values['-PR_COUNTRY-']
    reference_prices_dict['year'] = values['-PR_YEAR-']
    reference_prices_dict['currency_conversion_range'] = values['-PR_CUR_RANGE-']
    reference_prices_dict['currency_conversion_measure'] = values['-PR_CUR_POINT-']

    for grp in range(1, len(pr_group_dict) + 1):
        reference_prices_dict[f'price_group_{grp}'] = {}
        reference_prices_dict[f'price_group_{grp}']['ID'] = values[f'-PG{grp}_ID-']
        reference_prices_dict[f'price_group_{grp}']['MC_PDF'] = values[f'-PG{grp}_MC_TYPE-']
        extract_pdf_settings(reference_prices_dict[f'price_group_{grp}'], f'-PG{grp}')

        for lin in range(1, pr_group_dict[grp] + 1):
            reference_prices_dict[f'price_group_{grp}'][f'price_line_{lin}'] = {}
            reference_prices_dict[f'price_group_{grp}'][f'price_line_{lin}']['ID'] = values[f'-PG{grp}_LIN{lin}_ID-']
            reference_prices_dict[f'price_group_{grp}'][f'price_line_{lin}']['units'] = values[f'-PG{grp}_LIN{lin}_UN-']
            try:
                reference_prices_dict[f'price_group_{grp}'][f'price_line_{lin}']['nominal_value'] = float(
                    values[f'-PG{grp}_LIN{lin}_NV-'])
            except ValueError:
                reference_prices_dict[f'price_group_{grp}'][f'price_line_{lin}']['nominal_value'] = values[f'-PG{grp}_LIN{lin}_NV-']
            reference_prices_dict[f'price_group_{grp}'][f'price_line_{lin}']['currency'] = values[
                f'-PG{grp}_LIN{lin}_CUR-']
            reference_prices_dict[f'price_group_{grp}'][f'price_line_{lin}']['value_year'] = int(
                values[f'-PG{grp}_LIN{lin}_CURYR-'])
            reference_prices_dict[f'price_group_{grp}'][f'price_line_{lin}']['adjustment_factor'] = float(
                values[f'-PG{grp}_LIN{lin}_AF-'])
            reference_prices_dict[f'price_group_{grp}'][f'price_line_{lin}']['MC_PDF'] = values[
                f'-PG{grp}_LIN{lin}_MC_TYPE-']
            extract_pdf_settings(reference_prices_dict[f'price_group_{grp}'][f'price_line_{lin}'], f'-PG{grp}_LIN{lin}')
            reference_prices_dict[f'price_group_{grp}'][f'price_line_{lin}']['comments'] = values[
                f'-PG{grp}_LIN{lin}_CM-']

    if len(model_events_dict) > 0:
        for event in range(1, len(model_events_dict) + 1):
            event_model_dict[f'event_{event}'] = {}
            event_model_dict[f'event_{event}']['ID'] = model_events_user_settings[f'-EVENT{event}_ID-']
            event_model_dict[f'event_{event}']['event_depends'] = model_events_user_settings[f'-EVENT{event}_DEPENDS-']
            for outcome in range(1, model_events_dict[event]):
                event_model_dict[f'event_{event}'][f'outcome_{outcome}'] = {}
                event_model_dict[f'event_{event}'][f'outcome_{outcome}']['ID'] = model_events_user_settings[
                    f'-EVENT{event}_OUTCOME{outcome}_ID-']
                for scn in range(0, n_scenarios + 1):
                    event_model_dict[f'event_{event}'][f'outcome_{outcome}'][f'scenario_{scn}'] = {}
                    event_model_dict[f'event_{event}'][f'outcome_{outcome}'][f'scenario_{scn}']['period_range'] = \
                        model_events_user_settings[f'-EVENT{event}_OUTCOME{outcome}_SCN{scn}_RANGE-'].split('\n')
                    event_model_dict[f'event_{event}'][f'outcome_{outcome}'][f'scenario_{scn}']['outcome_weight'] = \
                        model_events_user_settings[f'-EVENT{event}_OUTCOME{outcome}_SCN{scn}_WEIGHT-'].split('\n')
                    event_model_dict[f'event_{event}'][f'outcome_{outcome}'][f'scenario_{scn}'][
                        'periods_without_repeat'] = \
                        model_events_user_settings[f'-EVENT{event}_OUTCOME{outcome}_SCN{scn}_NOREPS-']
                    event_model_dict[f'event_{event}'][f'outcome_{outcome}'][f'scenario_{scn}']['max_repeats'] = \
                        model_events_user_settings[f'-EVENT{event}_OUTCOME{outcome}_SCN{scn}_MAXREPS-']

    for scn in range(0, n_scenarios + 1):
        quantity_scenarios_dict[f'scenario_{scn}'] = {}
        quantity_scenarios_dict[f'scenario_{scn}']['scenario_description'] = values[f'-QNT_SCN{n_scenarios}_DESC-']
        for grp in range(1, len(qs_group_dict) + 1):
            quantity_scenarios_dict[f'scenario_{scn}'][f'quantity_group_{grp}'] = {}
            quantity_scenarios_dict[f'scenario_{scn}'][f'quantity_group_{grp}']['ID'] = values[f'-SCN{scn}_QG{grp}_ID-']
            quantity_scenarios_dict[f'scenario_{scn}'][f'quantity_group_{grp}']['group_value_type'] = \
                values[f'-SCN{scn}_QG{grp}_GRP_TYPE-']
            quantity_scenarios_dict[f'scenario_{scn}'][f'quantity_group_{grp}']['MC_PDF'] = \
                values[f'-SCN{scn}_QG{grp}_MC_TYPE-']
            extract_pdf_settings(quantity_scenarios_dict[f'scenario_{scn}'][f'quantity_group_{grp}'],
                                 f'-SCN{scn}_QG{grp}')
            for lin in range(1, qs_group_dict[grp] + 1):
                quantity_scenarios_dict[f'scenario_{scn}'][f'quantity_group_{grp}'][f'quantity_line_{lin}'] = {}
                quantity_scenarios_dict[f'scenario_{scn}'][f'quantity_group_{grp}'][f'quantity_line_{lin}']['ID'] = \
                    values[f'-SCN{scn}_QG{grp}_LIN{lin}_ID-']
                quantity_scenarios_dict[f'scenario_{scn}'][f'quantity_group_{grp}'][f'quantity_line_{lin}']['value'] = \
                    values[f'-SCN{scn}_QG{grp}_LIN{lin}_PRICE-']
                quantity_scenarios_dict[f'scenario_{scn}'][f'quantity_group_{grp}'][f'quantity_line_{lin}'][
                    'stakeholder_group'] = \
                    values[f'-SCN{scn}_QG{grp}_LIN{lin}_STKE-']
                quantity_scenarios_dict[f'scenario_{scn}'][f'quantity_group_{grp}'][f'quantity_line_{lin}'][
                    'geographic_zone'] = \
                    values[f'-SCN{scn}_QG{grp}_LIN{lin}_GEOZN-']
                quantity_scenarios_dict[f'scenario_{scn}'][f'quantity_group_{grp}'][f'quantity_line_{lin}'][
                    'period_range'] = \
                    values[f'-SCN{scn}_QG{grp}_LIN{lin}_PRANGE-'].split('\n')
                quantity_scenarios_dict[f'scenario_{scn}'][f'quantity_group_{grp}'][f'quantity_line_{lin}'][
                    'quantity'] = \
                    values[f'-SCN{scn}_QG{grp}_LIN{lin}_PQUANT-'].split('\n')
                quantity_scenarios_dict[f'scenario_{scn}'][f'quantity_group_{grp}'][f'quantity_line_{lin}']['MC_PDF'] = \
                    values[f'-SCN{scn}_QG{grp}_LIN{lin}_MC_TYPE-']
                extract_pdf_settings(quantity_scenarios_dict[f'scenario_{scn}'][f'quantity_group_{grp}'][
                                         f'quantity_line_{lin}'], f'-SCN{scn}_QG{grp}_LIN{lin}')
                quantity_scenarios_dict[f'scenario_{scn}'][f'quantity_group_{grp}'][f'quantity_line_{lin}'][
                    'outcome_dependency'] = \
                    values[f'-SCN{scn}_QG{grp}_LIN{lin}_DEPENDS-']
                quantity_scenarios_dict[f'scenario_{scn}'][f'quantity_group_{grp}'][f'quantity_line_{lin}'][
                    'comments'] = \
                    values[f'-SCN{scn}_QG{grp}_LIN{lin}_CM-']

    settings_dict = {
        'monte_carlo_settings': monte_carlo_dict,
        'discount_rate_settings': discount_rate_dict,
        'distribution_analysis_settings': distribution_dict,
        'function_settings': function_dict_j,
        'parameter_settings': parameter_dict_j,
        'reference_prices': reference_prices_dict,
        'event_model': event_model_dict,
        'quantity_scenarios': quantity_scenarios_dict
    }

    return jsondumps(settings_dict, indent=4)


### DISCOUNT RATE SECTION ---------------------------------------------------------------------------------------------
# Key list is a variable that enables cycling through a group of keys for succinct updating.
key_list = []


# Define a PDF Mean Test function.
def PDF_meantest(PDF_func, key_prefix, values_dictionary):
    PDF_min = float(values_dictionary.get(f'{key_prefix}_PDFMIN-', '0'))
    PDF_max = float(values_dictionary.get(f'{key_prefix}_PDFMAX-', '0'))
    PDF_mode = float(values_dictionary.get(f'{key_prefix}_PDFMOD-', '0'))
    PDF_mean = float(values_dictionary.get(f'{key_prefix}_PDFMEA-', '0'))
    PDF_stdev = float(values_dictionary.get(f'{key_prefix}_PDFSIG-', '0'))
    PDF_lambda = float(values_dictionary.get(f'{key_prefix}_PDFRAT-', '0'))
    PDF_shape = float(values_dictionary.get(f'{key_prefix}_PDFSHA-', '0'))
    PDF_median = float(values_dictionary.get(f'{key_prefix}_PDFMED-', '0'))
    PDF_xs = np.array(list(map(float, values_dictionary.get(f'{key_prefix}_PDFX-', '0').split('\n'))))
    PDF_ys = np.array(list(map(float, values_dictionary.get(f'{key_prefix}_PDFY-', '0').split('\n'))))

    if PDF_func == 'Uniform' or PDF_func == 'Bounded normal-like':
        try:
            return np.around((PDF_min + PDF_max) / 2, 3)
        except ValueError:
            return np.nan
    elif PDF_func == 'Triangular':
        try:
            return np.around((PDF_min + PDF_max + PDF_mode) / 3, 3)
        except ValueError:
            return np.nan
    elif PDF_func == 'PERT':
        try:
            return np.around((PDF_min + 4 * PDF_mode + PDF_max) / 6, 3)
        except ValueError:
            return np.nan
    elif PDF_func == 'Truncated normal':
        try:
            return np.around(truncnorm.stats(a=0, b=np.inf, loc=PDF_mean, scale=PDF_stdev, moments='m'), 3)
        except ValueError:
            return np.nan
    elif PDF_func == 'Folded normal':
        try:
            return np.around(foldnorm.stats(c=(PDF_mean / PDF_stdev), loc=0, scale=PDF_stdev, moments='m'), 3)
        except ValueError:
            return np.nan
    elif PDF_func == 'Exponential':
        try:
            return np.around(expon.stats(scale=(1 / PDF_lambda), moments='m'), 3)
        except ValueError:
            return np.nan
    elif PDF_func == 'Log-logistic':
        try:
            return np.around(fisk.stats(c=PDF_shape, scale=PDF_median, moments='m'), 3)
        except ValueError:
            return np.nan
    elif PDF_func == 'Log-normal':
        try:
            return np.around(lognorm.stats(s=PDF_stdev, scale=np.exp(PDF_mean), moments='m'), 3)
        except ValueError:
            return np.nan
    elif PDF_func == 'Custom Kinked PDF':
        try:
            set_custom_dfs(PDF_xs, PDF_ys)
            cust_kink = custom_kinks_gen(name='custom_kinks', a=np.min(PDF_xs), b=np.max(PDF_xs))
            return np.around(mean_df, 3)
        except ValueError:
            return np.nan
        except IndexError:
            return np.nan
    else:
        return np.nan


### DISTRIBUTION ANALYSIS SECTION -------------------------------------------------------------------------------------

# Define important variables
stakeholder_list = []
geozones_list = []
DA_table = []
income_table = []
pop_mean_income = 1.0
income_weight_parameter = 0.0
simpe_DA = True


# Define a custom popup to obtain weights for distribution analysis
def weight_matrix_simple():
    window.Disable()
    global DA_table
    global simple_DA
    simple_DA = True
    ROWS = len(DA_table)
    COLS = len(DA_table[1])

    updated_DA_table = DA_table

    # Define the first column as only the first entry in each row of DA_table
    swm_layout_C0 = [
        [sg.Text(DA_table[i][0])] for i in range(ROWS)
    ]
    # Define subsequent columns as the corresp value in the first row of DA_table, followed by rows of input cells.
    swm_layout_CN = [[]]
    for j in range(1, COLS):
        col = [[sg.Text(DA_table[0][j])]] + [
            [sg.Input(default_text=DA_table[i][j], key=f'-INP_WEIGHT{i}{j}-', size=(10))] for i in range(1, ROWS)]
        swm_layout_CN[0] += [sg.Column(col)]

    simple_matrix_layout = [
        [sg.Column(swm_layout_C0), sg.Column(swm_layout_CN)],
        [sg.Button('Submit'), sg.Button('Cancel')]
    ]

    swm_window = sg.Window('Basic Weight Matrix', simple_matrix_layout, enable_close_attempted_event=True)

    while True:
        swm_event, swm_values = swm_window.read()
        if swm_event == 'Submit':
            for i in range(1, ROWS):
                for j in range(1, COLS):
                    updated_DA_table[i][j] = float(swm_values[f'-INP_WEIGHT{i}{j}-'])
                    DA_table = updated_DA_table
            break

        if swm_event in (sg.WINDOW_CLOSE_ATTEMPTED_EVENT, 'Cancel'):
            break
    window.Enable()
    swm_window.close()


def weight_matrix_incomes():
    window.Disable()
    global DA_table
    global income_table
    global pop_mean_income
    global income_weight_parameter
    global simple_DA
    simple_DA = False
    ROWS = len(DA_table)
    COLS = len(DA_table[1])

    if len(income_table) < len(DA_table):
        income_table = DA_table

    ciwm_DA_table = DA_table
    new_income_table = income_table
    # Build the income weight matrix table for inputs.
    # Define the first column as only the first entry in each row of income_table
    iwm_layout_C0 = [
        [sg.Text(income_table[i][0])] for i in range(ROWS)
    ]
    # Define subsequent columns as the corresp value in the first row of income_table, followed by rows of input cells.
    iwm_layout_CN = [[]]
    for j in range(1, COLS):
        col = [[sg.Text(income_table[0][j])]] + [
            [sg.Input(default_text=income_table[i][j], key=f'-IND_MEAN_INC{i}{j}-', size=(10), enable_events=True)] for
            i in
            range(1, ROWS)]
        iwm_layout_CN[0] += [sg.Column(col)]

    income_matrix_layout = [
        [sg.Frame(title='Income Weight Parameters',
                  layout=[
                      [sg.Text('Population Average Income'),
                       sg.Input(key='-POP_MEAN_INC-', default_text=pop_mean_income, enable_events=True)],
                      [sg.Text('Income Weighting Parameter'),
                       sg.Input(default_text=income_weight_parameter, key='-INC_WEIGHT_SIG-', enable_events=True)],
                      [sg.Text('Note that an income weighting parameter value of 0 implies no income weighting.\nA '
                               'value of 1.0 implies an income-weighted approach where one person equals one vote.\nA '
                               'value greater than 1.0 implies a concave social welfare function where incremental\n '
                               'improvements in monetary welfare are worth more to people with lower incomes than\n '
                               'higher ones.')]
                  ]
                  )],
        [sg.HorizontalSeparator()],
        [sg.Text('Subgroup Average Income')],
        [sg.Column(iwm_layout_C0), sg.Column(iwm_layout_CN)],
        [sg.HorizontalSeparator()],
        [sg.Text('Computed Weights')],
        [sg.Text(text=pd.DataFrame(ciwm_DA_table[1:len(DA_table)], columns=ciwm_DA_table[0]).to_string(index=False),
                 key='-CIWM_TABLE_DISPLAY-', justification='right', font=('Courier New', 11),
                 relief="groove", border_width=1)],
        [sg.Button('Submit'), sg.Button('Cancel')]
    ]

    iwm_window = sg.Window('Income Weight Matrix', income_matrix_layout, enable_close_attempted_event=True)

    while True:
        iwm_event, iwm_values = iwm_window.read()

        # Check to see if there is an income table already, if there is update the computed weight matrix
        if iwm_event not in ('Submit', 'Cancel', sg.WIN_CLOSED):
            for j in range(1, COLS):
                for i in range(1, ROWS):
                    try:
                        inc_weight = np.around(
                            (float(iwm_values['-POP_MEAN_INC-']) / float(iwm_values[f'-IND_MEAN_INC{i}{j}-'])) ** (
                                float(iwm_values['-INC_WEIGHT_SIG-'])), 3)
                        ciwm_DA_table[i][j] = inc_weight
                        new_income_table[i][j] = iwm_values[f'-IND_MEAN_INC{i}{j}-']
                    except (ZeroDivisionError, ValueError):
                        ciwm_DA_table[i][j] = np.nan
                        new_income_table[i][j] = iwm_values[f'-IND_MEAN_INC{i}{j}-']
            iwm_window['-CIWM_TABLE_DISPLAY-'].update(
                pd.DataFrame(ciwm_DA_table[1:len(DA_table)], columns=ciwm_DA_table[0]).to_string(index=False))

        if iwm_event == 'Submit':
            DA_table = ciwm_DA_table
            pop_mean_income = iwm_values['-POP_MEAN_INC-']
            income_table = new_income_table
            income_weight_parameter = iwm_values['-INC_WEIGHT_SIG-']
            break

        if iwm_event in (sg.WINDOW_CLOSE_ATTEMPTED_EVENT, 'Cancel'):
            break

    window.Enable()
    iwm_window.close()


### FUNCTIONS SECTION -------------------------------------------------------------------------------------------------
n_functions = 0
function_names = []

def add_function():
    window.extend_layout(window['-FUNCTIONS-'], [
        [sg.Frame(f'Function {n_functions}', [
            [sg.Column([
                [sg.Text('Function ID:')],
                [sg.Input(key=f'-FNC{n_functions}_ID-', size=10, enable_events=True)]]),
                sg.VerticalSeparator(),
                sg.Column([
                    [sg.Text('x values:')],
                    [sg.Multiline(key=f'-FNC{n_functions}_XVALS-', size=(10, 4), enable_events=True)]
                ]),
                sg.Column([
                    [sg.Text('y values:')],
                    [sg.Multiline(key=f'-FNC{n_functions}_YVALS-', size=(10, 4), enable_events=True)]
                ]),
                sg.VerticalSeparator(),
                sg.Column([
                    [sg.Text('Interpolation\nmethod:')],
                    [sg.Combo(key=f'-FNC{n_functions}_INTERP-',
                              values=['Nearest', 'Previous', 'Next', 'Linear', 'Cubic spline', 'Akima', 'Smoothed cubic spline'],
                              enable_events=True, default_value='Linear', size=15)],
                    [sg.pin(sg.Text(key=f'-FNC{n_functions}_SMOOTHTXT-', text='Smoothing parameter (0-1):', visible=False))],
                    [sg.pin(sg.Input(key=f'-FNC{n_functions}_SMOOTHSET-', size=10, visible=False, default_text='0.0'))]
                ]),
                sg.Text(text='\U0001F4C8', key=f'-FNC{n_functions}_GRAPH-',
                        enable_events=True, font=(None, 15)),
                sg.VerticalSeparator(),
                sg.Column([
                    [sg.Text('Comments/Metadata:')],
                    [sg.Multiline(key=f'-FNC{n_functions}_CM-', size=(40, 4))]
                ])
            ]
        ])]
    ])


def function_popup(function):
    # get x and y values and plot a graph of the interpolated function.
    window.Disable()

    xs = np.array(list(map(int, re.split(r'\n', values[f'-FNC{function}_XVALS-']))))
    ys = np.array(list(map(int, re.split(r'\n', values[f'-FNC{function}_YVALS-']))))

    match values[f'-FNC{function}_INTERP-']:
        case 'Nearest':
            print('NEAREST')
            interper = interpolate.interp1d(xs, ys, kind='nearest')
        case 'Previous':
            interper = interpolate.interp1d(xs, ys, kind='previous')
        case 'Next':
            interper = interpolate.interp1d(xs, ys, kind='next')
        case 'Linear':
            interper = interpolate.interp1d(xs, ys, kind='linear')
        case 'Cubic spline':
            interper = interpolate.CubicSpline(xs, ys)
        case 'Akima':
            interper = interpolate.Akima1DInterpolator(xs, ys)
        case 'Smoothed cubic spline':
            '''
            The maximum practical value for s is the number of squared residuals when the number of knots equals 2,
            that is, when the number of knots is 2. Since the UnivariateSpline starts at 2 knots and increases them until
            the squared residuals are lower than a given s, setting s to np.inf forces the knots to equal 2, and the maximum 
            allowable s value can be determined. From there, a normalised smoother variable can be used to determined a 
            practically useful s. 
            '''
            smoother = float(values[f'-FNC{function}_SMOOTHSET-'])
            max_sq_resids = interpolate.UnivariateSpline(xs, ys, s=np.inf).get_residual()
            interper = interpolate.UnivariateSpline(xs, ys, s=(smoother*max_sq_resids))

    interp_layout = [
        [sg.Frame('', [[sg.Canvas(key='-INTERP_CANVAS-', expand_x=True, expand_y=True, )]], size=(640, 480))],
        [sg.Button('Close')]
    ]

    interp_window = sg.Window('Interpolation', interp_layout, finalize=True, resizable=True,
                                  font=('Helvetica', 10))

    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot()
    canvas = Canvas(fig, interp_window['-INTERP_CANVAS-'].Widget)

    ax.set_title('Interpolation function')
    ax.set_ylabel("Y values")
    ax.set_xlabel("X values")
    ax.plot(xs, ys, 'ok', ms=5)
    x_line = np.linspace(min(xs), max(xs), 1000)
    ax.plot(x_line, interper(x_line), 'saddlebrown', lw=3)
    canvas.draw()

    while True:
        interp_event, interp_values = interp_window.read()

        if interp_event == sg.WIN_CLOSED or interp_event == 'Close':
            break

    interp_window.close()
    window.Enable()
    window.BringToFront()
    window.Read()


def assign_function(function):
    xs = np.array(list(map(float, re.split(r'\n', values[f'-FNC{function}_XVALS-']))))
    ys = np.array(list(map(float, re.split(r'\n', values[f'-FNC{function}_YVALS-']))))

    match values[f'-FNC{function}_INTERP-']:
        case 'Nearest':
            print('NEAREST')
            interper = interpolate.interp1d(xs, ys, kind='nearest')
        case 'Previous':
            interper = interpolate.interp1d(xs, ys, kind='previous')
        case 'Next':
            interper = interpolate.interp1d(xs, ys, kind='next')
        case 'Linear':
            interper = interpolate.interp1d(xs, ys, kind='linear')
        case 'Cubic spline':
            interper = interpolate.CubicSpline(xs, ys)
        case 'Akima':
            interper = interpolate.Akima1DInterpolator(xs, ys)
        case 'Smoothed cubic spline':
            '''
            The maximum practical value for s is the number of squared residuals when the number of knots equals 2,
            that is, when the number of knots is 2. Since the UnivariateSpline starts at 2 knots and increases them until
            the squared residuals are lower than a given s, setting s to np.inf forces the knots to equal 2, and the maximum 
            allowable s value can be determined. From there, a normalised smoother variable can be used to determined a 
            practically useful s. 
            '''
            smoother = float(values[f'-FNC{function}_SMOOTHSET-'])
            max_sq_resids = interpolate.UnivariateSpline(xs, ys, s=np.inf).get_residual()
            interper = interpolate.UnivariateSpline(xs, ys, s=(smoother*max_sq_resids))

    return interper

### PARAMETERS SECTION ------------------------------------------------------------------------------------------------
n_params = 0


def add_parameter():
    window.extend_layout(window['-PARAMETERS-'], [
        [sg.Frame(f'Parameter {n_params}', [
            [sg.Column([
                [sg.Text('Parameter ID:')],
                [sg.Input(key=f'-PAR{n_params}_ID-', size=10, enable_events=True)]]),
                sg.VerticalSeparator(),
                sg.Column([
                    [sg.Text('Period range:')],
                    [sg.Multiline(key=f'-PAR{n_params}_PRANGE-', size=(10, 4), enable_events=True)]
                ]),
                sg.Column([
                    [sg.Text('Value:')],
                    [sg.Multiline(key=f'-PAR{n_params}_PVALUE-', size=(10, 4), enable_events=True)]
                ]),
                sg.VerticalSeparator(),
                sg.Column([
                    [sg.Text('Probability\nDistribution\nFunction:')],
                    [sg.Combo(key=f'-PAR{n_params}_MC_TYPE-',
                              values=['None', 'Uniform', 'Bounded normal-like', 'Triangular',
                                      'PERT', 'Folded normal', 'Truncated normal',
                                      'Exponential',
                                      'Log-logistic',
                                      'Log-normal', 'Custom Kinked PDF'], enable_events=True,
                              default_value='None',
                              size=15)]
                ]),
                sg.Column([
                    [sg.pin(
                        sg.Text(key=f'-PAR{n_params}_PDFSETTXT-', text='PDF settings:', visible=False)
                    )],
                    [sg.pin(
                        sg.Text(key=f'-PAR{n_params}_PDFSET-', text='', relief="groove", border_width=1, visible=False)
                    )]
                ]),
                sg.VerticalSeparator(),
                sg.Column([
                    [sg.Text('Comments/Metadata:')],
                    [sg.Multiline(key=f'-PAR{n_params}_CM-', size=(40, 4))]
                ])
            ]
        ])]
    ])


### REFERENCES PRICES SECTION -----------------------------------------------------------------------------------------

# Get the current year to autocomplete combo box.
current_year = dt.date.today().year

# Define a group dictionary to assist with looking up the number of groups and the number of price lines within them.
pr_group_dict = {1: 0}
n_pr_groups = len(pr_group_dict)

# Define a side-dictionary to store the relevant information for further processing
MC_holding_dict = {}

exchr_period_dict = {'1 year': 1, '2 years': 2, '5 years': 5, '10 years': 10, '20 years': 20, 'All dates': 'all'}

# Use the World Bank API to access the most recent macroeconomic data.
# exch_rate_table = wb.data.DataFrame('DPANUSSPB', time=range(1987, current_year+1), db=15)
# cpi_table = wb.data.DataFrame('FP.CPI.TOTL', time=range(1960, current_year+1), db=2)

'''cpi_table = pd.read_csv('D:/Local Account/Documents/MonteCarlo CBA/cpi.csv', index_col='economy')
exchr_table = pd.read_csv('D:/Local Account/Documents/MonteCarlo CBA/exchr.csv', index_col='economy')
econ_codes = pd.read_csv('D:/Local Account/Documents/MonteCarlo CBA/econ_codes.csv', index_col='economy')'''

cpi_table = pd.read_csv(os.getcwd() + '/cpi.csv', index_col='economy')
exchr_table = pd.read_csv(os.getcwd() + '/exchr.csv', index_col='economy')
econ_codes = pd.read_csv(os.getcwd() + '/econ_codes.csv', index_col='economy')

#print(os.getcwd())
cpi_table_cols = list(cpi_table.columns)
cpi_table_cols.reverse()
cpi_numr = 1
cpi_numr_yr = 0

ISO3_list = list(cpi_table.index.values)

country_names = list(econ_codes[econ_codes['ISO3'].isin(ISO3_list)].index.values)


# Scan the data for the most recent CPI year available for country.
def cpi_numerator(econ_code, current_year):
    global cpi_numr
    global cpi_numr_yr
    for y in range(current_year, 1959, -1):
        try:
            cpi_numr = cpi_table.loc[econ_code, f'{y}']
        except KeyError:
            cpi_numr = np.NaN

        if not np.isnan(cpi_numr):
            cpi_numr_yr = y
            break


# Scan CPI data for the target CPI and return the relevent adjustment factor.
# WARNING: If there is no CPI data for the chosen year then the adjustment factor will return 1.
# It will be up to the analyst to familiarise themselves with the CPI data and notice irregularities.
def cpi_factor(econ_code, value_year):
    try:
        cpi_denom = cpi_table.loc[econ_code, f'{value_year}']
    except KeyError:
        cpi_denom = cpi_numr

    if np.isnan(cpi_denom):
        cpi_denom = cpi_numr

    return cpi_numr / cpi_denom


# Default CPI to Australia.
cpi_numerator('AUS', current_year)


# Scan the data for the appropriate CPI data
def exchange_rate(econ_code, start_year, period, metric):
    if period == 'All Dates':
        years = list(exchr_table.columns)
    else:
        # Find first valid year, then count backwards.
        start_year_valid = start_year
        while True:
            try:
                exchr_table.loc[econ_code, f'{start_year_valid}']
                break
            except KeyError:
                start_year_valid -= 1

        years = [f'{yr}' for yr in range(start_year_valid, start_year_valid - exchr_period_dict[period], -1)]
    # This returns a series.
    filter_exchr_table = exchr_table.loc[econ_code, years]
    if metric == 'Median':
        return filter_exchr_table.median()
    if metric == 'Mean':
        return filter_exchr_table.mean()


# Default home exchange rate to Australia.
exchr_home = exchange_rate('AUS', current_year, '5 years', 'Median')


def set_size(element, size):
    # Only work for sg.Column when `scrollable=True` or `size not (None, None)`
    options = {'width': size[0], 'height': size[1]}
    if element.Scrollable or element.Size != (None, None):
        element.Widget.canvas.configure(**options)
    else:
        element.Widget.pack_propagate(0)
        element.set_size(size)


# Function to add a new group to the layout.
def add_price_group(grp):
    window.extend_layout(window['-PRICE_GROUPS-'], [
        [sg.Frame(title=f'Price Group {grp}', key=f'-PRICE_GROUP_{grp}-', layout=[
            [sg.Text('Group ID:'), sg.Input(key=f'-PG{grp}_ID-', size=10, enable_events=True)],
            [sg.Text('Probability\nDistribution\nFunction:'),
             sg.Combo(key=f'-PG{grp}_MC_TYPE-', values=['None', 'Uniform', 'Bounded normal-like', 'Triangular',
                                                        'PERT', 'Folded normal', 'Truncated normal', 'Exponential',
                                                        'Log-logistic',
                                                        'Log-normal', 'Custom Kinked PDF'], enable_events=True, default_value='None',
                      size=15),
             sg.Text(key=f'-PG{grp}_PDFSETTXT-', text='PDF settings:', visible=False),
             sg.Text(key=f'-PG{grp}_PDFSET-', text='', relief="groove", border_width=1, visible=False)],
            [sg.Button('Add Price Line', key=f'-ADD_TO_PG_{grp}-')]
        ])]])


# Function to add a new value line to a group.
def add_price_line(grp, default_country, default_year):
    lin = pr_group_dict[grp] + 1
    window.extend_layout(window[f'-PRICE_GROUP_{grp}-'], [
        [sg.Frame(f'Group {grp}, Item {lin}', [
            [sg.Column([
                [sg.Text('Value ID:')],
                [sg.Input(key=f'-PG{grp}_LIN{lin}_ID-', size=10, enable_events=True)]]),
                sg.VerticalSeparator(),
                sg.Column([
                    [sg.Text('Units:')],
                    [sg.Input(key=f'-PG{grp}_LIN{lin}_UN-', size=10, enable_events=True)]]),
                sg.VerticalSeparator(),
                sg.Column([
                    [sg.Text('Nominal Value:')],
                    [sg.Input(key=f'-PG{grp}_LIN{lin}_NV-', size=10, enable_events=True)]]),
                sg.VerticalSeparator(),
                sg.Column([
                    [sg.Text('Currency:')],
                    [sg.Combo(key=f'-PG{grp}_LIN{lin}_CUR-', values=ISO3_list,
                              default_value=econ_codes._get_value(default_country, 'ISO3'), size=10,
                              enable_events=True)]]),
                sg.VerticalSeparator(),
                sg.Column([
                    [sg.Text('Year:')],
                    [sg.Combo(key=f'-PG{grp}_LIN{lin}_CURYR-', values=list(range(current_year, 1950, -1)),
                              default_value=default_year, enable_events=True)]]),
                sg.Column([
                    [sg.Text('Currency Conversion\nFactor:')],
                    [sg.Text(key=f'-PG{grp}_LIN{lin}_CCF-', text='1.0', size=10, relief="groove", border_width=1)]]),
                sg.Column([
                    [sg.Text('Real Value:\n ')],
                    [sg.Text(key=f'-PG{grp}_LIN{lin}_RV-', text='', size=10, relief="groove", border_width=1)]]),
                sg.VerticalSeparator(),
                sg.Column([
                    [sg.Text('Adjustment Factor:\n ')],
                    [sg.Input(key=f'-PG{grp}_LIN{lin}_AF-', default_text=1.0, size=10, enable_events=True)]]),
                sg.Column([
                    [sg.Text('Real Adjusted\nValue:')],
                    [sg.Text(key=f'-PG{grp}_LIN{lin}_RAV-', text='', size=10, relief="groove", border_width=1)]]),
                sg.VerticalSeparator(),
                sg.Column([
                    [sg.Text('Probability Distribution\nFunction:')],
                    [sg.Combo(key=f'-PG{grp}_LIN{lin}_MC_TYPE-',
                              values=['None', 'Uniform', 'Bounded normal-like', 'Triangular',
                                      'PERT', 'Folded normal', 'Truncated normal',
                                      'Exponential',
                                      'Log-logistic',
                                      'Log-normal', 'Custom Kinked PDF'], enable_events=True,
                              default_value='None',
                              size=15)]]),
                sg.Column([
                    [sg.pin(sg.Text(key=f'-PG{grp}_LIN{lin}_PDFSETTXT-', text='PDF settings:', visible=False))],
                    [sg.pin(sg.Text(key=f'-PG{grp}_LIN{lin}_PDFSET-', text='', relief="groove", border_width=1,
                                    visible=False))]]),
                sg.VerticalSeparator(),
                sg.Column([
                    [sg.Text('Comments/ Metadata:')],
                    [sg.Multiline(key=f'-PG{grp}_LIN{lin}_CM-', size=(40, 4))]])
            ]
        ])]
    ])


def calc_real_adj_values():
    for grp in range(1, n_pr_groups + 1):
        for lin in range(1, pr_group_dict[grp] + 1):
            if values[f'-PG{grp}_LIN{lin}_AF-'] == '':
                window[f'-PG{grp}_LIN{lin}_AF-'].update('1.0')
                window.refresh()
            # Test to see if the numeric imputs are numeric, if they are not, assume that it is a parameterised expression.
            if values[f'-PG{grp}_LIN{lin}_NV-'] != '':
                try:
                    float(values[f'-PG{grp}_LIN{lin}_NV-'])
                    par_expression = False
                except ValueError:
                    par_expression = True
                    window.refresh()
                # Adjustment factor should always be numeric. If not, clear all non-numeric characters.
                try:
                    float(values[f'-PG{grp}_LIN{lin}_AF-'])
                except ValueError:
                    window[f'-PG{grp}_LIN{lin}_AF-'].update(
                        re.findall('\d*[.]\d*', values[f'-PG{grp}_LIN{lin}_AF-'])[0])
                    window.refresh()

                currency_conversion_factor = exchr_home / exchange_rate(values[f'-PG{grp}_LIN{lin}_CUR-'], current_year,
                                                                        values['-PR_CUR_RANGE-'],
                                                                        values['-PR_CUR_POINT-'])
                window[f'-PG{grp}_LIN{lin}_CCF-'].update(np.around(currency_conversion_factor, 3))
                inflation_factor = cpi_factor(values[f'-PG{grp}_LIN{lin}_CUR-'], values[f'-PG{grp}_LIN{lin}_CURYR-'])

                if par_expression is False:
                    # Do all the calcs, save the un-rounded value in the
                    real_value = float(values[f'-PG{grp}_LIN{lin}_NV-']) * currency_conversion_factor * inflation_factor
                    window[f'-PG{grp}_LIN{lin}_RV-'].update(np.around(real_value, 3))

                    real_adj_value = real_value * float(values[f'-PG{grp}_LIN{lin}_AF-'])
                    window[f'-PG{grp}_LIN{lin}_RAV-'].update(np.around(real_adj_value, 3))
                    MC_holding_dict[f'-PG{grp}_LIN{lin}_RAV_FULL-'] = real_adj_value
                    window.refresh()
                else:
                    real_value = f'{values[f"-PG{grp}_LIN{lin}_NV-"]}*{np.around(currency_conversion_factor * inflation_factor, 3)}'
                    window[f'-PG{grp}_LIN{lin}_RV-'].update(real_value)

                    real_adj_value = f'{values[f"-PG{grp}_LIN{lin}_NV-"]}*{np.around(currency_conversion_factor * inflation_factor* float(values[f"-PG{grp}_LIN{lin}_AF-"]), 3)}'
                    window[f'-PG{grp}_LIN{lin}_RAV-'].update(real_adj_value)
                    MC_holding_dict[f'-PG{grp}_LIN{lin}_RAV_FULL-'] = f'{values[f"-PG{grp}_LIN{lin}_NV-"]}*{currency_conversion_factor * inflation_factor* float(values[f"-PG{grp}_LIN{lin}_AF-"])}'
            else:
                # If there is no Nominal Value, clear the decks.
                window[f'-PG{grp}_LIN{lin}_CCF-'].update('')
                window[f'-PG{grp}_LIN{lin}_RV-'].update('')
                window[f'-PG{grp}_LIN{lin}_RAV-'].update('')
                MC_holding_dict[f'-PG{grp}_LIN{lin}_RAV_FULL-'] = 0
                window.refresh()


def pdf_window_update(key, mc_dict):

    key_prefix = key.replace('_MC_TYPE-', '')
    key_dict = {f'{key_prefix}_PDFMIN-': 'Minimum',
                f'{key_prefix}_PDFMAX-': 'Maximum',
                f'{key_prefix}_PDFMEA-': 'Mean',
                f'{key_prefix}_PDFSIG-': 'Sigma',
                f'{key_prefix}_PDFMOD-': 'Mode',
                f'{key_prefix}_PDFMED-': 'Median',
                f'{key_prefix}_PDFSHA-': 'Shape',
                f'{key_prefix}_PDFSCA-': 'Scale',
                f'{key_prefix}_PDFRAT-': 'Rate (lambda)',
                f'{key_prefix}_PDF_WARN-': 'Test Mean',
                f'{key_prefix}_PDFX-': 'Custom x points',
                f'{key_prefix}_PDFY-': 'Custom y points',
                f'{key_prefix}_PDF_REACHT-': 'Rand each t'}

    window[f'{key_prefix}_PDFSETTXT-'].update(visible=True)
    window[f'{key_prefix}_PDFSET-'].update(visible=True)

    pdf_seting_txt = ''
    for k in key_dict.keys():
        if k in mc_dict:
            pdf_seting_txt += f'\n{key_dict[k]}: {mc_dict[k]}'

    window[f'{key_prefix}_PDFSET-'].update(pdf_seting_txt.strip())
    window.refresh()

def get_pdf_settings(key, auto_close=False):
    window.Disable()

    key_prefix = key.replace('_MC_TYPE-', '')
    #print(key_prefix)
    old_values = MC_holding_dict.copy()
    new_values = MC_holding_dict.copy()

    key_list = [f'{key_prefix}_PDFMIN-',
                f'{key_prefix}_PDFMAX-',
                f'{key_prefix}_PDFMEA-',
                f'{key_prefix}_PDFSIG-',
                f'{key_prefix}_PDFMOD-',
                f'{key_prefix}_PDFMED-',
                f'{key_prefix}_PDFSHA-',
                f'{key_prefix}_PDFSCA-',
                f'{key_prefix}_PDFRAT-',
                f'{key_prefix}_PDF_WARN-',
                f'{key_prefix}_PDF_REACHT-',
                f'{key_prefix}_PDFX-',
                f'{key_prefix}_PDFY-']

    key_dict = {f'{key_prefix}_PDFMIN-': 'Minimum',
                f'{key_prefix}_PDFMAX-': 'Maximum',
                f'{key_prefix}_PDFMEA-': 'Mean',
                f'{key_prefix}_PDFSIG-': 'Sigma',
                f'{key_prefix}_PDFMOD-': 'Mode',
                f'{key_prefix}_PDFMED-': 'Median',
                f'{key_prefix}_PDFSHA-': 'Shape',
                f'{key_prefix}_PDFSCA-': 'Scale',
                f'{key_prefix}_PDFRAT-': 'Rate (lambda)',
                f'{key_prefix}_PDF_WARN-': 'Test Mean',
                f'{key_prefix}_PDF_REACHT-': 'Rand each t',
                f'{key_prefix}_PDFX-': 'Custom x points',
                f'{key_prefix}_PDFY-': 'Custom y points'}

    # Generate a new values dictionary and purge all associated PDF settings with the event.
    new_values.update({f'{key_prefix}_MC_TYPE-': values[f'{key_prefix}_MC_TYPE-']})

    for k in key_list:
        if k in old_values:
            new_values.pop(k, 'not present')

    # Make PDF setting text visible in the main window. If PDF type is None, simply return the purged values dictionary and make setting fields invisible.
    if new_values[key] == 'None':
        window[f'{key_prefix}_PDFSETTXT-'].update(visible=False)
        window[f'{key_prefix}_PDFSET-'].update('')
        window[f'{key_prefix}_PDFSET-'].update(visible=False)
        window.Enable()
        window.BringToFront()
        return new_values
    else:
        window[f'{key_prefix}_PDFSETTXT-'].update(visible=True)
        window[f'{key_prefix}_PDFSET-'].update(visible=True)

    # Create the layout pieces.
    pr_pdf_layout = [[sg.Text(f'The selected Probability Distribution Function is: {values[key]}')]]

    pr_pdf_min = [[sg.Text('PDF Minimum:'),
                   sg.Input(key=f'{key_prefix}_PDFMIN-', default_text=old_values.get(f'{key_prefix}_PDFMIN-', '0.9'),
                            enable_events=True, size=10)]]

    pr_pdf_max = [[sg.Text('PDF Maximum:'),
                   sg.Input(key=f'{key_prefix}_PDFMAX-', default_text=old_values.get(f'{key_prefix}_PDFMAX-', '1.1'),
                            enable_events=True, size=10)]]

    if values[f'{key_prefix}_MC_TYPE-'] == 'Log-normal':
        pr_pdf_mea = [[sg.Text('PDF Mean:'),
                       sg.Input(key=f'{key_prefix}_PDFMEA-',
                                default_text=old_values.get(f'{key_prefix}_PDFMEA-', '0.0'),
                                enable_events=True, size=10)]]
    else:
        pr_pdf_mea = [[sg.Text('PDF Mean:'),
                       sg.Input(key=f'{key_prefix}_PDFMEA-',
                                default_text=old_values.get(f'{key_prefix}_PDFMEA-', '1.0'),
                                enable_events=True, size=10)]]

    pr_pdf_sig = [[sg.Text('PDF Sigma:'),
                   sg.Input(key=f'{key_prefix}_PDFSIG-', default_text=old_values.get(f'{key_prefix}_PDFSIG-', '0.02'),
                            enable_events=True, size=10)]]

    pr_pdf_mod = [[sg.Text('PDF Mode:'),
                   sg.Input(key=f'{key_prefix}_PDFMOD-', default_text=old_values.get(f'{key_prefix}_PDFMOD-', '1.0'),
                            enable_events=True, size=10)]]

    pr_pdf_med = [[sg.Text('PDF Median:'),
                   sg.Input(key=f'{key_prefix}_PDFMED-', default_text=old_values.get(f'{key_prefix}_PDFMED-', '1.0'),
                            enable_events=True, size=10)]]

    pr_pdf_sha = [[sg.Text('PDF Shape:'),
                   sg.Input(key=f'{key_prefix}_PDFSHA-', default_text=old_values.get(f'{key_prefix}_PDFSHA-', '50.0'),
                            enable_events=True, size=10)]]

    pr_pdf_sca = [[sg.Text('PDF Scale:'),
                   sg.Input(key=f'{key_prefix}_PDFSCA-', default_text=old_values.get(f'{key_prefix}_PDFSCA-', ''),
                            enable_events=True, size=10)]]

    pr_pdf_rat = [[sg.Text('PDF Rate (lambda):'),
                   sg.Input(key=f'{key_prefix}_PDFRAT-', default_text=old_values.get(f'{key_prefix}_PDFRAT-', '1.0'),
                            enable_events=True, size=10)]]

    pr_pdf_xs = [[sg.Text('PDF Custom Xs:'),
                   sg.Multiline(key=f'{key_prefix}_PDFX-', default_text=old_values.get(f'{key_prefix}_PDFX-', '1.0'),
                            enable_events=True, size=(10, 4))]]

    pr_pdf_ys = [[sg.Text('PDF Custom Ys:'),
                  sg.Multiline(key=f'{key_prefix}_PDFY-', default_text=old_values.get(f'{key_prefix}_PDFY-', '1.0'),
                               enable_events=True, size=(10, 4))]]

    pr_pdf_warn = [[sg.Text('', key=f'{key_prefix}_PDF_WARN-')]]

    pr_pdf_all_per = [[sg.Checkbox('Randomise in each time period?',
                                   default=old_values.get(f'{key_prefix}_PDF_REACHT-', False),
                                   key=f'{key_prefix}_PDF_REACHT-')]]

    if new_values[key] in ['Uniform', 'Bounded normal-like']:
        pr_pdf_layout += pr_pdf_min, pr_pdf_max, pr_pdf_warn, pr_pdf_all_per

    if new_values[key] in ['Triangular', 'PERT']:
        pr_pdf_layout += pr_pdf_min, pr_pdf_mod, pr_pdf_max, pr_pdf_warn, pr_pdf_all_per

    if new_values[key] in ['Folded normal', 'Truncated normal', 'Log-normal']:
        pr_pdf_layout += pr_pdf_mea, pr_pdf_sig, pr_pdf_warn, pr_pdf_all_per

    if new_values[key] == 'Exponential':
        pr_pdf_layout += pr_pdf_rat, pr_pdf_warn, pr_pdf_all_per

    if new_values[key] == 'Log-logistic':
        pr_pdf_layout += pr_pdf_med, pr_pdf_sha, pr_pdf_warn, pr_pdf_all_per

    if new_values[key] == 'Custom Kinked PDF':
        pr_pdf_layout += pr_pdf_xs, pr_pdf_ys, pr_pdf_warn, pr_pdf_all_per

    pr_pdf_layout += [[sg.Button('Submit'), sg.Button('Cancel'), sg.Button('Hidden', visible=False)]]

    pr_pdf_window = sg.Window(f'Reference Price PDF Settings', pr_pdf_layout, finalize=True,
                              return_keyboard_events=True, disable_close=True)

    pr_pdf_window['Hidden'].click()

    # Main loop
    while True:
        pr_pdf_event, pr_pdf_values = pr_pdf_window.read()
        # If the window has any of the PDF input fields, update the new_values dictionary with those values.
        pdf_seting_txt = ''
        if auto_close is True:
            pr_pdf_window['Submit'].click()
        if pr_pdf_values is not None:
            for k in key_list:
                if k in pr_pdf_values:
                    new_values.update({k: pr_pdf_values[k]})
                    pr_pdf_window.refresh()
                    pdf_seting_txt += f'\n{key_dict[k]}: {pr_pdf_values[k]}'

        PDF_mean_warn = PDF_meantest(new_values[key], key_prefix, pr_pdf_values)
        if PDF_mean_warn == 1:
            pr_pdf_window[f'{key_prefix}_PDF_WARN-'].update('The PDF Mean is 1.0')
        else:
            pr_pdf_window[f'{key_prefix}_PDF_WARN-'].update(
                f'Warning: The PDF Mean is {PDF_mean_warn},\nplease ensure that the PDF is behaving as intended.')

        if pr_pdf_event == 'Submit':
            user_values = new_values
            window[f'{key_prefix}_PDFSET-'].update(pdf_seting_txt.strip())
            break
        if pr_pdf_event == 'Cancel':
            user_values = old_values
            break

    pr_pdf_window.close()
    window.Enable()
    window.BringToFront()
    return user_values


### QUANTITY SCENARIOS SECTION ----------------------------------------------------------------------------------------
# Counter for the number of scenarios additional to the base case (Scenario 0)
n_scenarios = 0

# Dictionary for the groups and quantity lines in each scenario tab.
qs_group_dict = {1: 1}


def add_scenario_tab():
    window['-QNT_TABGROUP-'].add_tab(sg.Tab(title=f'Scenario {n_scenarios}', layout=[
        [sg.Text(f'Scenario {n_scenarios}\nDescription:')],
        [sg.Multiline(key=f'-QNT_SCN{n_scenarios}_DESC-', size=(40, 4))]
    ], key=f'-QNT_SCENARIO_{n_scenarios}-'))

    window.extend_layout(window[f'-QNT_SCENARIO_{n_scenarios}-'], [
        [sg.Button(button_text='Add Quantity Group', key=f'ADD_QNT_GROUP_SCN{n_scenarios}')],
        [sg.Frame(title='', key=f'-QNT_SCENARIO_{n_scenarios}_FRAME-', layout=[])]
    ])

    for grp in range(1, len(qs_group_dict) + 1):
        add_quantity_group(n_scenarios, grp)
        for lin in range(1, qs_group_dict[grp] + 1):
            add_quantity_line(n_scenarios, grp)


def add_quantity_group(scenario, grp):
    if scenario == 0:
        quantity_group_type = [sg.Text('Group Type:'),
                               sg.Combo(key=f'-SCN{scenario}_QG{grp}_GRP_TYPE-', values=['Costs', 'Benefits'],
                                        default_value='Costs', size=10, enable_events=True)]
    else:
        quantity_group_type = [sg.Text('Group Type:'),
                               sg.Input(default_text='Costs', key=f'-SCN{scenario}_QG{grp}_GRP_TYPE-', disabled=True,
                                        size=10)]

    window.extend_layout(window[f'-QNT_SCENARIO_{scenario}_FRAME-'], [
        [sg.Frame(title=f'Quantity Group {grp}', key=f'-SCN{scenario}_QG{grp}-', layout=[
            [sg.Text('Group ID:'), sg.Input(key=f'-SCN{scenario}_QG{grp}_ID-', size=10, enable_events=True)],
            quantity_group_type,
            [sg.Text('Probability\nDistribution\nFunction:'),
             sg.Combo(key=f'-SCN{scenario}_QG{grp}_MC_TYPE-',
                      values=['None', 'Uniform', 'Bounded normal-like', 'Triangular',
                              'PERT', 'Folded normal', 'Truncated normal', 'Exponential',
                              'Log-logistic',
                              'Log-normal', 'Custom Kinked PDF'], enable_events=True, default_value='None',
                      size=15),
             sg.pin(sg.Text(key=f'-SCN{scenario}_QG{grp}_PDFSETTXT-', text='PDF settings:', visible=False)),
             sg.pin(sg.Text(key=f'-SCN{scenario}_QG{grp}_PDFSET-', text='', relief="groove", border_width=1,
                            visible=False))],
            [sg.Button('Add Quantity Line', key=f'-ADD_TO_QG_{grp}-')]
        ])]])

    window.refresh()

    try:
        window[f'-SCN{scenario}_QG{grp}_GRP_TYPE-'].update(values[f'-SCN0_QG{grp}_GRP_TYPE-'])
    except Exception:
        pass


def add_quantity_line(scenario, grp):
    lin = qs_group_dict[grp]
    try:
        prices = price_ID_list
    except:
        prices = []
    try:
        stk = stakeholder_list
    except:
        stk = []
    try:
        geo = geozones_list
    except:
        geo = []

    window.extend_layout(window[f'-SCN{scenario}_QG{grp}-'], [
        [sg.Column([
            [sg.Text('Line Item ID:')],
            [sg.Input(key=f'-SCN{scenario}_QG{grp}_LIN{lin}_ID-', size=10, enable_events=True)]]),
            sg.VerticalSeparator(),
            sg.Column([
                [sg.Text('Value:')],
                [sg.Combo(key=f'-SCN{scenario}_QG{grp}_LIN{lin}_PRICE-', size=10, enable_events=True,
                          values=prices)]]),
            sg.Column([
                [sg.Text('Price/Units:')],
                [sg.Text(key=f'-SCN{scenario}_QG{grp}_LIN{lin}_PRUN-', size=10, relief="groove", border_width=1)]]),
            sg.VerticalSeparator(),
            sg.Column([
                [sg.Text('Stakeholder\nGroup:')],
                [sg.Combo(key=f'-SCN{scenario}_QG{grp}_LIN{lin}_STKE-', size=10, enable_events=True,
                          values=stk)]]),
            sg.Column([
                [sg.Text('Geographic\nZone:')],
                [sg.Combo(key=f'-SCN{scenario}_QG{grp}_LIN{lin}_GEOZN-', size=10, enable_events=True,
                          values=geo)]]),
            sg.VerticalSeparator(),
            sg.Column([
                [sg.Text('Period range:')],
                [sg.Multiline(key=f'-SCN{scenario}_QG{grp}_LIN{lin}_PRANGE-', size=(10, 4), enable_events=True)]]),
            sg.Column([
                [sg.Text('Quantity:')],
                [sg.Multiline(key=f'-SCN{scenario}_QG{grp}_LIN{lin}_PQUANT-', size=(10, 4), enable_events=True)]]),
            sg.Text(text='\U0001F4C8', key=f'-SCN{scenario}_QG{grp}_LIN{lin}_PGRAPH-',
                    enable_events=True, font=(None, 15)),
            sg.VerticalSeparator(),
            sg.Column([
                [sg.Text('Probability\nDistribution\nFunction:')],
                [sg.Combo(key=f'-SCN{scenario}_QG{grp}_LIN{lin}_MC_TYPE-',
                          values=['None', 'Uniform', 'Bounded normal-like', 'Triangular',
                                  'PERT', 'Folded normal', 'Truncated normal',
                                  'Exponential',
                                  'Log-logistic',
                                  'Log-normal', 'Custom Kinked PDF'], enable_events=True,
                          default_value='None',
                          size=15)]]),
            sg.Column([
                [sg.pin(
                    sg.Text(key=f'-SCN{scenario}_QG{grp}_LIN{lin}_PDFSETTXT-', text='PDF settings:', visible=False))],
                [sg.pin(
                    sg.Text(key=f'-SCN{scenario}_QG{grp}_LIN{lin}_PDFSET-', text='', relief="groove", border_width=1,
                            visible=False))]]),
            sg.VerticalSeparator(),
            sg.Column([
                [sg.Text('Event Outcome\nDependency:')],
                [sg.Combo(key=f'-SCN{scenario}_QG{grp}_LIN{lin}_DEPENDS-', size=10, enable_events=True,
                          values=['None'] + model_outcome_list, default_value='None')]]),
            sg.Column([
                [sg.Text('Comments/ Metadata:')],
                [sg.Multiline(key=f'-SCN{scenario}_QG{grp}_LIN{lin}_CM-', size=(40, 4))]])
        ]])


class Canvas(FigureCanvasTkAgg):
    """
    Create a canvas for matplotlib pyplot under tkinter/PySimpleGUI canvas
    """

    def __init__(self, figure=None, master=None):
        super().__init__(figure=figure, master=master)
        self.canvas = self.get_tk_widget()
        self.canvas.pack(side='top', fill='both', expand=1)


def quantity_stream_popup(period_ranges, expressions):
    window.Disable()

    n_periods = max(map(int, re.split(r'\n+|,+', period_ranges)))
    period_ranges_list = period_ranges.split('\n')
    expressions_list = expressions.split('\n')
    n_lines = len(period_ranges_list)

    period_array = np.linspace(0, n_periods, num=n_periods + 1)
    quantity_array = np.zeros(n_periods + 1)

    for line in range(0, n_lines):
        start, end = map(int, period_ranges_list[line].split(','))
        end += 1
        expression = expressions_list[line]
        t = period_array[start:end]
        quantity_array[start:end] = eval(f'{expression}')

    qnt_stream_layout = [
        [sg.Frame('', [[sg.Canvas(key='-QNT_PLOT_CANVAS-', expand_x=True, expand_y=True, )]], size=(640, 480))],
        [sg.Button('Close')]
    ]

    qnt_stream_window = sg.Window('Quantity Stream', qnt_stream_layout, finalize=True, resizable=True,
                                  font=('Helvetica', 10))

    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot()
    canvas = Canvas(fig, qnt_stream_window['-QNT_PLOT_CANVAS-'].Widget)

    if '_PGRAPH-' in event:
        ax.set_title('Quantity Stream')
        ax.set_ylabel("Quantity")
    else:
        ax.set_title('Probability Timeline')
        ax.set_ylabel("Probability")
    ax.set_xlabel("Period")
    ax.plot(period_array, quantity_array)
    canvas.draw()

    while True:
        qnt_stream_event, qnt_stream_values = qnt_stream_window.read()

        if qnt_stream_event == sg.WIN_CLOSED or qnt_stream_event == 'Close':
            break

    qnt_stream_window.close()
    window.Enable()
    window.BringToFront()
    window.Read()


model_events_dict = {}
model_events_user_settings = {}
model_outcome_list = []


def event_model_builder():
    global model_events_dict
    global model_events_user_settings
    global model_outcome_list
    window.Disable()

    def outcome_list(model_dict, model_event=None):
        reg_match = re.compile(r'-EVENT\d*_OUTCOME\d*_ID-')
        sub_dict = {key: model_dict[key] for key in [outs for outs in list(model_dict.keys()) if reg_match.match(outs)]}
        if model_event is None:
            return list(sub_dict.values())
        else:
            sub_dict = {key: sub_dict[key] for key in
                        list(filter(lambda evs: f'EVENT{model_event}' not in evs, list(sub_dict.keys())))}
            return list(sub_dict.values())

    def add_event(model_event):
        if f'-EVENT{model_event}_ID-' in model_events_user_settings.keys():
            default_id = model_events_user_settings[f'-EVENT{model_event}_ID-']
        else:
            default_id = f'E{model_event}'

        if f'-EVENT{model_event}_DEPENDS-' in model_events_user_settings.keys():
            default_select = model_events_user_settings[f'-EVENT{model_event}_DEPENDS-']
        else:
            default_select = []

        emb_window.extend_layout(emb_window['-EVENT_FRAME-'], [
            [sg.Frame(title=f'Event {model_event}', key=f'-EVENT_{model_event}-', layout=[
                [sg.Text('Event ID:'),
                 sg.Input(key=f'-EVENT{model_event}_ID-', default_text=default_id, size=10, enable_events=True)],
                [sg.Button('Add Outcome', key=f'-ADD_OUTCOME_{model_event}-')],
                [sg.Text('Event Dependencies:'), sg.Listbox(key=f'-EVENT{model_event}_DEPENDS-', size=10,
                                                            select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE,
                                                            values=outcome_list(model_events_user_settings,
                                                                                model_event),
                                                            default_values=default_select,
                                                            enable_events=True),
                 sg.Text(default_select, key=f'-EVENT{model_event}_DEPENDS_TXT-')],
                [sg.Frame(title='', key=f'-EVENT_{model_event}_OUTCOMES_FRAME-', layout=[[]])]
            ])]
        ])
        emb_window.refresh()

    def add_outcome(model_event, outcome):

        if f'-EVENT{model_event}_OUTCOME{outcome}_ID-' in model_events_user_settings.keys():
            default_id = model_events_user_settings[f'-EVENT{model_event}_OUTCOME{outcome}_ID-']
        else:
            default_id = f'E{model_event}_O{outcome}'

        outcome_layout = [
            sg.Column([
                [sg.Text(f'Event {model_event}, Outcome {outcome}')],
                [sg.Text('Outcome ID:'), sg.Input(key=f'-EVENT{model_event}_OUTCOME{outcome}_ID-', size=10,
                                                  default_text=default_id)]
            ])
        ]

        for scn in range(0, n_scenarios + 1):
            if f'-EVENT{model_event}_OUTCOME{outcome}_SCN{scn}_RANGE-' in model_events_user_settings.keys():
                default_range = model_events_user_settings[f'-EVENT{model_event}_OUTCOME{outcome}_SCN{scn}_RANGE-']
            elif scn == 0:
                default_range = '...'
            else:
                default_range = '<<<'

            if f'-EVENT{model_event}_OUTCOME{outcome}_SCN{scn}_WEIGHT-' in model_events_user_settings.keys():
                default_weight = model_events_user_settings[f'-EVENT{model_event}_OUTCOME{outcome}_SCN{scn}_WEIGHT-']
            elif scn == 0:
                default_weight = '1.0'
            else:
                default_weight = '<<<'

            if f'-EVENT{model_event}_OUTCOME{outcome}_SCN{scn}_NOREPS-' in model_events_user_settings.keys():
                default_noreps = model_events_user_settings[f'-EVENT{model_event}_OUTCOME{outcome}_SCN{scn}_NOREPS-']
            elif scn == 0:
                default_noreps = '0'
            else:
                default_noreps = '<<<'

            if f'-EVENT{model_event}_OUTCOME{outcome}_SCN{scn}_MAXREPS-' in model_events_user_settings.keys():
                default_maxreps = model_events_user_settings[f'-EVENT{model_event}_OUTCOME{outcome}_SCN{scn}_MAXREPS-']
            elif scn == 0:
                default_maxreps = 'None'
            else:
                default_maxreps = '<<<'

            scenario_inputs = sg.Column([
                [sg.Text(f'Outcome Weights for Scenario {scn}')],
                [sg.Column([
                    [sg.Text('Outcome Range:')],
                    [sg.Multiline(key=f'-EVENT{model_event}_OUTCOME{outcome}_SCN{scn}_RANGE-', size=(10, 4),
                                  enable_events=True, default_text=default_range)]
                ]),
                    sg.Column([
                        [sg.Text('Outcome Weight:')],
                        [sg.Multiline(key=f'-EVENT{model_event}_OUTCOME{outcome}_SCN{scn}_WEIGHT-', size=(10, 4),
                                      enable_events=True, default_text=default_weight)]
                    ]),
                    sg.Column([
                        [sg.Text('Periods\nwithout\nrepetition:')],
                        [sg.Input(key=f'-EVENT{model_event}_OUTCOME{outcome}_SCN{scn}_NOREPS-', size=10,
                                  enable_events=True, default_text=default_noreps)]
                    ]),
                    sg.Column([
                        [sg.Text('Maximum\nnumber of\nrepetitions:')],
                        [sg.Input(key=f'-EVENT{model_event}_OUTCOME{outcome}_SCN{scn}_MAXREPS-', size=10,
                                  enable_events=True, default_text=default_maxreps)]
                    ]),
                    sg.VerticalSeparator()]
            ])
            outcome_layout.append(scenario_inputs)

        emb_window.extend_layout(emb_window[f'-EVENT_{model_event}_OUTCOMES_FRAME-'], [outcome_layout])
        emb_window.refresh()

    # Automatically adjust the window width for the number of scenarios.
    if n_scenarios > 2:
        emb_width = min([(1139 + 458 * (n_scenarios - 2)), 2000])
    else:
        emb_width = 1139

    # Automatically adjust the window height for the number of events and outcomes.
    emb_height = 3
    for e in range(1, len(model_events_dict) + 1):
        if emb_height < 900:
            emb_height += 133
        for o in range(1, model_events_dict[e] + 1):
            if emb_height < 900:
                emb_height += 138

    emb_layout = [
        [sg.Button('Add Event', key='-EVENT_ADD-')],
        [sg.Column(key='-EVENT_COLS-', scrollable=True, size=(emb_width, emb_height),
                   layout=[
                       [sg.Frame(layout=[[]], title='', key='-EVENT_FRAME-')]
                   ])],
        [sg.Button('Submit', key='-EVENT_SUBMIT-')]
    ]

    emb_window = sg.Window('Event Model Builder', emb_layout, finalize=True, enable_close_attempted_event=True,
                           resizable=True)

    # Pre-load window with fields.
    for event in range(1, len(model_events_dict) + 1):
        add_event(event)
        for outcome in range(1, model_events_dict[event] + 1):
            add_outcome(event, outcome)

    while True:
        emb_event, emb_values = emb_window.read()

        if emb_event == '-EVENT_ADD-':
            model_events_dict[len(model_events_dict) + 1] = 0
            add_event(len(model_events_dict))

            # Limit the height of the column to 900 pixels.
            if emb_window['-EVENT_COLS-'].get_size()[1] < 900:
                resize = 133 + emb_window['-EVENT_COLS-'].get_size()[1]
                set_size(emb_window['-EVENT_COLS-'], (None, resize))
            emb_window.refresh()
            emb_window['-EVENT_COLS-'].contents_changed()

        if any([re.match(r'-ADD_OUTCOME_\d*-', emb_event)]):
            g = [int(g) for g in re.findall(r'\d+', emb_event)]
            model_events_dict[g[0]] += 1
            add_outcome(g[0], model_events_dict[g[0]])
            # Pre-emptively add the default outcome name to the values dictionary so that outcome lists can be updated.
            emb_values[f'-EVENT{g[0]}_OUTCOME{model_events_dict[g[0]]}_ID-'] = f'E{g[0]}_O{model_events_dict[g[0]]}'
            # Limit the height of the column to 900 pixels.
            if emb_window['-EVENT_COLS-'].get_size()[1] < 900:
                resize = 138 + emb_window['-EVENT_COLS-'].get_size()[1]
                set_size(emb_window['-EVENT_COLS-'], (None, resize))
            emb_window.refresh()
            emb_window['-EVENT_COLS-'].contents_changed()

        if any([re.match(r'-ADD_OUTCOME_\d*-', emb_event), re.match(r'-EVENT\d*_OUTCOME\d*_ID-', emb_event)]):
            for event in range(1, len(model_events_dict) + 1):
                old_selects = emb_values[f'-EVENT{event}_DEPENDS-']
                new_outs_list = outcome_list(emb_values, event)
                emb_window[f'-EVENT{event}_DEPENDS-'].update(new_outs_list)
                select_indexes = [new_outs_list.index(sel) for sel in old_selects if sel in new_outs_list]
                emb_window[f'-EVENT{event}_DEPENDS-'].update(set_to_index=select_indexes)
                emb_window[f'-EVENT{event}_DEPENDS_TXT-'].update(value=emb_values[f'-EVENT{event}_DEPENDS-'])
                emb_window.refresh()
            #print(emb_window['-EVENT_COLS-'].get_size())

        if any([re.match(r'-EVENT\d*_DEPENDS-', emb_event)]):
            g = [int(g) for g in re.findall(r'\d+', emb_event)]
            emb_window[f'-EVENT{g[0]}_DEPENDS_TXT-'].update(value=emb_values[f'-EVENT{g[0]}_DEPENDS-'])

        if emb_event == '-EVENT_SUBMIT-':
            model_events_user_settings = emb_values
            model_outcome_list = outcome_list(model_events_user_settings)

            break
        if emb_event == sg.WINDOW_CLOSE_ATTEMPTED_EVENT:
            break
        # print(emb_event, emb_values)

    emb_window.close()
    window.Enable()
    window.BringToFront()


### RESULTS SECTION --------------------------------------------------------------------------------------------------
sim_initialisation = True
math_expressions = ['np', 'min', 'max', 'ceil', 'fabs', 'floor', 'trunc', 'exp', 'expm', 'log', 'pow', 'sqrt', 'pi', 'e', 'clip', 'inf']

def expression_to_array(n_periods, period_ranges, expressions):
    period_ranges_list = period_ranges.split('\n')
    expressions_list = expressions.split('\n')
    n_lines = len(period_ranges_list)

    #period_array = np.linspace(0, n_periods - 1, num=n_periods)
    out_array = np.zeros((n_simulations, n_periods))

    end = 0
    for line in range(0, n_lines):
        if any([period_ranges_list[line] == '...', period_ranges_list[line] == '…']):
            start = end
            end = n_periods + 1
        else:
            start, end = map(int, period_ranges_list[line].split(','))
            end += 1
        expression = expressions_list[line]

        # Search the expression for any parameter or function strings and replace them with a dictionary address
        strs_to_replace = re.findall(r'[a-zA-Z^]+', expression)
        strs_iters = re.finditer(r'[a-zA-Z^]+', expression)
        replacement_strs = []
        for m in range(0, len(strs_to_replace)):
            if strs_to_replace[m] not in math_expressions:
                if strs_to_replace[m] in function_names:
                    replacement_strs.append('function_dict[' + f'"{strs_to_replace[m]}"]')
                else:
                    replacement_strs.append('parameter_dict["' + f'{strs_to_replace[m]}' + f'"]["expression"][:,{start}:{end}]')
            else:
                replacement_strs.append(f'np.{strs_to_replace[m]}')
        #replacement_strs = ['parameter_dict["' + f'{ps}' + f'"]["expression"][:,{start}:{end}]' for ps in strs_to_replace]

        replacement_list = []
        rep = 0
        for match in strs_iters:
            replacement_list.append((match, replacement_strs[rep]))
            rep += 1
        for r in reversed(replacement_list):
            expression = expression[:r[0].start()] + r[1] + expression[r[0].end():]
        print(expression)
        #t = period_array[start:end]
        out_array[:, start:end] = eval(f'{expression}')
        #print(out_array)

    return out_array


def set_custom_dfs(x_points, y_points):
    global prob_df
    global y_points_norm
    global zone_areas
    global mean_df
    global cuml_vect

    area_df = 0
    for zone in range(0, len(x_points)-1):
        miny = min(y_points[zone], y_points[zone+1])
        maxy = max(y_points[zone], y_points[zone+1])
        widthx = x_points[zone+1]-x_points[zone]
        area_df += widthx*miny + 0.5*widthx*(maxy-miny)

    y_points_norm = y_points / area_df
    prob_df = interpolate.interp1d(x=x_points, y=y_points_norm, bounds_error=False, fill_value=0)

    zone_areas = np.zeros(len(x_points)-1)
    zone_means = np.zeros(len(x_points)-1)

    for zone in range(0, len(x_points) - 1):
        miny = min(y_points_norm[zone], y_points_norm[zone + 1])
        maxy = max(y_points_norm[zone], y_points_norm[zone + 1])
        widthx = x_points[zone + 1] - x_points[zone]
        zone_areas[zone] = widthx * miny + 0.5 * widthx * (maxy - miny)
        zone_means[zone] = (x_points[zone]*prob_df(x_points[zone])+x_points[zone+1]*prob_df(x_points[zone+1]))/(prob_df(x_points[zone])+prob_df(x_points[zone+1]))

    mean_df = np.sum(zone_areas * zone_means)

    def cuml_df(x):
        for zone in range(0, len(x_points) - 1):
            if zone == 0:
                prev_area = 0
            else:
                prev_area = np.sum(zone_areas[:zone])
            if x_points[zone] < x < x_points[zone + 1]:
                miny = min(y_points_norm[zone], prob_df(x))
                maxy = max(y_points_norm[zone], prob_df(x))
                x_start = x_points[zone]
                return prev_area + (x - x_start) * miny + 0.5 * (x - x_start) * (maxy - miny)

    cuml_vect = np.vectorize(cuml_df)


class custom_kinks_gen(rv_continuous):

    "Custom Kinks distribution"

    def _pdf(self, x):
        return prob_df(x)

    def _cdf(self, x):
        return cuml_vect(x)


def monte_carlo_expression(prefix, PDF_func, n_simulations):
    # Set the size based on whether the MC settings requires RNG in all time periods.
    if f'{prefix}_PDF_REACHT-' in MC_holding_dict:
        if MC_holding_dict[f'{prefix}_PDF_REACHT-'] is True:
            dims = (n_simulations, n_simulation_periods)
        else:
            dims = (n_simulations, 1)
    else:
        dims = (n_simulations, 1)

    MC_expr_dict = {}

    MC_expr_dict['PDF_min'] = MC_holding_dict.get(f'{prefix}_PDFMIN-', '0')
    MC_expr_dict['PDF_max'] = MC_holding_dict.get(f'{prefix}_PDFMAX-', '0')
    MC_expr_dict['PDF_mode'] = MC_holding_dict.get(f'{prefix}_PDFMOD-', '0')
    MC_expr_dict['PDF_mean'] = MC_holding_dict.get(f'{prefix}_PDFMEA-', '0')
    MC_expr_dict['PDF_stdev'] = MC_holding_dict.get(f'{prefix}_PDFSIG-', '0')
    MC_expr_dict['PDF_lambda'] = MC_holding_dict.get(f'{prefix}_PDFRAT-', '0')
    MC_expr_dict['PDF_shape'] = MC_holding_dict.get(f'{prefix}_PDFSHA-', '0')
    MC_expr_dict['PDF_median'] = MC_holding_dict.get(f'{prefix}_PDFMED-', '0')
    MC_expr_dict['PDF_xs'] = list(MC_holding_dict.get(f'{prefix}_PDFX-', '0').split('\n'))
    MC_expr_dict['PDF_ys'] = list(MC_holding_dict.get(f'{prefix}_PDFY-', '0').split('\n'))
    #MC_expr_dict['PDF_xs'] = np.array(list(map(float, MC_holding_dict.get(f'{prefix}_PDFX-', '0').split('\n'))))
    #MC_expr_dict['PDF_ys'] = np.array(list(map(float, MC_holding_dict.get(f'{prefix}_PDFY-', '0').split('\n'))))

    for v in ['PDF_min', 'PDF_max', 'PDF_mode', 'PDF_mean', 'PDF_stdev', 'PDF_lambda',
              'PDF_shape', 'PDF_median']:
        try:
            MC_expr_dict[v] = float(MC_expr_dict[v])
        except ValueError:
            MC_expr_dict[v] = parameter_dict[MC_expr_dict[v]]['expression']

    par_kink_flag = False
    for v in ['PDF_xs', 'PDF_ys']:
        try:
            MC_expr_dict[v] = np.array(list(map(float, MC_expr_dict[v])))
        except ValueError:
            par_kink_flag = True
            PDF_points_arr = np.zeros((len(MC_expr_dict[v]), n_simulation_periods))
            for p in MC_expr_dict[v]:
                try:
                    PDF_points_arr[p, :] = float(p)
                except ValueError:
                    PDF_points_arr[p, :] = parameter_dict[p]['expression']
            MC_expr_dict[v] = PDF_points_arr



    if PDF_func == 'None':
        return np.ones(dims)
    if PDF_func == 'Uniform':
        return uniform.rvs(loc=MC_expr_dict['PDF_min'],
                           scale=MC_expr_dict['PDF_max'] - MC_expr_dict['PDF_min'],
                           size=dims,
                           random_state=random_generator)
    if PDF_func == 'Bounded normal-like':
        bnl_a = uniform.rvs(loc=MC_expr_dict['PDF_min'],
                           scale=MC_expr_dict['PDF_max'] - MC_expr_dict['PDF_min'],
                           size=dims,
                           random_state=random_generator)
        bnl_b = uniform.rvs(loc=MC_expr_dict['PDF_min'],
                           scale=MC_expr_dict['PDF_max'] - MC_expr_dict['PDF_min'],
                           size=dims,
                           random_state=random_generator)
        bnl_c = uniform.rvs(loc=MC_expr_dict['PDF_min'],
                           scale=MC_expr_dict['PDF_max'] - MC_expr_dict['PDF_min'],
                           size=dims,
                           random_state=random_generator)
        return (bnl_a + bnl_b + bnl_c) / 3
    if PDF_func == 'Triangular':
        return triang.rvs(c=(MC_expr_dict['PDF_mode'] - MC_expr_dict['PDF_min'])/(MC_expr_dict['PDF_max'] - MC_expr_dict['PDF_min']),
                          loc=MC_expr_dict['PDF_min'],
                          scale=MC_expr_dict['PDF_max'] - MC_expr_dict['PDF_min'],
                          size=dims,
                          random_state=random_generator)
    if PDF_func == 'PERT':
        r = MC_expr_dict['PDF_max'] - MC_expr_dict['PDF_min']
        a = 1 + 4 * (MC_expr_dict['PDF_mode'] - MC_expr_dict['PDF_min']) / r
        b = 1 + 4 * (MC_expr_dict['PDF_max'] - MC_expr_dict['PDF_mode']) / r
        return beta.rvs(a=a, b=b, scale=r, loc=MC_expr_dict['PDF_min'], size=dims,
                                                                      random_state=random_generator)
    if PDF_func == 'Folded normal':
        return foldnorm.rvs(
            c=MC_expr_dict['PDF_mean'] / MC_expr_dict['PDF_stdev'],
            loc=0,
            scale=MC_expr_dict['PDF_stdev'],
            size=dims,
            random_state=random_generator)
    if PDF_func == 'Truncated normal':
        return truncnorm.rvs(a=0, b=np.inf,
                             loc=MC_expr_dict['PDF_mean'],
                             scale=MC_expr_dict['PDF_stdev'],
                             size=dims,
                             random_state=random_generator)
    if PDF_func == 'Exponential':
        return expon.rvs(scale=(1 / MC_expr_dict['PDF_lambda']),
                         size=dims,
                         random_state=random_generator)
    if PDF_func == 'Log-logistic':
        return fisk.rvs(c=MC_expr_dict['PDF_shape'],
                        scale=MC_expr_dict['PDF_median'],
                        size=dims,
                        random_state=random_generator)
    if PDF_func == 'Log-normal':
        return lognorm.rvs(s=MC_expr_dict['PDF_stdev'],
                           scale=np.exp(MC_expr_dict['PDF_mean']),
                           size=dims,
                           random_state=random_generator)
    if PDF_func == 'Custom Kinked PDF':
        if par_kink_flag is False:
            xs = MC_expr_dict['PDF_xs']
            ys = MC_expr_dict['PDF_ys']
            set_custom_dfs(xs, ys)
            cust_kink = custom_kinks_gen(name='custom_kinks', a=np.min(xs), b=np.max(xs))
            return cust_kink.rvs(size=dims, random_state=random_generator)
        else:
            out_arr = np.zeros(dims)
            for t in range(0, n_simulation_periods):
                xs = MC_expr_dict['PDF_xs'][:, t]
                ys = MC_expr_dict['PDF_ys'][:, t]
                set_custom_dfs(xs, ys)
                cust_kink = custom_kinks_gen(name='custom_kinks', a=np.min(xs), b=np.max(xs))
                out_arr[:, t] = cust_kink.rvs(size=(n_simulations, 1), random_state=random_generator)[:, 0]
            return out_arr
    else:
        return None


def outcome_array_line(outcome_map, weights, noreps, maxreps):
    with np.nditer(outcome_map, flags=['f_index'], op_flags=['readwrite']) as it:
        iter_weights = weights
        for x in it:
            period = it.index
            if np.sum(iter_weights[:, period]) != 0 and x != 0:
                # Locate the minimum of the normalised weight matrix
                outn = np.clip(
                    (np.cumsum(iter_weights[:, period] / np.sum(iter_weights[:, period])) - x), a_min=0, a_max=2)
                out_num = np.argmin(np.ma.masked_where(outn == 0, outn)) + 1
                outcome_map[period] = out_num
                out_ind = out_num - 1
                # Test for non_reps and alter iter_weights as appropriate
                if noreps[out_ind] > 0:
                    filter_start = min(period + 1, n_simulation_periods - 1)
                    filter_end = min(filter_start + noreps[out_ind], n_simulation_periods - 1)
                    iter_weights[out_ind][filter_start:filter_end] = 0
                # Test for if the number of outcomes reaches max limit and alter iter_weights as appropriate
                if maxreps[out_ind] > 0 and maxreps[out_ind] == np.count_nonzero(outcome_map == x):
                    filter_start = period + 1
                    iter_weights[out_ind][filter_start:n_simulation_periods] = 0

    return outcome_map


def outcome_arrays(weights, noreps, maxreps, monte_carlo_sims, depends_mask):
    # return a dictionary of key = outcome and value = array of periods where outcome occurs
    outcome_map = monte_carlo_sims * depends_mask
    outcome_map_dict = {}

    for s in range(0, n_simulations):
        sim_weights = weights[:, s, :]
        outcome_array_line(outcome_map[s, :], weights=sim_weights, noreps=noreps, maxreps=maxreps)

    """np.apply_along_axis(outcome_array_line, 1, arr=outcome_map,
                        weights=weights, noreps=noreps, maxreps=maxreps)
    Needed to change the code to accommodate the possibility of different weights in each simulation."""

    for outs in range(1, len(weights) + 1):
        outcome_map_dict[outs] = outcome_map == outs

    return outcome_map_dict


def results_display_table(run, measure, weights, net_scn_z, distribution_segments=['All'], bcr_denom='All'):
    fallback_bcr = False

    display_table = run_results[run][['scenario', 'group_id', 'line_id', 'type', 'value', 'stakeholder',
                                      'geography', 'weight', 'discounted_total_value']].copy()

    # print(display_table.to_string())
    if distribution_segments != ['All']:
        display_table['stk_geo'] = list(zip(display_table['stakeholder'], display_table['geography']))
        display_table = display_table.loc[display_table['stk_geo'].isin(distribution_segments)]

    # Add a zero benefits and costs line to the display table for each scenario to assist with handling null cases
    for scn in range(0, n_scenarios + 1):
        display_table_null = pd.DataFrame(
            [[scn, '', '', 'Costs', '', '', '', 1.0, np.zeros(n_simulations)],
             [scn, '', '', 'Benefits', '', '', '', 1.0, np.zeros(n_simulations)]],
            columns=['scenario', 'group_id', 'line_id', 'type', 'value', 'stakeholder',
                     'geography', 'weight', 'discounted_total_value'])
        display_table = pd.concat([display_table, display_table_null])

    if net_scn_z == 'Yes':
        scn_z_frame = display_table.loc[display_table['scenario'] == 0].rename(
            columns={'discounted_total_value': 'discounted_total_value_0'})

        display_table = pd.merge(display_table, scn_z_frame[['group_id', 'line_id', 'discounted_total_value_0']],
                                 how='left', on=['group_id', 'line_id'])

        display_table['discounted_total_value'] = display_table['discounted_total_value'] - display_table[
            'discounted_total_value_0']

    if weights == 'Weighted':
        display_table['discounted_total_value'] = display_table['discounted_total_value'] * display_table['weight']

    display_table = display_table[['scenario', 'group_id', 'line_id', 'type', 'value', 'stakeholder',
                                   'geography', 'discounted_total_value']]

    if measure == 'Average':
        display_table['calc_total'] = display_table.discounted_total_value.apply(lambda x: np.average(x))
    else:
        # If the user selects a percentile to display, calculate and order npv for each scenario
        # Calculate the Raw Net Benefits for each scenario.
        raw_total_records = {}
        for scn in range(0, n_scenarios + 1):
            raw_benefits = display_table.loc[
                ((display_table['scenario'] == scn) & (display_table['type'] == 'Benefits'))]
            raw_costs = display_table.loc[((display_table['scenario'] == scn) & (display_table['type'] == 'Costs'))]
            raw_npv = raw_benefits.discounted_total_value.sum() - raw_costs.discounted_total_value.sum()
            raw_npv_argsort = np.argsort(raw_npv)
            raw_total_records[f'scn_{scn}'] = raw_npv_argsort
        display_table['disc_tot_val_argsort'] = display_table.apply(
            lambda row: np.take_along_axis(row['discounted_total_value'],
                                           raw_total_records[f'scn_{row["scenario"]}'],
                                           axis=0), axis=1)

        # Find the representative median in a 1% window around the percentile nominated by the user.
        measure_dict = {'P5': 5, 'P10': 10, 'P20': 20, 'P30': 30, 'P40': 40, 'P50': 50, 'P60': 60, 'P70': 70, 'P80': 80,
                        'P90': 90, 'P95': 95}
        percentile = measure_dict[measure]
        lower_index = int(np.around((n_simulations / 100) * (percentile - 1), 0))
        upper_index = int(np.around((n_simulations / 100) * (percentile + 1), 0))
        display_table['calc_total'] = display_table.disc_tot_val_argsort.apply(
            lambda x: np.median(x[lower_index:upper_index]))

    # Calculate Total Costs
    display_table_costs = display_table.loc[(((display_table['type'] == 'Costs') & (display_table['calc_total'] >= 0)) |
                                             ((display_table['type'] == 'Benefits') & (
                                                     display_table['calc_total'] < 0)))].copy()
    # print(display_table_costs.to_string())
    display_table_costs['calc_total'] = display_table_costs['calc_total'].abs()
    display_table_costs = pd.pivot_table(display_table_costs, values='calc_total', columns=['scenario'],
                                         index=['group_id', 'line_id'], aggfunc=np.sum)
    # print(display_table_costs.to_string())

    # Calculate Total Benefits
    display_table_benefits = display_table.loc[
        (((display_table['type'] == 'Benefits') & (display_table['calc_total'] >= 0)) |
         ((display_table['type'] == 'Costs') & (display_table['calc_total'] < 0)))].copy()
    display_table_benefits['calc_total'] = display_table_benefits['calc_total'].abs()

    display_table_benefits = pd.pivot_table(display_table_benefits, values='calc_total', columns=['scenario'],
                                            index=['group_id', 'line_id'], aggfunc=np.sum)

    # Calculate Net Benefits
    display_table_npv = pd.pivot_table(display_table, values='calc_total', columns=['scenario'],
                                       index=['type'], aggfunc=np.sum)
    display_table_npv.loc['Costs'] = display_table_npv.loc['Costs'] * -1

    display_table = pd.pivot_table(display_table, values='calc_total', columns=['scenario'],
                                   index=['type', 'group_id', 'line_id'], aggfunc=np.sum)
    display_table.sort_index(level=0, ascending=False, inplace=True)

    display_table.loc['TOTAL COSTS', :] = display_table_costs.sum().values
    display_table.loc['TOTAL BENEFITS', :] = display_table_benefits.sum().values
    display_table.loc['NPV', :] = display_table_npv.sum().values

    if bcr_denom == 'All':
        display_table.loc['BCR', :] = display_table.loc[['TOTAL BENEFITS']].values / display_table.loc[
            ['TOTAL COSTS']].values
        # Will give RuntimeWarning: invalid value encountered in divide when calculating Net of Scn0 due to div by 0
    else:
        try:
            display_table.loc['BCR*', :] = (display_table.loc[['NPV']].values + display_table.loc[
                ('Costs', bcr_denom,)].sum().values) / \
                                           display_table.loc[('Costs', bcr_denom,)].sum().values
        except KeyError:
            display_table.loc['BCR', :] = display_table.loc[['TOTAL BENEFITS']].values / display_table.loc[
                ['TOTAL COSTS']].values
            result_windows[run][f'-RUN{run_number}_BCR_DENOM-'].update(value='All')
            fallback_bcr = True
    # print(display_table.to_string())

    cba_results = display_table.round(3).to_string()
    cba_results = cba_results.split('\n')
    ml_width = max(map(len, cba_results))
    ml_height = len(cba_results)

    n_benefits = display_table.loc[['Benefits']].shape[0]

    blank_line = ' ' * ml_width
    dash_line = '_' * ml_width
    dot_line = '.' * ml_width

    cba_results.insert(ml_height - 2, dot_line)
    cba_results.insert(ml_height - 4, dash_line)
    cba_results.insert(ml_height - 4, blank_line)
    cba_results.insert(ml_height - 4 - n_benefits, blank_line)
    cba_results.insert(2, dash_line)
    cba_results.insert(1, dot_line)

    if bcr_denom != 'All' and fallback_bcr is False:
        cba_results.append(f'*BCR denominator is {bcr_denom}')

    ml_height = len(cba_results) + 1
    ml_width = len(cba_results[0])

    cba_results = '\n'.join(cba_results)
    print(cba_results)
    return cba_results, ml_height, ml_width


def sensitivity_display_table(run, weights, net_scn_z, distribution_segments=['All']):
    regression_table = run_results[run][['scenario', 'group_id', 'line_id', 'type', 'value', 'stakeholder',
                                      'geography', 'weight', 'discounted_total_value']].copy()

    # print(display_table.to_string())
    if distribution_segments != ['All']:
        regression_table['stk_geo'] = list(zip(regression_table['stakeholder'], regression_table['geography']))
        regression_table = regression_table.loc[regression_table['stk_geo'].isin(distribution_segments)]

    # Add a zero benefits and costs line to the display table for each scenario to assist with handling null cases
    for scn in range(0, n_scenarios + 1):
        display_table_null = pd.DataFrame(
            [[scn, '', '', 'Costs', '', '', '', 1.0, np.zeros(n_simulations)],
             [scn, '', '', 'Benefits', '', '', '', 1.0, np.zeros(n_simulations)]],
            columns=['scenario', 'group_id', 'line_id', 'type', 'value', 'stakeholder',
                     'geography', 'weight', 'discounted_total_value'])
        regression_table = pd.concat([regression_table, display_table_null])

    if net_scn_z == 'Yes':
        scn_z_frame = regression_table.loc[regression_table['scenario'] == 0].rename(
            columns={'discounted_total_value': 'discounted_total_value_0'})

        regression_table = pd.merge(regression_table, scn_z_frame[['group_id', 'line_id', 'discounted_total_value_0']],
                                 how='left', on=['group_id', 'line_id'])

        regression_table['discounted_total_value'] = regression_table['discounted_total_value'] - regression_table[
            'discounted_total_value_0']

    if weights == 'Weighted':
        regression_table['discounted_total_value'] = regression_table['discounted_total_value'] * regression_table['weight']

    regression_table = regression_table[['scenario', 'group_id', 'line_id', 'type', 'value', 'stakeholder',
                                   'geography', 'discounted_total_value']]

    # Calculate the Raw Net Benefits for each scenario.
    raw_total_records = {}
    for scn in range(0, n_scenarios + 1):
        raw_benefits = regression_table.loc[
            ((regression_table['scenario'] == scn) & (regression_table['type'] == 'Benefits'))]
        raw_costs = regression_table.loc[
            ((regression_table['scenario'] == scn) & (regression_table['type'] == 'Costs'))]
        raw_npv = raw_benefits.discounted_total_value.sum() - raw_costs.discounted_total_value.sum()
        raw_npv_argsort = np.argsort(raw_npv)
        raw_total_records[f'scn_{scn}'] = raw_npv_argsort
    regression_table['disc_tot_val_argsort'] = regression_table.apply(
        lambda row: np.take_along_axis(row['discounted_total_value'],
                                       raw_total_records[f'scn_{row["scenario"]}'],
                                       axis=0), axis=1)

    # Divide the discounted total value for each line item by its

    '''if measure == 'Average':
        regression_table['calc_total'] = regression_table.discounted_total_value.apply(lambda x: np.average(x))
    else:
        # If the user selects a percentile to display, calculate and order npv for each scenario
        # Calculate the Raw Net Benefits for each scenario.
        raw_total_records = {}
        for scn in range(0, n_scenarios + 1):
            raw_benefits = regression_table.loc[
                ((regression_table['scenario'] == scn) & (regression_table['type'] == 'Benefits'))]
            raw_costs = regression_table.loc[((regression_table['scenario'] == scn) & (regression_table['type'] == 'Costs'))]
            raw_npv = raw_benefits.discounted_total_value.sum() - raw_costs.discounted_total_value.sum()
            raw_npv_argsort = np.argsort(raw_npv)
            raw_total_records[f'scn_{scn}'] = raw_npv_argsort
        regression_table['disc_tot_val_argsort'] = regression_table.apply(
            lambda row: np.take_along_axis(row['discounted_total_value'],
                                           raw_total_records[f'scn_{row["scenario"]}'],
                                           axis=0), axis=1)

        # Find the representative median in a 1% window around the percentile nominated by the user.
        measure_dict = {'P5': 5, 'P10': 10, 'P20': 20, 'P30': 30, 'P40': 40, 'P50': 50, 'P60': 60, 'P70': 70, 'P80': 80,
                        'P90': 90, 'P95': 95}
        percentile = measure_dict[measure]
        lower_index = int(np.around((n_simulations / 100) * (percentile - 1), 0))
        upper_index = int(np.around((n_simulations / 100) * (percentile + 1), 0))
        regression_table['calc_total'] = regression_table.disc_tot_val_argsort.apply(
            lambda x: np.median(x[lower_index:upper_index]))'''

    # Calculate Total Costs
    regression_table_costs = regression_table.loc[(((regression_table['type'] == 'Costs') & (regression_table['calc_total'] >= 0)) |
                                             ((regression_table['type'] == 'Benefits') & (
                                                     regression_table['calc_total'] < 0)))].copy()
    # print(display_table_costs.to_string())
    regression_table_costs['calc_total'] = regression_table_costs['calc_total'].abs()
    regression_table_costs = pd.pivot_table(regression_table_costs, values='calc_total', columns=['scenario'],
                                         index=['group_id', 'line_id'], aggfunc=np.sum)
    # print(display_table_costs.to_string())

    # Calculate Total Benefits
    regression_table_benefits = regression_table.loc[
        (((regression_table['type'] == 'Benefits') & (regression_table['calc_total'] >= 0)) |
         ((regression_table['type'] == 'Costs') & (regression_table['calc_total'] < 0)))].copy()
    regression_table_benefits['calc_total'] = regression_table_benefits['calc_total'].abs()

    regression_table_benefits = pd.pivot_table(regression_table_benefits, values='calc_total', columns=['scenario'],
                                            index=['group_id', 'line_id'], aggfunc=np.sum)

    # Calculate Net Benefits
    regression_table_npv = pd.pivot_table(regression_table, values='calc_total', columns=['scenario'],
                                       index=['type'], aggfunc=np.sum)
    regression_table_npv.loc['Costs'] = regression_table_npv.loc['Costs'] * -1

    regression_table = pd.pivot_table(regression_table, values='calc_total', columns=['scenario'],
                                   index=['type', 'group_id', 'line_id'], aggfunc=np.sum)
    regression_table.sort_index(level=0, ascending=False, inplace=True)

    regression_table.loc['TOTAL COSTS', :] = regression_table_costs.sum().values
    regression_table.loc['TOTAL BENEFITS', :] = regression_table_benefits.sum().values
    regression_table.loc['NPV', :] = regression_table_npv.sum().values

    '''if bcr_denom == 'All':
        display_table.loc['BCR', :] = display_table.loc[['TOTAL BENEFITS']].values / display_table.loc[
            ['TOTAL COSTS']].values
        # Will give RuntimeWarning: invalid value encountered in divide when calculating Net of Scn0 due to div by 0
    else:
        try:
            display_table.loc['BCR*', :] = (display_table.loc[['NPV']].values + display_table.loc[
                ('Costs', bcr_denom,)].sum().values) / \
                                           display_table.loc[('Costs', bcr_denom,)].sum().values
        except KeyError:
            display_table.loc['BCR', :] = display_table.loc[['TOTAL BENEFITS']].values / display_table.loc[
                ['TOTAL COSTS']].values
            result_windows[run][f'-RUN{run_number}_BCR_DENOM-'].update(value='All')
            fallback_bcr = True'''
    # print(display_table.to_string())

    cba_results = regression_table.round(3).to_string()
    cba_results = cba_results.split('\n')
    ml_width = max(map(len, cba_results))
    ml_height = len(cba_results)

    n_benefits = regression_table.loc[['Benefits']].shape[0]

    blank_line = ' ' * ml_width
    dash_line = '_' * ml_width
    dot_line = '.' * ml_width

    cba_results.insert(ml_height - 2, dot_line)
    cba_results.insert(ml_height - 4, dash_line)
    cba_results.insert(ml_height - 4, blank_line)
    cba_results.insert(ml_height - 4 - n_benefits, blank_line)
    cba_results.insert(2, dash_line)
    cba_results.insert(1, dot_line)

    if bcr_denom != 'All' and fallback_bcr is False:
        cba_results.append(f'*BCR denominator is {bcr_denom}')

    ml_height = len(cba_results) + 1
    ml_width = len(cba_results[0])

    cba_results = '\n'.join(cba_results)
    print(cba_results)
    return cba_results


def plot_histo_cdf(run_number, scenario, weights, net_scn_z, distribution_segments):
    global result_graphs

    plot_data = run_results[run_number][
        ['scenario', 'group_id', 'line_id', 'type', 'stakeholder', 'geography', 'weight', 'discounted_total_value']].copy()

    if distribution_segments != ['All']:
        plot_data['stk_geo'] = list(zip(plot_data['stakeholder'], plot_data['geography']))
        plot_data = plot_data.loc[plot_data['stk_geo'].isin(distribution_segments)]

    # Add a zero benefits and costs line to the display table for each scenario to assist with handling null cases
    for scn in range(0, n_scenarios + 1):
        plot_data_null = pd.DataFrame(
            [[scn, '', '', 'Costs', '', '', 1, np.zeros(n_simulations)],
             [scn, '', '', 'Benefits', '', '', 1, np.zeros(n_simulations)]],
            columns=['scenario', 'group_id', 'line_id', 'type', 'stakeholder',
                     'geography', 'weight', 'discounted_total_value'])
        plot_data = pd.concat([plot_data, plot_data_null])
        #print(plot_data.to_string())
    if net_scn_z == 'Yes':
        scn_z_frame = plot_data.loc[plot_data['scenario'] == 0].rename(
            columns={'discounted_total_value': 'discounted_total_value_0'})
        plot_data = pd.merge(plot_data, scn_z_frame[['group_id', 'line_id', 'discounted_total_value_0']],
                             how='left', on=['group_id', 'line_id'])

        plot_data['discounted_total_value'] = plot_data['discounted_total_value'] - plot_data[
            'discounted_total_value_0']

    if weights == 'Weighted':
        plot_data['discounted_total_value'] = plot_data['discounted_total_value'] * plot_data['weight']
        print(plot_data.to_string())
    plot_data = plot_data[['scenario', 'group_id', 'line_id', 'type', 'stakeholder',
                           'geography', 'discounted_total_value']]

    plot_data = plot_data.loc[plot_data['scenario'] == scenario]

    plot_data = plot_data[['type', 'discounted_total_value']]

    plot_data = pd.pivot_table(plot_data, values='discounted_total_value', columns=['type'], aggfunc=np.sum)

    # Net Benefit
    plot_data['Net_Present_Value'] = plot_data['Benefits'] - plot_data['Costs']

    plot_values = plot_data.Net_Present_Value.to_numpy()[0]

    # Check if plot already exists, initialise or clear as necessary.
    if f'fig_{run_number}' in result_graphs.keys():
        result_graphs[f'ax_{run_number}'].cla()
        result_graphs[f'ax2_{run_number}'].cla()
    else:
        result_graphs[f'fig_{run_number}'] = Figure(figsize=(5, 4), dpi=100)
        result_graphs[f'ax_{run_number}'] = result_graphs[f'fig_{run_number}'].add_subplot()
        result_graphs[f'ax2_{run_number}'] = result_graphs[f'ax_{run_number}'].twinx()
        result_graphs[f'canvas_{run_number}'] = Canvas(result_graphs[f'fig_{run_number}'],
                                                       result_windows[run_number][f'-RUN{run_number}_GRAPH-'].Widget)

    # Create Plot
    result_graphs[f'ax_{run_number}'].set_title(f'Histogram and CDF - Scenario {scenario}')
    result_graphs[f'ax_{run_number}'].set_ylabel('Histogram Counts')
    result_graphs[f'ax2_{run_number}'].set_ylabel('CDF Density')
    result_graphs[f'ax_{run_number}'].set_xlabel('Net Benefits')

    result_graphs[f'ax_{run_number}'].hist(plot_values, bins='auto', histtype='stepfilled', label='Histogram Counts',
                                           color='coral')
    result_graphs[f'ax2_{run_number}'].hist(plot_values, bins='auto', density=True, cumulative=True, histtype='step',
                                            label='CDF Density',
                                            color='saddlebrown')

    result_graphs[f'canvas_{run_number}'].draw()


run_cba_settings = {}
run_results = {}
run_number = 0
result_windows = {}
result_graphs = {}


def result_popup(run_number):
    global run_results
    global run_cba_settings
    run_results[run_number] = quantity_stream_table
    cost_groups = run_results[run_number].loc[run_results[run_number]['type'] == 'Costs']['group_id'].to_list()
    cost_groups = ['All'] + [*set(cost_groups)]

    da_radio_layout = sg.Text('')
    if values['-DA_STKGRP-'] != '' and values['-DA_GEOGRP-'] != '':
        ROWS = len(DA_table)
        COLS = len(DA_table[1])

        da_radio_layout_C0 = [
            [sg.Text(DA_table[i][0])] for i in range(ROWS)
        ]

        da_radio_layout_CN = [[]]
        for j in range(1, COLS):
            col = [[sg.Text(DA_table[0][j])]] + [
                [sg.Checkbox(default=True, text='', key=f'-RUN{run_number}_DA_SELECT{i}_{j}-', enable_events=True)] for
                i in
                range(1, ROWS)]
            da_radio_layout_CN[0] += [sg.Column(col)]

        da_radio_layout = sg.Frame(title='Distribution Analysis', vertical_alignment='top', layout=[
            [sg.Text('Display '),
             sg.Combo(values=['Unweighted', 'Weighted'], default_value='Unweighted', auto_size_text=True,
                      key=f'-RUN{run_number}_WEIGHT_USE-', enable_events=True),
             sg.Text(' results')],
            [sg.Column(da_radio_layout_C0), sg.Column(da_radio_layout_CN)]
        ])

    run_cba_settings[run_number] = {'measure': 'Average', 'use_weights': 'Unweighted', 'net_z': 'No',
                                    'distr_segs': ['All'],
                                    'bcr_denom': 'All'}

    cba_results, ml_height, ml_width = results_display_table(run_number,
                                                             run_cba_settings[run_number]['measure'],
                                                             run_cba_settings[run_number]['use_weights'],
                                                             run_cba_settings[run_number]['net_z'],
                                                             run_cba_settings[run_number]['distr_segs'],
                                                             run_cba_settings[run_number]['bcr_denom'])

    # Sensitivity analysis for a future version
    '''sensitivity_results = sensitivity_display_table(run_number,
                                                             run_cba_settings[run_number]['use_weights'],
                                                             run_cba_settings[run_number]['net_z'],
                                                             run_cba_settings[run_number]['distr_segs'])'''

    results_layout = [
        [sg.Frame(title='Initialisation', vertical_alignment='top', layout=[
            [sg.Text(f'Date/Time: {dt.datetime.now()}', key=f'-RUN{run_number}_DATE-')],
            [sg.Text(f'Number of Simulations: {n_simulations}', key=f'-RUN{run_number}_N_SIMS-')],
            [sg.Text(f'Number of Periods: {n_simulation_periods}', key=f'-RUN{run_number}_N_PERIODS-')],
            [sg.Text('Scenarios 1+ net of Scenario 0?'), sg.Combo(values=['Yes', 'No'], default_value='No',
                                                                  key=f'-RUN{run_number}_NET_SCN0-',
                                                                  enable_events=True)],
            [sg.Text('Monte Carlo Results to Display:'), sg.Combo(values=['Average', 'P5', 'P10', 'P20', 'P30', 'P40',
                                                                          'P50', 'P60', 'P70', 'P80', 'P90', 'P95'],
                                                                  default_value='Average', auto_size_text=True,
                                                                  key=f'-RUN{run_number}_MEASURE-',
                                                                  enable_events=True)],
            [sg.Text('BCR Denominator:'), sg.Combo(values=cost_groups, default_value='All', auto_size_text=True,
                                                   key=f'-RUN{run_number}_BCR_DENOM-', enable_events=True)]
        ]), da_radio_layout],
        [sg.Frame(title='CBA RESULTS', vertical_alignment='top', layout=[
            [sg.Multiline(default_text=cba_results,
                          key=f'-RUN{run_number}_CBA_TABLE_DISPLAY-', justification='left', font=('Courier New', 11),
                          write_only=True, no_scrollbar=True, size=(ml_width, ml_height))],
            [sg.Button(button_text='Export Simulation Data', key=f'-RUN{run_number}_EXPORT_SIMS-')]
        ]),
         sg.Frame(title='', size=(640, 480), vertical_alignment='top', layout=[
             [sg.Text('Scenario to graph:'),
              sg.Combo(values=list(range(0, n_scenarios + 1)), default_value=0, key=f'-RUN{run_number}_SCN_GRAPH-',
                       enable_events=True)],
             [sg.Canvas(key=f'-RUN{run_number}_GRAPH-', expand_x=True, expand_y=True)],
             [sg.Button(button_text='Save Graph', key=f'-RUN{run_number}_SAVE_GRAPH-')]
         ])]
    ]

    # Sensitivity code for a future version.
    '''sensitivity_layout = [
        [sg.Frame(title='Description', vertical_alignment=top, layout=[
            [sg.Text('The sensitivity analysis performed by RUBICS follows from Frey, C. H., & Patil, S. R. (2002), who identify Regression Analysis as a method of analysing the responsiveness of dependent variables (here, NPV) to changes in indepedent variables.')],
            [sg.Text('RUBICS first standardises the NPV and all line items in the CBA by dividing them by their standard deviation. It then performs a linear regression, predicting NPV on all line item regressors, such that the coefficients of each line item indicate the degree to which a 1-unit movement in the standard deviation of a line item results in a change in the standard deviation of the NPV.')],
            [sg.Text('The magnitude (absolute value), of the coefficients can then be directly compared to determine which line items bear the greatest responsibility for variation in the NPV.')],
            [sg.Text('Frey, C. H., & Patil, S. R. (2002). Identification and Review of Sensitivity Analysis Methods. Risk Analysis, 22(3), 553–578. https://doi.org/10.1111/0272-4332.00039')]
        ])],
        [sg.Frame(title='Regression results', layout=[
            [sg.Multiline(default_text=sensitivity_results,
                          key=f'-RUN{run_number}_SENS_DISPLAY-', justification='left', font=('Courier New', 11),
                          write_only=True, no_scrollbar=True, size=(ml_width, ml_height)
                          )]

    ])]
    ]'''

    user_settings_text = json_settings()
    user_settings_readout = [
        [sg.Multiline(key=f'-RUN{run_number}_SETTINGS_READOUT-', size=(70, 50), disabled=True,
                      default_text=user_settings_text)],
        [sg.Button(button_text='Save Settings as .json', key=f'-RUN{run_number}_SAVE_SETS-')]
    ]

    # Including the sensitivity analysis section is a task for another time.
    results_window_layout = [[sg.TabGroup([[
        sg.Tab(title='CBA results', layout=results_layout),
        #sg.Tab(title='Sensitivity analysis', layout=sensitivity_layout),
        sg.Tab(title='Settings readout', layout=user_settings_readout)
    ]])]]

    return sg.Window(f'CBA Results {run_number}', results_window_layout, finalize=True,
                     enable_close_attempted_event=True, resizable=True)


def loop_result_window_events(run):
    run_event, run_values = result_windows[run].read(timeout=100)
    if run_event != '__TIMEOUT__':
        print(run_event)
    if run_event == '-WINDOW CLOSE ATTEMPTED-':
        result_windows[run].close()
        result_windows[run] = None

    if run_event == f'-RUN{run}_SAVE_GRAPH-':
        try:
            result_graphs[f'fig_{run}'].savefig(fname=sg.popup_get_file(message='Filepath to save as:',
                                                                        file_types=(('PNG', '*.png'), ('PDF', '*.pdf'),
                                                                                    ('SVG', '*.svg'),
                                                                                    ('TIFF', '*.tiff'),
                                                                                    ('JPEG', '*,jpeg'),
                                                                                    ('JPG', '*.jpg')),
                                                                        save_as=True), dpi=600)
        except Exception as e:
            sg.popup_quick_message(f'ERROR:\n{e}')

    if run_event == f'-RUN{run}_SAVE_SETS-':
        try:
            file_to_save = open(sg.popup_get_file(message='Filepath to save as:',
                                                  file_types=(('JSON', '*.json'),),
                                                  save_as=True),
                                'w')
            file_to_save.write(run_values[f'-RUN{run}_SETTINGS_READOUT-'])
        except Exception as e:
            sg.popup_quick_message(f'ERROR:\n{e}')

    if run_event == f'-RUN{run}_EXPORT_SIMS-':
        # Add simulation run numbers to table and then explode to separate rows.
        try:
            run_results[run]['sim_n'] = run_results[run].group.apply(lambda x: np.arange(0, n_simulations, 1))
            run_results[run].explode(column=['line_monte_carlo_sims',
                                             'group_monte_carlo_sims',
                                             'monte_carlo_streams',
                                             'outcome_monte_carlo_sims',
                                             'combined_stream_quantity_events',
                                             'monte_carlo_values',
                                             'nominal_quantity_value_stream',
                                             'discounted_quantity_value_stream',
                                             'nominal_total_value',
                                             'discounted_total_value',
                                             'sim_n']).to_csv(
                path_or_buf=sg.popup_get_file(message='Filepath to save as:',
                                              file_types=(('CSV', '*.csv'),),
                                              default_extension='*.csv',
                                              save_as=True))
        except Exception as e:
            sg.popup_quick_message(f'ERROR:\n{e}')

    if run_event in [f'-RUN{run}_MEASURE-', f'-RUN{run}_WEIGHT_USE-', f'-RUN{run}_NET_SCN0-',
                     f'-RUN{run}_SCN_GRAPH-', f'-RUN{run}_BCR_DENOM-'] or any(
        [re.match(r'-RUN\d*_DA_SELECT\d*_\d*-', run_event)]):
        run_cba_settings[run]['measure'] = run_values[f'-RUN{run}_MEASURE-']
        run_cba_settings[run]['net_z'] = run_values[f'-RUN{run}_NET_SCN0-']
        run_cba_settings[run]['bcr_denom'] = run_values[f'-RUN{run}_BCR_DENOM-']

        if len(DA_table) > 0:
            run_cba_settings[run]['use_weights'] = run_values[f'-RUN{run}_WEIGHT_USE-']
            run_cba_settings[run]['distr_segs'] = []
            for stk in range(1, len(DA_table)):
                for geo in range(1, len(DA_table[1])):
                    if run_values[f'-RUN{run}_DA_SELECT{stk}_{geo}-'] is True:
                        run_cba_settings[run]['distr_segs'].append((stakeholder_list[stk - 1], geozones_list[geo - 1]))

        result_windows[run][f'-RUN{run}_CBA_TABLE_DISPLAY-'].update(results_display_table(run,
                                                                                          run_cba_settings[run][
                                                                                              'measure'],
                                                                                          run_cba_settings[run][
                                                                                              'use_weights'],
                                                                                          run_cba_settings[run][
                                                                                              'net_z'],
                                                                                          run_cba_settings[run][
                                                                                              'distr_segs'],
                                                                                          run_cba_settings[run][
                                                                                              'bcr_denom'])[0])

        plot_histo_cdf(run,
                       int(run_values[f'-RUN{run}_SCN_GRAPH-']),
                       run_cba_settings[run]['use_weights'],
                       run_cba_settings[run]['net_z'],
                       run_cba_settings[run]['distr_segs'])


######################################################################################################################
# SECTION 3: SUB-SECTION LAYOUTS
######################################################################################################################

### DISCOUNT RATE SECTION --------------------------------------------------------------------------------------------
dr_layout = [
    # Load and save settings menu.
    [sg.Menu([['&File',
               ['&Load Settings', ['&From .json', '&From .xlsx'],
                '&Save Settings']],
              ['&Presets',
               ['&Update CPI from WorldBank', '&Update Exchange Rates from WorldBank']]
              ])],

    # Get discount rate method.
    [sg.Text('Discounting method'),
     sg.Combo(key='-DR_TYPE-', values=['Constant', 'Stepped', 'Gamma time declining'], default_value='Constant',
              enable_events=True),
     sg.VerticalSeparator(),
     sg.Column([
         [sg.pin(sg.Text('Discount Rate:', key='-DR_MAINTXT-', visible=True)),
          sg.pin(sg.Input(key='-DR_MAIN-', visible=True, enable_events=True, size=10, default_text='0.0'))],
         [sg.pin(sg.Text('Discount Std. Dev:', key='-DR_SIGTXT-', visible=False)),
          sg.pin(sg.Input(key='-DR_SIG-', visible=False, enable_events=True, size=10))]
     ]),
     sg.Column([
         [sg.pin(sg.Text('Step Range:', key='-DR_STEP_RANGETXT-', visible=False))],
         [sg.pin(sg.Multiline(key='-DR_STEP_RANGE-', visible=False, size=(10, 4)))]]),
     sg.Column([
         [sg.pin(sg.Text('Discount Rate:', key='-DR_STEP_RATETXT-', visible=False))],
         [sg.pin(sg.Multiline(key='-DR_STEP_RATE-', visible=False, size=(10, 4)))]]),
     sg.Column([
         [sg.pin(sg.Text('All periods thereafter:', key='-DR_STEP_BASETXT-', visible=False))],
         [sg.pin(sg.Input(key='-DR_STEP_BASE-', visible=False, size=10, default_text='0.0'))]]),
     ],

    # Get the discount rate probability distribution settings.
    [sg.Text('Probability\nDistribution\nFunction:'),
     sg.Combo(key='-DR_MC_TYPE-', values=['None', 'Uniform', 'Bounded normal-like', 'Triangular',
                                          'PERT', 'Folded normal', 'Truncated normal',
                                          'Exponential',
                                          'Log-logistic',
                                          'Log-normal', 'Custom Kinked PDF'], enable_events=True,
              default_value='None',
              size=15),
     sg.pin(sg.Text(key=f'-DR_PDFSETTXT-', text='PDF settings:', visible=False)),
     sg.pin(sg.Text(key=f'-DR_PDFSET-', text='', relief="groove", border_width=1, visible=False))],
    [sg.Text('Comments/\nMetadata:'),
     sg.Multiline(key=f'-DR_CM-', size=(40, 4))]
]

### DISTRIBUTION ANALYSIS SECTION -------------------------------------------------------------------------------------
da_layout = [
    # Ask for a list of stakeholder groups & geographic zones
    [sg.Frame('', layout=[[sg.Text('List Stakeholder Groups')],
                          [sg.Multiline(size=(20, 4), key='-DA_STKGRP-', enable_events=True)]])],
    [sg.Frame('', layout=[[sg.Text('List Geographic Zones')],
                          [sg.Multiline(size=(20, 4), key='-DA_GEOGRP-', enable_events=True)]])],
    [sg.HorizontalSeparator()],
    # Display the Stakeholder Weight Matrix
    [sg.Text('Weight Matrix')],
    [sg.Text('Enter weights or click calculate from income levels to generate weights.')],
    [sg.pin(sg.Text(text='', key='-DA_TABLE_DISPLAY-', visible=False, justification='right', font=('Courier New', 11),
                    relief="groove", border_width=1))],
    [sg.Button('Input Basic Weights', key='-DA_INPUT WEIGHTS_BASIC-', disabled=True)],
    [sg.Button('Input Income Weights', key='-DA_INPUT WEIGHTS_INCOME-', disabled=True)]
]

### FUNCTIONS SECTION -------------------------------------------------------------------------------------------------
fnc_layout = [
    [sg.Text('This section allows you to interpolate a continuous 1d function from discrete data points.')],
    [sg.Text('Functions for which the user already has a parametised form can be used directly in the Parameter section.')],
    [sg.Text('The interpolation options here are intended to fit exactly through known x,y points. If attempting to fit noisy \ndata it is recommended to explicitly parametise the relationship eg using regression methods, and apply the \ncoefficients to the data in the Parameters section.')],
    [sg.Text('The Smoothed cubic spline is provided to assist the analyst to model non-linear relationships, however the\nbehaviour of the function is highly senstitive to the choice of smoothing function.')],
    [sg.Button('Add Function')],
    [sg.Column(key='-FNC_COLS-', size=(1500, 280), scrollable=True, vertical_scroll_only=True,
               layout=[
                   [sg.Frame(title='Functions', key='-FUNCTIONS-',  layout=[])]
               ])]
]

### PARAMETER VALUES SECTION ------------------------------------------------------------------------------------------
par_layout = [
    [sg.Button('Add Parameter')],
    [sg.Column(key='-PAR_COLS-', size=(1500, 280), scrollable=True, vertical_scroll_only=True,
               layout=[
                   [sg.Frame(title='Parameters', key='-PARAMETERS-', layout=[])]
               ])]
]

### REFERENCE PRICE SECTION -------------------------------------------------------------------------------------------
pr_layout = [
    [sg.Frame('Analysis Settings', [
        [sg.Text('Select Country'),
         sg.Combo(key='-PR_COUNTRY-', values=country_names, default_value='Australia', enable_events=True)],
        [sg.Text('Select Year of Analysis'), sg.Combo(key='-PR_YEAR-', values=list(range(current_year, 2000, -1)),
                                                      default_value=current_year, enable_events=True)],
        [sg.Text(key='-CPI_NUMRTXT-',
                 text=f'The CPI conversion numerator is {np.around(cpi_numr, 3)} based on most recent data for year {cpi_numr_yr}')],
        [sg.Text('Currency Conversion Range:'),
         sg.Combo(key='-PR_CUR_RANGE-', values=['1 year', '2 years', '5 years', '10 years', '20 years', 'All dates'],
                  default_value='5 years', enable_events=True)],
        [sg.Text('Currency Conversion Measure:'),
         sg.Combo(key='-PR_CUR_POINT-', values=['Median', 'Mean'], default_value='Median', enable_events=True)],
        [sg.Text(key='-EXCHR_TXT-',
                 text=f'The benchmark exchange rate is {econ_codes._get_value("Australia", "ISO3")}{np.around(exchr_home, 3)}/USD.')]
    ])],
    [sg.Button('Add New Group', key='-ADD_PRICE_GROUP-')],
    [sg.Column(key='-PG_COLS-', size=(1500, 280), scrollable=True, vertical_scroll_only=True,
               layout=[[sg.Frame(title='', key='-PRICE_GROUPS-', layout=[])]]
               )],
    [sg.Button('Trigger Refresh', visible=False)]
]

### QUANTITY SCENARIOS SECTION ----------------------------------------------------------------------------------------
qnt_layout = [
    [sg.Button('Add Scenario'), sg.Button('Event Model Builder')],
    [sg.Column(key='-QS_COLS-', size=(1500, 500), scrollable=True, vertical_scroll_only=True,
               layout=[
                   [sg.TabGroup(layout=[], key='-QNT_TABGROUP-')]
               ])]
]

### RESULTS SECTION --------------------------------------------------------------------------------------------------
res_layout = [
    [sg.TabGroup(layout=[
        [sg.Tab(title='Monte Carlo\nInitialisation', layout=[
            [sg.Text('Set Seed:'),
             # Suggest a random seed chosen from the milliseconds of current time at program open.
             sg.Input(key='-RES_SET_SEED-', enable_events=True, size=10,
                      default_text=dt.datetime.now().strftime('%f'))],
            [sg.Text('Number of Simulations:'),
             sg.Input(key='-RES_NUM_SIMS-', enable_events=True, size=10, default_text=50000)],
            [sg.Button('Run Monte Carlo Analysis', key='-RES_RUN-')]
        ])]
    ], key='-RES_TABGROUP-')]
]

######################################################################################################################
# SECTION 4: MAIN WINDOW LAYOUT
######################################################################################################################

layout = [
    [sg.TabGroup([[
        sg.Tab(title='Discount Rate', layout=dr_layout),
        sg.Tab(title='Distribution Analysis', layout=da_layout),
        sg.Tab(title='Functions', layout=fnc_layout),
        sg.Tab(title='Parameters', layout=par_layout),
        sg.Tab(title='Reference Prices', layout=pr_layout),
        sg.Tab(title='Quantity Scenarios', layout=qnt_layout),
        sg.Tab(title='Results', layout=res_layout)
    ]])]
]

# Window
window = sg.Window('RUBICS', layout, finalize=True, return_keyboard_events=True, resizable=True, font=('Helvetica', 10),
                   icon=tbar_icon)

# Initiate the reference price section.
if pr_group_dict[1] == 0:
    add_price_group(n_pr_groups)
    n_pr_groups = len(pr_group_dict)
    add_price_line(1, 'Australia', current_year)
    pr_group_dict[1] += 1

# Initiate the Quantity scenarios section with the Base case and First Scenario
if n_scenarios == 0:
    add_scenario_tab()
    window[f'-QNT_SCENARIO_0-'].update('Base Case')
    n_scenarios += 1
    add_scenario_tab()

######################################################################################################################
# SECTION 5: WHILE EVENT LOOP
######################################################################################################################

while True:
    event, values = window.read(timeout=100)

    ### LOAD/SAVE SETTINGS -------------------------------------------------------------------------------------------
    # If settings are loaded, run the load_settings script.
    if event == 'From .json':
        user_settings_file = sg.popup_get_file('Select settings file to open')
        load_settings_json(user_settings_file)  # return a dictionary called user_setting
        print(MC_update_display)
        print(MC_holding_dict)
        for m in MC_update_display:
            pdf_window_update(m, MC_holding_dict)

    if event == 'From .xlsx':
        user_settings_file = sg.popup_get_file('Select settings file to open')
        load_settings_xlsx(user_settings_file)
        for m in MC_update_display:
            pdf_window_update(m, MC_holding_dict)

    if event == 'Save Settings':
        file_to_save = open(sg.popup_get_file(message='Filepath to save as:',
                                              file_types=(('JSON', '*.json'),),
                                              save_as=True),
                            'w')
        file_to_save.write(json_settings())


    ### WORLD BANK DATA SETTINGS --------------------------------------------------------------------------------------
    if event == 'Update CPI from WorldBank':
        api = f'http://api.worldbank.org/v2/country/all/indicator/FP.CPI.TOTL?date=1960:{current_year + 1}&format=json&per_page=20000'
        api_req = requests.get(api)
        if api_req.status_code == 200:
            raw_json = api_req.json()
            json_table = pd.json_normalize(raw_json[1])
            json_table = json_table.loc[json_table['countryiso3code'] != np.NaN]
            json_table = json_table.set_index(['countryiso3code', 'date'], append=True)
            json_table = json_table.pivot_table(index='countryiso3code', columns='date', values='value').rename_axis(
                'economy')
            json_table.to_csv(os.getcwd() + '/cpi.csv', index='economy')
            cpi_table = pd.read_csv(os.getcwd() + '/cpi.csv', index_col='economy')
        else:
            sg.popup_quick_message("The API request failed. This may be because the World Bank API is offline.")

    if event == 'Update Exchange Rates from WorldBank':
        api = f'http://api.worldbank.org/v2/country/all/indicator/DPANUSSPB?date=1987:{current_year+1}&format=json&per_page=10000'
        api_req = requests.get(api)
        if api_req.status_code == 200:
            raw_json = api_req.json()
            json_table = pd.json_normalize(raw_json[1]).pivot(index='country.id', columns='date',
                                                              values='value').rename_axis('economy')
            json_table.to_csv(os.getcwd() + '/exchr.csv', index='economy')
            exchr_table = pd.read_csv(os.getcwd() + '/exchr.csv', index_col='economy')
        else:
            sg.popup_quick_message("The API request failed. This may be because the World Bank API is offline.")


    ### DISCOUNT RATE EVENTS ------------------------------------------------------------------------------------------
    # If dropdown options are chosen, make fields visible based on input
    if event in ['-DR_TYPE-', 'Trigger Refresh']:
        window['-DR_MAINTXT-'].update(visible=False)
        window['-DR_MAIN-'].update(visible=False)
        window['-DR_SIGTXT-'].update(visible=False)
        window['-DR_SIG-'].update(visible=False)
        window['-DR_STEP_RANGETXT-'].update(visible=False)
        window['-DR_STEP_RANGE-'].update(visible=False)
        window['-DR_STEP_RATETXT-'].update(visible=False)
        window['-DR_STEP_RATE-'].update(visible=False)
        window['-DR_STEP_BASETXT-'].update(visible=False)
        window['-DR_STEP_BASE-'].update(visible=False)
        if values['-DR_TYPE-'] == 'Constant':
            window['-DR_MAINTXT-'].update(visible=True)
            window['-DR_MAIN-'].update(visible=True)
        if values['-DR_TYPE-'] == 'Stepped':
            window['-DR_STEP_RANGETXT-'].update(visible=True)
            window['-DR_STEP_RANGE-'].update(visible=True)
            window['-DR_STEP_RATETXT-'].update(visible=True)
            window['-DR_STEP_RATE-'].update(visible=True)
            window['-DR_STEP_BASETXT-'].update(visible=True)
            window['-DR_STEP_BASE-'].update(visible=True)
        if values['-DR_TYPE-'] == 'Gamma time declining':
            window['-DR_MAINTXT-'].update(visible=True)
            window['-DR_MAIN-'].update(visible=True)
            window['-DR_SIGTXT-'].update(visible=True)
            window['-DR_SIG-'].update(visible=True)

    # If PDF settings are selected. Open new window to recieve settings and create a new values dictionary with added entires.
    if event == '-DR_MC_TYPE-':
        MC_holding_dict = get_pdf_settings(event)
        window.refresh()

    ### DISTRIBUTION ANALYSIS EVENTS ----------------------------------------------------------------------------------
    # Turn DA groups into a table filled with ones
    if event in ['-DA_STKGRP-', '-DA_GEOGRP-', 'Trigger Refresh']:
        # Grab input from stakeholder and geo zones inputs and transmute into lists.
        stakeholder_list = values['-DA_STKGRP-'].split('\n')
        geozones_list = values['-DA_GEOGRP-'].split('\n')
        # Generate new DA and income tables if the stakheholder groups are changed.
        if event in ['-DA_STKGRP-', '-DA_GEOGRP-']:
            DA_table = [['DA Table'] + geozones_list]
            for r in range(1, len(stakeholder_list) + 1):
                new_row = [stakeholder_list[r - 1]] + [1.0] * len(geozones_list)
                DA_table.append(new_row)
            # Generate a new table to hold income data.
            income_table = [['Income Table'] + geozones_list]
            for r in range(1, len(stakeholder_list) + 1):
                new_row = [stakeholder_list[r - 1]] + [1.0] * len(geozones_list)
                income_table.append(new_row)

        if len(values['-DA_STKGRP-']) > 0 and len(values['-DA_GEOGRP-']) > 0:
            window['-DA_INPUT WEIGHTS_BASIC-'].update(disabled=False)
            window['-DA_INPUT WEIGHTS_INCOME-'].update(disabled=False)
            window['-DA_TABLE_DISPLAY-'].update(visible=True)
            window['-DA_TABLE_DISPLAY-'].update(
                pd.DataFrame(DA_table[1:len(DA_table)], columns=DA_table[0]).to_string(index=False))
        else:
            window['-DA_INPUT WEIGHTS_BASIC-'].update(disabled=True)
            window['-DA_INPUT WEIGHTS_INCOME-'].update(disabled=True)
            window['-DA_TABLE_DISPLAY-'].update(visible=False)

        # Update the list of stakeholders and geozones in the Quantity Scenarios lists
        for scn in range(0, n_scenarios + 1):
            for grp in range(1, len(qs_group_dict) + 1):
                for lin in range(1, qs_group_dict[grp] + 1):
                    window[f'-SCN{scn}_QG{grp}_LIN{lin}_STKE-'].update(values=stakeholder_list, value=values[
                        f'-SCN{scn}_QG{grp}_LIN{lin}_STKE-'])
                    window[f'-SCN{scn}_QG{grp}_LIN{lin}_GEOZN-'].update(values=geozones_list, value=values[
                        f'-SCN{scn}_QG{grp}_LIN{lin}_GEOZN-'])

    # Open popup to enter custom weights for distribution analysis
    if event == '-DA_INPUT WEIGHTS_BASIC-':
        # Send DA_table to a new popup that will generate a weight matrix. When the user updates the weight matrix, DA_table will be updated.
        weight_matrix_simple()
        window['-DA_TABLE_DISPLAY-'].update(
            pd.DataFrame(DA_table[1:len(DA_table)], columns=DA_table[0]).to_string(index=False))
    if event == '-DA_INPUT WEIGHTS_INCOME-':
        weight_matrix_incomes()
        window['-DA_TABLE_DISPLAY-'].update(
            pd.DataFrame(DA_table[1:len(DA_table)], columns=DA_table[0]).to_string(index=False))

    ### PARAMETER EVENTS ----------------------------------------------------------------------------------------------
    if event == 'Add Parameter':
        n_params += 1
        add_parameter()
        if window['-PAR_COLS-'].get_size()[1] < 700:
            resize = 280 + window['-PAR_COLS-'].get_size()[1]
            set_size(window['-PAR_COLS-'], (None, resize))
        window.refresh()
        window['-PAR_COLS-'].contents_changed()

    # If PDF settings are changed, open dialogue to receive PDF inputs and update window.
    if re.match(r'-PAR\d*_MC_TYPE-', event):
        MC_holding_dict = get_pdf_settings(event)
        window.refresh()

    ### FUNCTION EVENTS -----------------------------------------------------------------------------------------------
    # Add a new function when required.
    if event == 'Add Function':
        n_functions += 1
        add_function()

    # If the Smoothed Cubic Function is selected, make the smoothing parameter option field visible.
    if any([re.match(r'-FNC\d*_INTERP-', event)]):
        fnc = [int(g) for g in re.findall(r'\d+', event)][0]
        if values[f'-FNC{fnc}_INTERP-'] == 'Smoothed cubic spline':
            window[f'-FNC{fnc}_SMOOTHTXT-'].update(visible=True)
            window[f'-FNC{fnc}_SMOOTHSET-'].update(visible=True)
        else:
            window[f'-FNC{fnc}_SMOOTHTXT-'].update(visible=False)
            window[f'-FNC{fnc}_SMOOTHSET-'].update(visible=False)
        window.refresh()

        if any([re.match(r'-FUNC\d*_GRAPH-', event)]):
            fnc = [int(g) for g in re.findall(r'\d+', event)][0]
            function_popup(values[f'-FNC{fnc}_XVALS-'], values[f'-FNC{fnc}_YVALS-'], values[f'-FNC{fnc}_INTERP-'])

    # Create a popup to preview the smoothing function on the data provided.
    if any([re.match(r'-FNC\d*_GRAPH-', event)]):
        fnc = [int(g) for g in re.findall(r'\d+', event)][0]
        function_popup(fnc)


    ### REFERENCE PRICES EVENTS ---------------------------------------------------------------------------------------
    if event == '-ADD_PRICE_GROUP-':
        pr_group_dict[n_pr_groups + 1] = 0
        n_pr_groups += 1
        add_price_group(n_pr_groups)
        add_price_line(n_pr_groups, values['-PR_COUNTRY-'], values['-PR_YEAR-'])
        pr_group_dict[n_pr_groups] += 1
        # Limit the height of the column to 700 pixels
        if window['-PG_COLS-'].get_size()[1] < 700:
            resize = 280 + window['-PG_COLS-'].get_size()[1]
            set_size(window['-PG_COLS-'], (None, resize))
        window.refresh()
        window['-PG_COLS-'].contents_changed()

    # Add a new Value line to an existing price group.
    if event in [f'-ADD_TO_PG_{i}-' for i in range(1, n_pr_groups + 1)]:
        g = [int(g) for g in re.findall(r'\d+',
                                        event)]  # use RegEx to find the integer within the event, which should be the group number.
        add_price_line(g[0], values['-PR_COUNTRY-'], values['-PR_YEAR-'])
        pr_group_dict[g[0]] += 1
        if window['-PG_COLS-'].get_size()[1] < 700:
            resize = 145 + window['-PG_COLS-'].get_size()[1]
            set_size(window['-PG_COLS-'], (None, resize))
        window.refresh()
        window['-PG_COLS-'].contents_changed()

    # Make updates when any of the base settings are changed.
    if event in ['-PR_COUNTRY-', '-PR_YEAR-', '-PR_CUR_RANGE-', '-PR_CUR_POINT-', 'Trigger Refresh']:
        current_year = values['-PR_YEAR-']
        econ_ISO3 = econ_codes.loc[values['-PR_COUNTRY-'], 'ISO3']
        cpi_numerator(econ_ISO3, current_year)
        window['-CPI_NUMRTXT-'].update(
            f'The CPI conversion numerator is {np.around(cpi_numr, 3)} based on most recent data for year {cpi_numr_yr}')

        exchr_home = exchange_rate(econ_ISO3, current_year, values['-PR_CUR_RANGE-'], values['-PR_CUR_POINT-'])
        window['-EXCHR_TXT-'].update(
            f'The benchmark exchange rate is {econ_codes._get_value(values["-PR_COUNTRY-"], "ISO3")}{np.around(exchr_home, 3)}/USD.')
        window.refresh()
        calc_real_adj_values()

    # Check to see if price updates are made in any of the line items and trigger calculations.
    if any([re.match(r'-PG\d*_LIN\d*_NV-', event), re.match(r'-PG\d*_LIN\d*_CUR-', event),
            re.match(r'-PG\d*_LIN\d*_CURYR-', event), re.match(r'-PG\d*_LIN\d*_AF-', event), event == '-RES_RUN-']):
        calc_real_adj_values()

    # If line item names are changed, or quantity lines added collate all names and update the Price per Unit Combo box in the Quantity Scenarios tab.
    if any([re.match(r'-PG\d*_LIN\d*_ID-', event), re.match(r'-ADD_TO_QG_\d*-', event), event == 'Trigger Refresh']):
        price_ID_list = []
        for grp in range(1, n_pr_groups + 1):
            for lin in range(1, pr_group_dict[grp] + 1):
                price_ID_list.append(values[f'-PG{grp}_LIN{lin}_ID-'])

        for scn in range(0, n_scenarios + 1):
            for grp in range(1, len(qs_group_dict) + 1):
                for lin in range(1, qs_group_dict[grp] + 1):
                    window[f'-SCN{scn}_QG{grp}_LIN{lin}_PRICE-'].update(values=price_ID_list, value=values[
                        f'-SCN{scn}_QG{grp}_LIN{lin}_PRICE-'])
        window.refresh()

    # If PDF settings are selected. Open new window to recieve settings and create a new values dictionary with added entires.
    if any([re.match(r'-PG\d*_MC_TYPE-', event), re.match('-PG\d*_LIN\d*_MC_TYPE-', event)]):
        MC_holding_dict = get_pdf_settings(event)
        #print('this is the new holding dictionary', MC_holding_dict)
        window.refresh()

    ### QUANTITY SCENARIO EVENTS --------------------------------------------------------------------------------------
    # Add Scenarios
    if event == 'Add Scenario':
        n_scenarios += 1
        add_scenario_tab()

    # Open Event Model Builder
    if event == 'Event Model Builder':
        event_model_builder()

    if event in ('Event Model Builder', 'Trigger Refresh'):
        # Update the lists of available event outcome dependencies.
        for scn in range(0, n_scenarios + 1):
            for grp in range(1, len(qs_group_dict) + 1):
                for lin in range(1, qs_group_dict[grp] + 1):
                    old_select = values[f'-SCN{scn}_QG{grp}_LIN{lin}_DEPENDS-']
                    window[f'-SCN{scn}_QG{grp}_LIN{lin}_DEPENDS-'].update(values=['None'] + model_outcome_list)
                    if old_select in ['None'] + model_outcome_list:
                        window[f'-SCN{scn}_QG{grp}_LIN{lin}_DEPENDS-'].update(value=old_select)
                    else:
                        window[f'-SCN{scn}_QG{grp}_LIN{lin}_DEPENDS-'].update(value='None')
        window.refresh()

    # Add Groups.
    if any([re.match(r'ADD_QNT_GROUP_SCN\d*', event)]):
        qs_group_dict[len(qs_group_dict) + 1] = 0
        for s in range(0, n_scenarios + 1):
            add_quantity_group(s, len(qs_group_dict))
        # Limit the height of the column to 950 pixelx
        if window['-QS_COLS-'].get_size()[1] < 950:
            resize = 168 + window['-QS_COLS-'].get_size()[1]
            set_size(window['-QS_COLS-'], (None, resize))
        window.refresh()
        window['-QS_COLS-'].contents_changed()

    # Add Lines.
    if any([re.match(r'-ADD_TO_QG_\d*-', event)]):
        g = [int(g) for g in re.findall(r'\d+', event)]
        qs_group_dict[g[0]] += 1
        for s in range(0, n_scenarios + 1):
            add_quantity_line(s, g[0])
        # Limit the height of the column to 950 pixelx
        if window['-QS_COLS-'].get_size()[1] < 950:
            resize = 106 + window['-QS_COLS-'].get_size()[1]
            set_size(window['-QS_COLS-'], (None, resize))
        window.refresh()
        window['-QS_COLS-'].contents_changed()

    # Change Quantity Group Type across all Scenarios.
    if any([re.match(r'-SCN0_QG\d*_GRP_TYPE-', event)]):
        g = [int(g) for g in re.findall(r'\d+', event)]
        for s in range(1, n_scenarios + 1):
            window[f'-SCN{s}_QG{g[1]}_GRP_TYPE-'].update(values[f'-SCN0_QG{g[1]}_GRP_TYPE-'])

    if event == 'Trigger Refresh':
        for s in range(1, n_scenarios + 1):
            for grp in range(1, len(qs_group_dict) + 1):
                window[f'-SCN{s}_QG{grp}_GRP_TYPE-'].update(values[f'-SCN0_QG{grp}_GRP_TYPE-'])

    # Update the Price/Units text if the relevant inputs are changed.
    if any([re.match(r'-SCN\d*_QG\d*_LIN\d*_PRICE-', event), re.match(r'-PG\d*_LIN\d*_ID-', event),
            re.match(r'-PG\d*_LIN\d*_NV-', event), re.match(r'-PG\d*_LIN\d*_CUR-', event),
            re.match(r'-PG\d*_LIN\d*_CURYR-', event), re.match(r'-PG\d*_LIN\d*_AF-', event),
            event == 'Trigger Refresh']):
        for scn in range(0, n_scenarios + 1):
            for grp in range(1, len(qs_group_dict) + 1):
                for lin in range(1, qs_group_dict[grp] + 1):
                    price_group_line = [grp, lin]
                    '''price_group_line = [int(g) for g in re.findall(r'\d+', next(
                        key for key, value in values.items() if
                        value == values[f'-SCN{scn}_QG{grp}_LIN{lin}_PRICE-'] and key[0:3] == '-PG'))]'''
                    try:
                        price_unit_text = str(
                            MC_holding_dict[f'-PG{price_group_line[0]}_LIN{price_group_line[1]}_RAV_FULL-']) + '/' + values[f'-PG{price_group_line[0]}_LIN{price_group_line[1]}_UN-']
                        window[f'-SCN{scn}_QG{grp}_LIN{lin}_PRUN-'].update(price_unit_text)
                    except KeyError:
                        window[f'-SCN{scn}_QG{grp}_LIN{lin}_PRUN-'].update('NO VALUES')
                    except IndexError:
                        pass

    # Show Quantity Plot popups on request.
    if any([re.match(r'-SCN\d*_QG\d*_LIN\d*_PGRAPH-', event)]):
        scn, grp, lin = [int(g) for g in re.findall(r'\d+', event)]
        quantity_stream_popup(values[f'-SCN{scn}_QG{grp}_LIN{lin}_PRANGE-'],
                              values[f'-SCN{scn}_QG{grp}_LIN{lin}_PQUANT-'])

    # Show Event Plot popups on request.
    if any([re.match(r'-SCN\d*_QG\d*_LIN\d*_EGRAPH-', event), re.match(r'-SCN\d*_QG\d*_EGRAPH-', event)]):
        if '_LIN' in event:
            scn, grp, lin = [int(g) for g in re.findall(r'\d+', event)]
            quantity_stream_popup(values[f'-SCN{scn}_QG{grp}_LIN{lin}_ERANGE-'],
                                  values[f'-SCN{scn}_QG{grp}_LIN{lin}_EPROB-'])
        else:
            scn, grp = [int(g) for g in re.findall(r'\d+', event)]
            quantity_stream_popup(values[f'-SCN{scn}_QG{grp}_ERANGE-'],
                                  values[f'-SCN{scn}_QG{grp}_EPROB-'])

    # Insert here: get PDF settings for the quantity entries as well.
    # If PDF settings are selected. Open new window to recieve settings and create a new values dictionary with added entires.
    if any([re.match(r'-SCN\d*_QG\d*_MC_TYPE-', event), re.match('-SCN\d*_QG\d*_LIN\d*_MC_TYPE-', event)]):
        MC_holding_dict = get_pdf_settings(event)
        #print('this is the new holding dictionary', MC_holding_dict)
        window.refresh()

    # RESULTS EVENTS ---------------------------------------------------------------------------------------------------
    if event == '-RES_RUN-':
        try:
            # Prepare Random Number Generators and Simulation Counter
            seed = int(values['-RES_SET_SEED-'])
            random_generator = np.random.default_rng(seed)
            n_simulations = int(values['-RES_NUM_SIMS-'])
            split_pranges = [list(map(int, re.split(',|\n', values[key]))) for key in values if 'PRANGE-' in str(key)]
            split_pranges_flat = [period for sublist in split_pranges for period in sublist]
            # Get the maximum number of simulation periods from the Quantity Scenarios Tab
            n_simulation_periods = np.amax(np.array(split_pranges_flat)) + 1
            # Create a map of ones of the size of the total arrays to facilitate broadcasting to the right shape.
            one_map = np.ones((n_simulations, n_simulation_periods))

            # Function events
            function_dict = {}
            if n_functions > 0:
                for f in range(1, n_functions+1):
                    # Add function names to a list so that other functions can check.
                    function_names.append(values[f'-FNC{f}_ID-'])

                    # Assign functions to a dictionary to be called.
                    function_dict[values[f'-FNC{f}_ID-']] = assign_function(f)

            # Parameters Dictionary
            parameter_dict = {}
            parameter_dict['t'] = {}
            parameter_dict['t']['expression'] = np.linspace(0, n_simulation_periods - 1, num=n_simulation_periods)*one_map
            if n_params > 0:
                parameter_checklist = []
                for par in range(1, n_params + 1):
                    parameter_checklist.append(values[f'-PAR{par}_ID-'])
                for par in range(1, n_params+1):
                    parameter_dict[f'{values[f"-PAR{par}_ID-"]}'] = {}
                    parameter_dict[f'{values[f"-PAR{par}_ID-"]}']['index'] = par
                    parameter_dict[f'{values[f"-PAR{par}_ID-"]}']['expression'] = 'EMPTY'

                while len(parameter_checklist) > 0:
                    for par in parameter_checklist:
                        # Check to see if the parameter has any dependencies that have not been calculated yet.
                        if parameter_dict[f'{par}']['expression'] == 'EMPTY':
                            par_skip_flag = False
                            p_n = parameter_dict[par]['index']
                            par_deps_list = re.findall(r'[a-zA-Z^]+', values[f'-PAR{p_n}_PVALUE-'])
                            par_deps_list = [m for m in par_deps_list if m not in math_expressions]
                            par_deps_list = [m for m in par_deps_list if m not in function_names]
                            if 'None' in par_deps_list:
                                par_deps_list.remove('None')
                            for par_dep in par_deps_list:
                                if parameter_dict[f'{par_dep}']['expression'] == 'EMPTY':
                                    par_skip_flag = True
                                    break
                            if par_skip_flag is not True:
                                par_num = parameter_dict[f'{par}']['index']
                                parameter_dict[f'{par}']['expression'] = expression_to_array(
                                    n_simulation_periods,
                                    values[f'-PAR{par_num}_PRANGE-'],
                                    values[f'-PAR{par_num}_PVALUE-'])
                                # Broadcast Monte Carlo random noise array over the expression array.
                                par_monte_carlo_sims = monte_carlo_expression(
                                    f'-PAR{par_num}',
                                    values[f'-PAR{par_num}_MC_TYPE-'],
                                    n_simulations)
                                parameter_dict[f'{par}']['expression'] = parameter_dict[f'{par}']['expression']*par_monte_carlo_sims
                                parameter_checklist.remove(par)
                                np.savetxt(f"D:/Local Account/Documents/Graduate Diploma/MASTERS DISSERTAATION/BUS505/Parameters/{par}.csv", parameter_dict[f'{par}']['expression'], delimiter=',')

            # Discount Rate Array
            if values['-DR_TYPE-'] == 'Constant':
                dr_base_array = np.full((1, n_simulation_periods), float(values['-DR_MAIN-']) / 100)
            elif values['-DR_TYPE-'] == 'Stepped':
                step_ranges_list = values['-DR_STEP_RANGE-'].split('\n')
                step_rates_list = values['-DR_STEP_RATE-'].split('\n')
                n_steps = len(step_ranges_list)
                dr_base_array = np.zeros(n_simulation_periods)
                end = 0
                for step in range(0, n_steps):
                    start, end = map(int, step_ranges_list[step].split(','))
                    end += 1
                    dr_base_array[start:end] = float(step_rates_list[step]) / 100
                if end <= n_simulation_periods:
                    dr_base_array[end:n_simulation_periods] = float(values['-DR_STEP_BASE-']) / 100
            elif values['-DR_TYPE-'] == 'Gamma time declining':
                mu = float(values['-DR_MAIN-']) / 100
                sig = float(values['-DR_SIG-']) / 100
                dr_base_array = np.array([(mu / (1 + ((t - 1) * sig ** 2) / mu)) for t in range(0, n_simulation_periods)])
            dr_base_array[0][0] = 0
            dr_base_array = np.tile(dr_base_array, (n_simulations, 1))
            dr_monte_carlo_factors = monte_carlo_expression('-DR', values['-DR_MC_TYPE-'], n_simulations)
            # Produce a 2d array where rows=simulation and columns=time.
            # Note that if dr_monte_carlo_factors is a single column array then the function should broadcast.
            dr_monte_carlo_array = np.cumprod(1 / (1 + dr_monte_carlo_factors * dr_base_array), axis=1)

            # Prices Array
            reference_price_monte_carlo_table = []
            #print(pr_group_dict)
            for grp in pr_group_dict:
                # Column matrix for simulations.
                group_monte_carlo_sims = monte_carlo_expression(
                    f'-PG{grp}',
                    values[f'-PG{grp}_MC_TYPE-'],
                    n_simulations)

                for lin in range(1, pr_group_dict[grp] + 1):
                    reference_price_monte_carlo_table.append({
                        'group': grp,
                        'line': lin,
                        'value': values[f'-PG{grp}_LIN{lin}_ID-'],
                        'prefix': f'-PG{grp}_LIN{lin}',
                        'real_adj_value': expression_to_array(n_simulation_periods, f'0,{n_simulation_periods-1}',
                                                              str(MC_holding_dict[f'-PG{grp}_LIN{lin}_RAV_FULL-'])),
                        'line_monte_carlo_sims': monte_carlo_expression(
                            f'-PG{grp}_LIN{lin}',
                            values[f'-PG{grp}_LIN{lin}_MC_TYPE-'],
                            n_simulations
                        ),
                        'group_monte_carlo_sims': group_monte_carlo_sims
                    })
            reference_price_monte_carlo_table = pd.DataFrame.from_records(reference_price_monte_carlo_table)

            # Create Column matrix for the weighted monte carlo-shifted values.
            reference_price_monte_carlo_table['monte_carlo_values'] = (reference_price_monte_carlo_table[
                                                                           'line_monte_carlo_sims'] +
                                                                       reference_price_monte_carlo_table[
                                                                           'group_monte_carlo_sims'] - 1) * \
                                                                      reference_price_monte_carlo_table['real_adj_value']

            # Event Outcomes Array
            # Convert shorthands to appropriate data.
            for scn in range(0, n_scenarios + 1):
                for model_event in range(1, len(model_events_dict) + 1):
                    for outs in range(1, model_events_dict[model_event] + 1):
                        for suffix in ['RANGE', 'WEIGHT', 'NOREPS', 'MAXREPS']:
                            if model_events_user_settings[f'-EVENT{model_event}_OUTCOME{outs}_SCN{scn}_{suffix}-'] == '<<<':
                                model_events_user_settings[f'-EVENT{model_event}_OUTCOME{outs}_SCN{scn}_{suffix}-'] = \
                                    model_events_user_settings[f'-EVENT{model_event}_OUTCOME{outs}_SCN{0}_{suffix}-']
                        if model_events_user_settings[f'-EVENT{model_event}_OUTCOME{outs}_SCN{scn}_MAXREPS-'] == 'None':
                            model_events_user_settings[f'-EVENT{model_event}_OUTCOME{outs}_SCN{scn}_MAXREPS-'] = '-1'

            event_outcome_records = []
            # Inlcude a 'none' record for where there are no dependent events.
            for scn in range(0, n_scenarios + 1):
                event_outcome_records.append({
                    'event_id': 'None',
                    'event': 'None',
                    'outcome_id': 'None',
                    'outcome': 'None',
                    'scenario': scn,
                    'outcome_monte_carlo_sims': np.full((n_simulations, n_simulation_periods), True)
                })

            model_events_list = list(model_events_dict.keys())
            if len(model_events_list) == 0:
                event_outcome_table = pd.DataFrame.from_records(event_outcome_records)
            else:
                event_outcome_table = pd.DataFrame({
                    'event_id': ['blank'],
                    'event': ['blank'],
                    'outcome_id': ['blank'],
                    'scenario': ['blank'],
                    'outcome_monte_carlo_sims': ['blank']
                })
            #print(uniform.rvs(size=(1), random_state=random_generator))
            while len(model_events_list) > 0:
                for model_event in model_events_list:
                    depends_list = model_events_user_settings[f'-EVENT{model_event}_DEPENDS-']
                    if all(depend in event_outcome_table['outcome_id'].tolist() for depend in depends_list):
                        model_event_monte_carlo_sims = uniform.rvs(size=(n_simulations, n_simulation_periods),
                                                                   random_state=random_generator)
                        for scn in range(0, n_scenarios + 1):

                            # Filter the table for the outcomes and scenarios needed, prepare the dependency mask.
                            if not depends_list:
                                model_event_depends_mask = np.ones((n_simulations, n_simulation_periods), dtype=int)
                            else:
                                depends_stack = \
                                    event_outcome_table.loc[(event_outcome_table['outcome_id'].isin(depends_list)) &
                                                            (event_outcome_table['scenario'] == scn)][
                                        'outcome_monte_carlo_sims'].tolist()
                                depends_stack.append((np.full((n_simulations, n_simulation_periods), True)))
                                model_event_depends_mask = np.prod(depends_stack, axis=0)

                            model_event_out_weights = np.stack([np.array(expression_to_array(n_simulation_periods,
                                                                                             model_events_user_settings[
                                                                                                 f'-EVENT{model_event}_OUTCOME{o}_SCN{scn}_RANGE-'],
                                                                                             model_events_user_settings[
                                                                                                 f'-EVENT{model_event}_OUTCOME{o}_SCN{scn}_WEIGHT-']))
                                                                for o in range(1, model_events_dict[model_event] + 1)])

                            model_event_out_noreps = np.stack(
                                [int(model_events_user_settings[f'-EVENT{model_event}_OUTCOME{o}_SCN{scn}_NOREPS-'])
                                 for o in range(1, model_events_dict[model_event] + 1)])

                            model_event_out_maxreps = np.stack(
                                [int(model_events_user_settings[f'-EVENT{model_event}_OUTCOME{o}_SCN{scn}_MAXREPS-'])
                                 for o in range(1, model_events_dict[model_event] + 1)])

                            outcome_array_dict = outcome_arrays(model_event_out_weights, model_event_out_noreps,
                                                                model_event_out_maxreps, model_event_monte_carlo_sims,
                                                                model_event_depends_mask)

                            for out in range(1, model_events_dict[model_event] + 1):
                                event_outcome_records.append({
                                    'event_id': model_events_user_settings[f'-EVENT{model_event}_ID-'],
                                    'event': model_event,
                                    'outcome_id': model_events_user_settings[f'-EVENT{model_event}_OUTCOME{out}_ID-'],
                                    'outcome': out,
                                    'scenario': scn,
                                    'outcome_monte_carlo_sims': outcome_array_dict[out]
                                })
                            event_outcome_table = pd.DataFrame.from_records(event_outcome_records)
                        model_events_list.remove(model_event)

            # Quantity Scenarios Array
            quantity_stream_records = []
            for scn in range(0, n_scenarios + 1):
                for grp in qs_group_dict:
                    group_monte_carlo_sims = monte_carlo_expression(
                        f'-SCN{scn}_QG{grp}',
                        values[f'-SCN{scn}_QG{grp}_MC_TYPE-'],
                        n_simulations)
                    for lin in range(1, qs_group_dict[grp] + 1):
                        quantity_stream_records.append({
                            'scenario': scn,
                            'group': grp,
                            'group_id': values[f'-SCN{scn}_QG{grp}_ID-'],
                            'type': values[f'-SCN{scn}_QG{grp}_GRP_TYPE-'],
                            'line': lin,
                            'line_id': values[f'-SCN{scn}_QG{grp}_LIN{lin}_ID-'],
                            'prefix': f'-SCN{scn}_QG{grp}_LIN{lin}',
                            'value': values[f'-SCN{scn}_QG{grp}_LIN{lin}_PRICE-'],
                            'stakeholder': values[f'-SCN{scn}_QG{grp}_LIN{lin}_STKE-'],
                            'geography': values[f'-SCN{scn}_QG{grp}_LIN{lin}_GEOZN-'],
                            'quantity_stream': expression_to_array(n_simulation_periods,
                                                                   values[f'-SCN{scn}_QG{grp}_LIN{lin}_PRANGE-'],
                                                                   values[f'-SCN{scn}_QG{grp}_LIN{lin}_PQUANT-']),
                            'line_monte_carlo_sims': monte_carlo_expression(
                                f'-SCN{scn}_QG{grp}_LIN{lin}',
                                values[f'-SCN{scn}_QG{grp}_LIN{lin}_MC_TYPE-'],
                                n_simulations),
                            'group_monte_carlo_sims': group_monte_carlo_sims,
                            'outcome_id': values[f'-SCN{scn}_QG{grp}_LIN{lin}_DEPENDS-']
                        })
            quantity_stream_table = pd.DataFrame.from_records(quantity_stream_records)
            # print(quantity_stream_table.to_string())
            # Create column for the weighted monte carlo shifted quantity streams.
            quantity_stream_table['monte_carlo_streams'] = (quantity_stream_table['line_monte_carlo_sims'] +
                                                            quantity_stream_table['group_monte_carlo_sims'] - 1) * \
                                                           quantity_stream_table['quantity_stream']

            # Add the results of the outcome table to the quantity table.
            try:
                quantity_stream_table = pd.merge(quantity_stream_table, event_outcome_table[['outcome_id', 'scenario',
                                                                                             'outcome_monte_carlo_sims']],
                                                 how='left', on=['outcome_id', 'scenario'])
            except ValueError:
                quantity_stream_table['outcome_monte_carlo_sims'] = 1

            # Create column for the combined event and stream quantities.
            quantity_stream_table['combined_stream_quantity_events'] = quantity_stream_table['monte_carlo_streams'] * \
                                                                       quantity_stream_table['outcome_monte_carlo_sims']

            # Merge the values table with the quantity streams table.
            quantity_stream_table = pd.merge(quantity_stream_table, reference_price_monte_carlo_table[['value',
                                                                                                       'monte_carlo_values']],
                                             how='left', on='value')

            # Calculate nominal and discounted value streams.
            quantity_stream_table['nominal_quantity_value_stream'] = quantity_stream_table[
                                                                         'combined_stream_quantity_events'] * \
                                                                     quantity_stream_table['monte_carlo_values']

            quantity_stream_table[
                'discounted_quantity_value_stream'] = quantity_stream_table.nominal_quantity_value_stream.apply(
                lambda x: x * dr_monte_carlo_array)

            # Calculate nominal and discounted total value at points.
            quantity_stream_table['nominal_total_value'] = quantity_stream_table.nominal_quantity_value_stream.apply(
                lambda x: np.sum(x, axis=1))
            quantity_stream_table['discounted_total_value'] = quantity_stream_table.discounted_quantity_value_stream.apply(
                lambda x: np.sum(x, axis=1))
            quantity_stream_table['nominal_total_value_mean'] = quantity_stream_table.nominal_total_value.apply(
                lambda x: np.average(x))
            quantity_stream_table['discounted_total_value_mean'] = quantity_stream_table.discounted_total_value.apply(
                lambda x: np.average(x))
            quantity_stream_table['discounted_total_value_perc'] = quantity_stream_table.discounted_total_value.apply(
                lambda x: np.percentile(x, [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95], method='linear'))
            for perc in [(0, 5), (1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60), (7, 70), (8, 80), (9, 90), (10, 95)]:
                quantity_stream_table[
                    f'discounted_total_value_P{perc[1]}'] = quantity_stream_table.discounted_total_value_perc.apply(
                    lambda x: x[perc[0]]
                )

            # Attatch the DA_table weights to the table.
            DA_frame = pd.DataFrame({'stakeholder': [''], 'geography': [''], 'weight': [1]})
            if len(DA_table) > 0:
                DA_table[0][0] = 'stakeholder'
                DA_frame_items = pd.DataFrame(DA_table[1:], columns=DA_table[0])
                DA_frame_items = pd.melt(DA_frame_items, id_vars='stakeholder', value_vars=DA_table[0][1:],
                                         value_name='weight',
                                         var_name='geography')
                DA_frame = pd.concat([DA_frame, DA_frame_items])
            quantity_stream_table = pd.merge(quantity_stream_table, DA_frame, how='left', on=['stakeholder', 'geography'])
            # print(quantity_stream_table.to_string())
            run_number += 1
            result_windows[run_number] = result_popup(run_number)
            plot_histo_cdf(run_number,
                           0,
                           run_cba_settings[run_number]['use_weights'],
                           run_cba_settings[run_number]['net_z'],
                           run_cba_settings[run_number]['distr_segs'])

        except Exception as issue:
            print("A problem occurred with the inputs!")
            sg.popup(
                f'Here is the error that was raised:\n{type(issue)}: {issue}',
                title="Whoops! An error occurred!"
            )
    try:
        if run_number > 0:
            for run in range(1, run_number + 1):
                if result_windows[run] is not None:
                    loop_result_window_events(run)
    except Exception as result_display_issue:
        print("A problem occurred with displaying the results!")
        sg.popup(
            f'Here is the error that was raised:\n{type(result_display_issue)}: {result_display_issue}',
            title="Whoops! An error occurred!"
        )

    if event == sg.WIN_CLOSED:
        break

    if event != '__TIMEOUT__':
        print(event, values)
window.close()
