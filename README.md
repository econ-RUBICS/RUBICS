# RUBICS
RUBICS (RUB/CS): Random Uncertainty Benefit / Cost Simulator is an open source program to undertake Monte Carlo and Distributional Benefit-Cost Analysis, enabling easy peer review of inputs and results.

RUBICSv1.0.0 STAR Changelog

1. New feature allowing the user to define their own functions and use them to in calculated parameters. The user can upload a series of x and y values and choose and interpolation method (nearest, previous, next, linear, cubic spline, akima, and smoothed cubic spline) to produce a function that can be used in calculated parameters. 

2. Minor changes to the definition of exchange rate and cpi tables to account for changes in how those tables are downloaded. 

3. Minor changes to the list of special mathematical function names. Now include: 'np', 'min', 'max', 'ceil', 'fabs', 'floor', 'trunc', 'exp', 'expm', 'log', 'pow', 'sqrt', 'pi', 'e', 'clip', 'inf'

4. Placeholder code in function sensitivity_display_table() for a future regression analysis operation. Not used in the current version. 

5. New menu options that allow the user to download new data from the WorldBank API (dependent on whether the API is currently working) 

6. Placed RUN event loop in a try, except envelope in order to catch bugs and alert the user, rather than allowing the program to crash entirely. 

7. Minor bug fixes in the calculation of result outputs. 

RUBICSv0.2.0 Changelog
1. New feature allowing parameters to be defined and used as inputs in price and quantity equations. 

2. New feature allowing the user to specify a custom PDF from an array of x and y points using linear interpolation.

3. Changed the way percentiles are displayed for the result readouts to calculate a 'point median percentile' rather than use actual percentiles to minimise the risk that extreme line values are displayed. 

4. Fixed a calculation error in the discount rate formula.

5. Fixed a bug in the calculation of the total number of periods exlcuding the last period.

6. Fixed a bug in the report generation that threw an alert when no distribution analysis table is found. 

7. Fixed a bug that made the exporting data to CSV fail when there is no event model. 

8. Ensured that when new quantity scenario lines are added all values and distribution analysis lists are available.

9. New feature enabling random price/quantity factors to be generated in each period. 

10. Changed layout of reference price section to be more compact. 

11. Fixed an issue where calculating result tables returned SettingWithCopy warnings due to ambiguous dataframes. In this instance, setting values based on a copy of the original is the expected behaviour. 

12. Fixed a bug that caused the old Monte Carlo settings to fail to load into the dialogue box when updating.

13. Made it so that the main window updates properly to display the PDF settings on load.

14. Fixed bug with the calculation of random variables with the Tirangualr and PERT distributions.


RUBICSv0.1.1 Changelog

1. Fixed a bug that caused simple weights to be returned as strings instead of floats. 

2. Fixed bug that caused the event model window to not open at the correct size. 

3. Made the Event Model Window resizable. 

4. Rearranged elements of the results window to better utilise space. Made window resizable.

5. Fixed bug that caused the window close button on the results window to close the entire program. Removed fallback exit button. 

6. Enable user to export an image or pdf file of the graph generated in the results section.

7. Enable user to export a .csv of the tabulated simulation data. 

8. Fixed text prompt to load settings file. 

9. Enable user to import settings from a .xlsx file.

10. Fixed a bug that caused the list of values in the quantity scenarios tabs to fail to update. 

11. Fixed a visual bug that caused the Price/Units view on the quantity scenarios tab to show incorrectly. 

12. Changed read to model_events_user_settings for no_reps and max_reps to string to enable .json to write properly.

13. Fixed settings readout in generated results for period ranges and quantities to read as lists, not strings. And added missing event_depends in event model. And added PDF settings to readout.

14. Enabled export settings button in main window.

15. Removed export simulations to file button on results tab.
