# RUBICS
RUBICS (RUB/CS): Random Uncertainty Benefit / Cost Simulator is an open source program to undertake Monte Carlo and Distributional Benefit-Cost Analysis, enabling easy peer review of inputs and results.

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
