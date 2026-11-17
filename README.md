# Data Visualization Laboratory (PH 4131 Engineering Physics Laboratory)

Python scripts used to analyse and visualize the seven data sets of the Data Visualization laboratory sheet, Department of Physics, University of Colombo.

| Folder | Exercise | Script(s) |
| --- | --- | --- |
| `exercise1` | Pointer finger length | `ex1_plot1_left_vs_right.py` to `ex1_plot5_height_vs_difference.py` |
| `exercise2` | Thermometer calibration | `ex2_thermometer_calibration.py` |
| `exercise3` | LED emission spectra | `ex3_led_spectra.py` |
| `exercise4` | Insurance charges | `ex4_stepwise_analysis.py` |
| `exercise5` | Head size and brain weight | `ex5_brain_stepwise.py` |
| `exercise6` | COVID 19 in Asia | `ex6_covid_stepwise.py` |
| `exercise7` | Star classification (HR diagram) | `ex7_stars_stepwise.py` |
| `hr_explorer` | Interactive Hertzsprung Russell explorer | `index.html` (open in a web browser) |

## How to run

1. Install Python 3 and the libraries:
   ```
   pip install -r requirements.txt
   ```
2. Go into an exercise folder and run its script, for example:
   ```
   cd exercise2
   python ex2_thermometer_calibration.py
   ```
3. The figures and result tables are saved in an output folder inside the exercise folder, and all numbers are printed in the terminal.

Each script reads the Excel data file that is in the same folder. In Exercise 1 the student registration numbers were replaced by Student_01, Student_02, and so on.
