# FalqinParaLib

![falqin_logo](00_readme_images/Logo_icon_colors.png)

© René Falquier, 2025: Logo authorized for use exclusively in connection with FalqinParaLib, see license for more details.

# 1. Project Overview
## 1.1 Description
The purpose of this project is to provide cross-country (XC) paraglider pilots with visual analytics to provide insights into their flights. 

The falqin class uses B-record slicing to parse an .igc file with an averaging window (default: 15 seconds) and uses deterministic logic to classify each GPS point as either:

- Thermal: When climbing and turning 
    - Climb Rate > 0 & Heading Rate of Change >= 10 °/s
- Glide: When heading is stable and glider is making forward progress
    - Heading Rate of Change <10 °/s & ground speed > 3 m/s
- Undefined: Point fits neither of the above criteria

The parsing logic makes no assumption as to the "quality" of the XC flight provided and the user needs to interpret the results in the context of the file(s) provided (distance achieved, conditions flown, geographical area, competition flight etc.). 

An XC flight is, by definition, performed in active air and the corresponding metrics need to be interpreted with this in mind. 

The analytics are presented using SI units (e.g. m/s instead of km/h or ft/s).

![track](00_readme_images/track.png)

**Example:** The ```plot_3d_track()``` method outputs a 3D visualization of a provided track, color-coded to show the user how their flight-points were classified.

## 1.2 Project Status
The project is **ACTIVE** and open to pull-requests. Please see [§3. Wishlist / Future Work](#3-wishlist--future-work).

## 1.3 Methods Used
* Descriptive Statistics
* Filtering 
* GPS Coordinate Conversions
* Haversine Distance Calculations
* Spherical Trigonometry Heading Calculations
* Object Oriented Programming

## 1.4 Technologies
* Jupyter
* Matplotlib
* Numpy 
* Pandas
* Plotly
* Python

# 2. Instructions
## 2.1 Dependencies
- Python 3.10.16
    - matplotlib 3.10.0
    - numpy 2.2.1
    - pandas 2.2.3
    - plotly 5.24.1
    
**NOTE:** Jupyter notebooks are required to manipulate the provided [example notebook](falqinparalib/falqinparalib_example.ipynb).

## 2.2 falqin Class Initialization 
The ```falqin()``` class is initialized with the following inputs:

- filepath : str
    - Path to a single IGC file, or a directory containing *.IGC files if ```folder=True```.
- avg_window : int, default 15
    - Rolling window length (samples) used to compute averaged quantities and flight-state classification.
- speed_bins : array-like ground-speed bin edges (m/s), default uses ```np.linspace(7.25, 16.25, 19)```
    - Each array input is the edge of a ground-speed bin for polar calculations
- LD_max : int or float, default 14
    - Maximum glide ratio retained when filtering gliding data.
        - **NOTE**: For a low-efficiency airfoil, Oswald efficiency factor e<1, e≈0.8 (typical for non-elliptical lift distributions), AR=7, CD0≈0.025 (higher due to poor efficiency) i.e. max **theoretical** L/D for a paraglider of AR= 7 ≈14. Heuristically, remove two glide points per unit AR reduction.
- folder : bool, default False
    - If ```True```, parse all *.IGC files in `filepath` and aggregate results.
- verbose : bool, default False
    - If ```True```, print configuration and progress messages.

## 2.3 falqin Class Methods
The provided [example notebook](falqinparalib/falqinparalib_example.ipynb) showcases an overview of the methods in ```falqinparalib``` with file parsing and folder-level parsing setups. 

## 2.4 Interpretation Considerations & Input-File Considerations
### 2.4.2 Things to Consider While Interpreting Results
An XC flight takes place in active air and your results will therefore be representative of your contextual performance in the airmass and conditions that were flown. Care should be taken when comparing performance between two flights, especially if they were flown in different conditions. 

It is also important to note that glide calculations are performed relative to ground, not air. As such, they are not representative of aerodynamic performance, but rather of ground-track performance relative to the day's conditions (a day spent fighting headwind will look very different than a downwinder over the flatlands). 

![heatmap](00_readme_images/Thermal_distr_heatmap_thermal_basis.png)

**Example:** The heatmap above shows the distribution of climb rates vs. rate of turn for an XC competition flight, of which one hour was spent circling at the top-of lift with the start-gaggle prior to task-start. The remaining 70km task was then flown in 2.5h with roughly 10 thermals and corresponding glides. Knowing that changes how one will interpret this image.

![prob_threshold](00_readme_images/prob_threshold.png)

**Example:** The Cumulate Density Function above is for a group of flights flown in St. Andre in windy conditions during a week of competition with relatively low cloudbase. How might my glide distribution look if I were flying downwind under cloud-streets?

### 2.4.3 Things to Consider for Input Files/Folders
- **Flight-Level Analysis** As with most data processing, more is generally more, so a longer flight will provide a better statistical overview of glide and climb metrics. The caveat here is that "true" XC flights (e.g. >= 50km at >= 15 km/h average XC speed) will be more representative of your performance in active air than 2.5h of sightseeing at your local thermal site (not that there's anything wrong with that!). 
- **Folder-Level Analysis**: When providing a folder of files for analysis, the data will be more meaningful if you provide "grouped" data rather than the .igc files for every flight you have ever flown. For starters, a group of flights should have at least been flown on the same glider. Secondly, the files should be relatively consistent in terms of conditions flown / geographical area e.g. a group of flights flown in Fiesch in the spring will contain very different data than a group of flights flown in Bassano in the stability of summer. 

# 3. Wishlist / Future Work
**NOTE** Please submit pull requests with the corresponding information or equivalently descriptive information. Recommended to touch-base before kicking off your work.

- **20250902_001:** Option for non-SI units commonly used in the paragliding community (km/h for ground speed, seconds/turn for heading rate of turn etc.).
- **20250902_002:** Methods return contextually important KPI(s) instead of just plots/visuals.
- **20250902_003:** Providing user the parameters to fine-tune classification to their preference (e.g. heading rate and climb rate thresholds).
- **20250902_004:** Deployment to web application (e.g. Streamlit)
- **20250902_005:** Comparison outputs for two or more .igc files

# 4. Conclusion
When .igc file from an XC flight has a statistically sufficient number of points to approach an analysis statistically (at least 30 per speed bin) there is a wealth of information to be gained from even basic visualizations.

The Speed-bin histograms also show neat distributions that appear to approach a normal shape (in the statisical sense of the word). As such, they could theoretically be approached using the law of large numbers to extract means of means for each statistically relevant metric if a sufficient number of flights flown on the same glider in similar conditions are collected.

# 5. Team
## 5.1 Lead(s)
- **René Falquier**
    - [GitHub](https://github.com/rmfalquier)
    - [LinkedIn](https://www.linkedin.com/in/rmfalquier/)
- **YOU! Get in touch :-)**

## 5.2 Contributing Members
- **YOU! Get in touch :-)**