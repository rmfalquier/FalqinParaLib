import os
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class falqin:
    """
    Falcon-quality IGC (paragliding) parser and analyzer.

    This class parses one IGC file—or all *.IGC files in a folder—into
    derived flight time series (ground speed, heading, climb rates, etc.),
    segments flight states (glide vs thermal) with a rolling window, and
    prepares speed-binned statistics and helper plots for analysis.

    Parameters
    ----------
    filepath : str
        Path to a single IGC file, or a directory containing *.IGC files if
        `folder=True`.
    avg_window : int, default 15
        Rolling window length (samples) used to compute averaged quantities
        and flight-state classification.
    speed_bins : array-like or "default", default "default"
        Ground-speed bin edges (m/s). If "default", uses np.linspace(7.25, 16.25, 19).
    LD_max : int or float, default 14
        Maximum glide ratio retained when filtering gliding data.
    folder : bool, default False
        If True, parse all *.IGC files in `filepath` and aggregate results.
    verbose : bool, default False
        If True, print a short configuration and progress messages.

    Attributes
    ----------
    file : str
        Current file path being parsed (auto-updates when provided a path to a folder with the corresponding boolean).
    avg_window : int
        Rolling window length.
    speed_bins : np.ndarray
        Ground-speed bin edges.
    LD_max : int or float
        Glide-ratio cutoff used for filtering.
    folder : bool
        Whether running in folder-mode.
    verbose : bool
        Verbose flag.
    general : pd.DataFrame
        Per-fix time series with derived columns (ground speed, heading,
        climb rates, averaged values, flight_state, glide_ratio).
    thermals : pd.DataFrame
        Per-thermal aggregates (height gain, turn count/rate, climb rates).
    gliding : pd.DataFrame
        Filtered subset of `general` in glide state with bounds applied.
    speed_bin_stats : pd.DataFrame
        Bin-level stats (mean/median/std/dev/IQR) for glide ratio and climb rates.
    glide_ratio_counts : list[int]
        Count of valid glide-ratio samples per ground-speed bin.
    climb_rate_counts : list[int]
        Count of valid climb-rate samples per ground-speed bin.
    """

    def __init__(self, 
                 filepath: str,
                 avg_window: int = 15,
                 speed_bins = "default",
                 LD_max = 14,
                 folder: bool = False,
                 verbose: bool = False):
        """Initialize configuration, then parse either a single file or a folder."""
        self.file = filepath
        self.avg_window = avg_window
        if speed_bins == "default":
            # Default ground-speed binning: 7.25–16.25 m/s, 18 bins (19 edges)
            self.speed_bins = np.linspace(7.25, 16.25, 19)
        else:
            self.speed_bins = speed_bins
        self.LD_max = LD_max
        self.folder = folder
        self.verbose = verbose

        if self.verbose:
            print("IGC parsing initialized with:")
            print(f"   -Filepath: {self.file}")
            print(f"   -Averaging window: {self.avg_window}")
            print(f"   -Speed bins: min {self.speed_bins.min()}, max {self.speed_bins.max()}, count {self.speed_bins.size}")
            print(f"   -Max L/D: {self.LD_max}")
            print(f"   -Folder: {self.folder}")
            print(f"   -Verbose: {self.verbose}\n")

        if self.folder :
            # Folder mode: iterate *.IGC files and aggregate parsed outputs
            igc_files = [f for f in os.listdir(self.file) if f.endswith('.IGC')]
            dir_path = self.file
            speed_bin_stats_list = []
            glide_ratio_counts_list = []
            climb_rate_counts_list = []

            for i, igc in enumerate(igc_files):
                # Update current file pointer and parse
                self.file = os.path.join(dir_path, igc)
                if i == 0 :
                    self.general = self.__general_parse()
                    self.thermals = self.__thermal_parsing()
                    self.gliding, speed_bin_stats, glide_ratio_counts, climb_rate_counts = self.__glide_polar_parsing()

                    # Keep parallel lists for later aggregation
                    speed_bin_stats_list.append(speed_bin_stats.drop(columns='Speed Bin'))
                    glide_ratio_counts_list.append(glide_ratio_counts)
                    climb_rate_counts_list.append(climb_rate_counts)

                else :
                    # Concatenate per-file outputs
                    self.general = pd.concat([self.general, self.__general_parse()], ignore_index=True)
                    self.thermals = pd.concat([self.thermals, self.__thermal_parsing()], ignore_index=True)
                    gliding, speed_bin_stats, glide_ratio_counts, climb_rate_counts = self.__glide_polar_parsing()
                    self.gliding = pd.concat([self.gliding, gliding], ignore_index=True)

                    speed_bin_stats_list.append(speed_bin_stats.drop(columns='Speed Bin'))
                    glide_ratio_counts_list.append(glide_ratio_counts)
                    climb_rate_counts_list.append(climb_rate_counts)

                if verbose:
                    print(f"IGC file {i+1}/{len(igc_files)} parsed for General, Thermals, and Gliding")

            # Aggregate by averaging per-bin statistics across files
            self.speed_bin_stats = pd.concat(speed_bin_stats_list).groupby(level=0).mean()

            # Aggregate counts across files (sum per bin)
            self.glide_ratio_counts = np.sum(glide_ratio_counts_list, axis=0)
            self.climb_rate_counts = np.sum(climb_rate_counts_list, axis=0)

        else :
            # Single-file mode
            self.general = self.__general_parse()
            if self.verbose:
                print("General parsing complete.")

            self.thermals = self.__thermal_parsing()
            if self.verbose:
                print("Thermals parsing complete.")
            
            self.gliding, self.speed_bin_stats, self.glide_ratio_counts, self.climb_rate_counts = self.__glide_polar_parsing()
            if self.verbose:
                print("Glide parsing complete.")

    # Heading Data Plot
    def plot_heading_data(self):
        """Plot heading, heading rate, and the distribution of heading rate."""
        if self.folder:
            print("This plot does not work for folder-level parsing, try a file instead.")
        else:
            fig_hdg,ax_hdg = plt.subplots(3)
            ax_hdg[0].set_xticks([])
            ax_hdg[0].plot(self.general['time'],self.general['heading'])
            ax_hdg[0].set_yticks(range(0,360,90))
            ax_hdg[0].set_ylabel('Heading (°)')
            ax_hdg[0].set_xlabel('Time (s)')

            ax_hdg[1].set_xticks([])
            ax_hdg[1].plot(self.general['time'],self.general['heading_roc'])
            ax_hdg[1].set_ylim(-40,40)
            ax_hdg[1].set_ylabel('Heading rate (°/s)')
            ax_hdg[1].set_xlabel('Time (s)')

            ax_hdg[2].hist(self.general['heading_roc'], range=(-40,40),density=True)
            ax_hdg[2].set_ylabel('Percent (%)')
            ax_hdg[2].set_xlabel('Heading rate (°/s)')

            fig_hdg.suptitle('Heading Data', fontsize=14)
            fig_hdg.tight_layout()

    # Gliding and Climbing Distribution
    def plot_glide_climb_distr(self):
        """Plot distributions for glide-state glide ratio and climb rate, and thermal climb rate."""
        fig_cng,ax_cng = plt.subplots(3)
        ax_cng[0].hist(self.general[(self.general['flight_state']==1) & (self.general['glide_ratio']<14)]['glide_ratio'], density=True)
        ax_cng[0].set_ylabel('Percent (%)')
        ax_cng[0].set_xlabel('Glide ratio (-)')
        ax_cng[0].set_title('Glide state')

        ax_cng[1].hist(self.general[(self.general['flight_state']==1) & (self.general['glide_ratio']<14)]['avg_baro_climb_rate'], density=True)
        ax_cng[1].set_ylabel('Percent (%)')
        ax_cng[1].set_xlabel('Average barometric climb rate (m/s)')
        ax_cng[1].set_title('Glide state')

        ax_cng[2].hist(self.general[self.general['flight_state']==2]['avg_baro_climb_rate'], density=True)
        ax_cng[2].set_ylabel('Percent (%)')
        ax_cng[2].set_xlabel('Average barometric climb rate (m/s)')
        ax_cng[2].set_title('Thermal state')

        fig_cng.suptitle('Glide ratio and climb rate distributions for relevant flight states', fontsize=14)
        fig_cng.tight_layout()

    # Glide Polar Scatterplots
    def plot_glide_polar_scatters(self):
        """Scatter plots of sink rate and glide ratio against ground speed (glide state)."""
        fig_polars,ax_polars = plt.subplots(2)

        ax_polars[0].scatter(self.general[(self.general['flight_state']==1) & (self.general['glide_ratio']<14)]['ground_speed'],self.general[(self.general['flight_state']==1) & (self.general['glide_ratio']<14)]['avg_baro_climb_rate'])
        ax_polars[0].set_ylabel('Avg. bar. sink (m/s)')
        ax_polars[0].set_xlabel('Ground speed (m/s)')

        ax_polars[1].scatter(self.general[(self.general['flight_state']==1) & (self.general['glide_ratio']<14)]['ground_speed'],self.general[(self.general['flight_state']==1) & (self.general['glide_ratio']<14)]['glide_ratio'])
        ax_polars[1].set_ylabel('Glide ratio (-)')
        ax_polars[1].set_xlabel('Ground speed (m/s)')

        fig_polars.suptitle('Glide Polar Scatterplots', fontsize=14)
        fig_polars.tight_layout()

    # Glide Sink Rate Heatmap
    def plot_glide_sink_heatmap(self):
        """2D histogram (heatmap) of sink rate vs ground speed for glide state."""
        heatmap_climb_v_gspeed, xedges_climb_v_gspeed, yedges_climb_v_gspeed = np.histogram2d(self.general[(self.general['flight_state']==1) & (self.general['glide_ratio']<20)]['ground_speed'],self.general[(self.general['flight_state']==1) & (self.general['glide_ratio']<20)]['avg_baro_climb_rate'], bins=50, density=True)

        fig_raw_polar_srate = plt.hist2d(self.general[(self.general['flight_state']==1) & (self.general['glide_ratio']<20)]['ground_speed'],self.general[(self.general['flight_state']==1) & (self.general['glide_ratio']<20)]['avg_baro_climb_rate'], bins=50, cmap='viridis', density=True)

        plt.colorbar(label='Percentage Density (%)')
        plt.title('Glide Polar Scatterplot, Sink Rate', fontsize=14)
        plt.xlabel('Ground speed (m/s)')
        plt.ylabel('Avg. bar. sink (m/s)')

    # Glide Ratio Heatmap
    def plot_glide_ratio_heatmap(self):
        """2D histogram (heatmap) of glide ratio vs ground speed for glide state."""
        heatmap_climb_v_gratio, xedges_climb_v_gratio, yedges_climb_v_gratio = np.histogram2d(self.general[(self.general['flight_state']==1) & (self.general['glide_ratio']<14)]['ground_speed'],self.general[(self.general['flight_state']==1) & (self.general['glide_ratio']<14)]['glide_ratio'], bins=50, density=True)

        fig_raw_polar_gratio = plt.hist2d(self.general[(self.general['flight_state']==1) & (self.general['glide_ratio']<14)]['ground_speed'],self.general[(self.general['flight_state']==1) & (self.general['glide_ratio']<14)]['glide_ratio'], bins=50, cmap='viridis', density=True)

        plt.colorbar(label='Percentage Density (%)')
        plt.title('Glide Polar Scatterplot, Glide Ratio', fontsize=14)
        plt.xlabel('Ground speed (m/s)')
        plt.ylabel('Glide ratio (-)')

    # Ground Speed Histogram
    def plot_ground_speed_hist(self):
        """Histogram of ground speed in glide state with light filtering."""
        fig_gspd,ax_gspd = plt.subplots()

        ax_gspd.hist(self.general[(self.general['flight_state']==1) & (self.general['glide_ratio']<14) & (self.general['ground_speed']<16)]['ground_speed'],bins=16,density=True)
        ax_gspd.set_ylabel('Percent (%)')
        ax_gspd.set_xlabel('Ground speed (m/s)')

        fig_gspd.suptitle('Ground speed distribution for glide state', fontsize=14)
        fig_gspd.tight_layout()

    # Plot 3D Track
    def plot_3d_track(self):
        """Interactive 3D track (lon/lat/alt) with flight_state as marker color."""
        if self.folder:
            print("This plot does not work for folder-level parsing, try a file instead.")
        else:
            fig_track = go.Figure()

            fig_track.add_trace(
                go.Scatter3d(
                    x=self.general['longitude'],  # Longitude on X-axis
                    y=self.general['latitude'],   # Latitude on Y-axis
                    z=self.general['gps_altitude'],   # Altitude on Z-axis
                    # mode='lines+markers',  # Show both line and markers
                    marker=dict(size=3, color=self.general['flight_state'], colorscale='Inferno', showscale=True),
                    # line=dict(color='blue', width=2),
                    name='Flight Track'
                )
            )

            # Update layout for better visualization
            fig_track.update_layout(
                scene=dict(
                    xaxis_title='Longitude',
                    yaxis_title='Latitude',
                    zaxis_title='Altitude (m)'
                ),
                title='3D IGC Track Plot',
                margin=dict(l=0, r=0, b=0, t=40)
            )

            # Show the plot
            fig_track.show()

    # Climb Rate vs. Rate of Turn Heatmap, Global
    def plot_climb_rot_heatmap_global(self):
        """Heatmap of |avg turn rate| vs avg barometric climb rate across all thermal-state fixes."""
        heatmap_climb_v_rot, xedges_climb_v_rot, yedges_climb_v_rot = np.histogram2d(self.general[(self.general['flight_state']==2)]['avg_heading_roc'].abs(),self.general[(self.general['flight_state']==2)]['avg_baro_climb_rate'], bins=50, density=True)

        fig_raw_thermal = plt.hist2d(self.general[(self.general['flight_state']==2)]['avg_heading_roc'].abs(),self.general[(self.general['flight_state']==2)]['avg_baro_climb_rate'], bins=50, cmap='viridis', density=True)

        plt.colorbar(label='Percentage Density (%)')
        plt.title('Thermal Distribution Heatmap, global thermal state basis', fontsize=14)
        plt.xlabel('Average Rate of Turn (°/s)')
        plt.ylabel('Climb Rate (m/s)')

    # Rate of Turn Histogram, Global
    def plot_rot_rate_hist_global(self):
        """Histogram of |avg heading rate of change| for thermal-state fixes."""
        fig_roc_glob_th,ax_roc_glob_th = plt.subplots()

        ax_roc_glob_th.hist(self.general[(self.general['flight_state']==2)]['avg_heading_roc'].abs(),density=True)
        ax_roc_glob_th.set_ylabel('Percent (%)')
        ax_roc_glob_th.set_xlabel('Heading Rate of Change (°/s)')

        fig_roc_glob_th.suptitle('Heading rate of change, global thermal state basis', fontsize=14)
        fig_roc_glob_th.tight_layout()

    # Rate of Turn Histogram per Thermal
    def plot_rot_rate_hist_per_thermal(self):
        """Histogram of |avg turn rate| computed on a per-thermal aggregate basis."""
        fig_roc_th,ax_roc_th = plt.subplots()

        ax_roc_th.hist(self.thermals['avg_turn_rate'].abs(),density=True)
        ax_roc_th.set_ylabel('Percent (%)')
        ax_roc_th.set_xlabel('Heading Rate of Change (°/s)')

        fig_roc_th.suptitle('Average heading rate of change on a per thermal basis', fontsize=14)
        fig_roc_th.tight_layout()

    # Turn Direction Distribution per Thermal
    def plot_turn_dir_distr_per_thermal(self):
        """Distribution of turn direction sign per thermal (+1 right, −1 left)."""
        fig_roc_dir_th,ax_roc_dir_th = plt.subplots()

        ax_roc_dir_th.hist(self.thermals['turn_direction'],bins=2,density=True)
        ax_roc_dir_th.set_ylabel('Percent (%)')
        ax_roc_dir_th.set_xlabel('Turn Direction (Positive: Right, Negative: Left)')

        fig_roc_dir_th.suptitle('Turn direction distribution on a per thermal basis', fontsize=14)
        fig_roc_dir_th.tight_layout()

    # Climb Rate vs. Rate of Turn Heatmap per Thermal
    def plot_climb_rot_heatmap_per_thermal(self):
        """Heatmap of |avg turn rate| vs avg climb rate using per-thermal aggregates."""
        heatmap_climb_v_rot, xedges_climb_v_rot, yedges_climb_v_rot = np.histogram2d(self.thermals['avg_turn_rate'].abs(),self.thermals['avg_baro_climb_rate'], bins=20, density=True)

        fig_procesed_thermal = plt.hist2d(self.thermals['avg_turn_rate'].abs(),self.thermals['avg_gps_climb_rate'], bins=20, cmap='viridis', density=True)

        plt.colorbar(label='Percentage Density (%)')
        plt.title('Thermal Distribution Heatmap, per thermal basis', fontsize=14)
        plt.xlabel('Average Rate of Turn (°/s)')
        plt.ylabel('Climb Rate (m/s)')

    # Climb Rate vs. Rate of Turn Scatterplot per Thermal
    def plot_climb_rot_scatter_per_thermal(self):
        """Scatter plots of per-thermal avg climb rates (GPS & baro) vs |avg turn rate|."""
        fig_climb_scatter,ax_climb_scatter = plt.subplots(2)

        ax_climb_scatter[0].scatter(self.thermals['avg_turn_rate'].abs(),self.thermals['avg_gps_climb_rate'])
        ax_climb_scatter[0].set_ylabel('Avg. GPS Climb Rate (m/s)')
        ax_climb_scatter[0].set_xlabel('Average Rate of Turn (°/s)')

        ax_climb_scatter[1].scatter(self.thermals['avg_turn_rate'].abs(),self.thermals['avg_baro_climb_rate'])
        ax_climb_scatter[1].set_ylabel('Avg. Baro. Climb Rate (m/s)')
        ax_climb_scatter[1].set_xlabel('Average Rate of Turn (°/s)')

        fig_climb_scatter.suptitle('Climb Rate (m/s) vs. Rate of Turn (°/s), per thermal basis', fontsize=14)
        fig_climb_scatter.tight_layout()

    # Glide Ratio and Climb Rate Counts
    def plot_glide_climb_counts(self):
        """Bar charts of sample counts per ground-speed bin for glide ratio and climb rate."""
        # Create a bar plot for glide ratio and climb rate counts
        fig_speed_bin_counts, ax_speed_bin_counts = plt.subplots(2)

        ax_speed_bin_counts[0].bar(self.speed_bins[:-1], self.glide_ratio_counts, width=1.0, align='edge')
        ax_speed_bin_counts[0].set_ylabel('Count')
        ax_speed_bin_counts[0].set_xlabel('Ground Speed (m/s)')
        ax_speed_bin_counts[0].set_title('Glide Ratio Counts')

        ax_speed_bin_counts[1].bar(self.speed_bins[:-1], self.climb_rate_counts, width=1.0, align='edge')
        ax_speed_bin_counts[1].set_ylabel('Count')
        ax_speed_bin_counts[1].set_xlabel('Ground Speed (m/s)')
        ax_speed_bin_counts[1].set_title('Climb Rate Counts')

        fig_speed_bin_counts.suptitle('Glide Ratio and Climb Rate Counts per Ground Speed Bin', fontsize=14)
        fig_speed_bin_counts.tight_layout()

    # Glide Ratio per Speed Bin Histograms
    def plot_speed_bin_glide_ratio_hists(self):
        """Small multiples of glide-ratio histograms per ground-speed bin."""
        fig_speed_bin_gr_hist, ax_speed_bin_gr_hist = plt.subplots(5, 4, figsize=(20, 15))

        for i in range(len(self.speed_bins) - 1):
            ax_speed_bin_gr_hist[i // 4, i % 4].hist(self.gliding[(self.gliding['ground_speed'] >= self.speed_bins[i]) & (self.gliding['ground_speed'] < self.speed_bins[i + 1])]['glide_ratio'], bins=20, density=True)
            ax_speed_bin_gr_hist[i // 4, i % 4].set_title(f'Ground Speed: {self.speed_bins[i]:.2f} - {self.speed_bins[i + 1]:.2f} m/s')
            ax_speed_bin_gr_hist[i // 4, i % 4].set_ylabel('Percent (%)')
            ax_speed_bin_gr_hist[i // 4, i % 4].set_xlabel('Glide Ratio (-)')
            ax_speed_bin_gr_hist[i // 4, i % 4].set_xlim(0, 20)

        fig_speed_bin_gr_hist.suptitle('Glide Ratio Histograms per Ground Speed Bin', fontsize=14)
        fig_speed_bin_gr_hist.tight_layout()

    # Baro Climb Rate per Speed Bin Histograms
    def plot_speed_bin_baro_climb_hists(self):
        """Small multiples of barometric climb-rate histograms per ground-speed bin."""
        fig_speed_bin_baro_hist, ax_speed_bin_baro_hist_climb = plt.subplots(5, 4, figsize=(20, 15))

        for i in range(len(self.speed_bins) - 1):
            ax_speed_bin_baro_hist_climb[i // 4, i % 4].hist(self.gliding[(self.gliding['ground_speed'] >= self.speed_bins[i]) & (self.gliding['ground_speed'] < self.speed_bins[i + 1])]['avg_baro_climb_rate'], bins=20, density=True)
            ax_speed_bin_baro_hist_climb[i // 4, i % 4].set_title(f'Ground Speed: {self.speed_bins[i]:.2f} - {self.speed_bins[i + 1]:.2f} m/s')
            ax_speed_bin_baro_hist_climb[i // 4, i % 4].set_ylabel('Percent (%)')
            ax_speed_bin_baro_hist_climb[i // 4, i % 4].set_xlabel('Barometric Climb Rate (m/s)')
            ax_speed_bin_baro_hist_climb[i // 4, i % 4].set_xlim(-5, 5)

        fig_speed_bin_baro_hist.suptitle('Barometric Climb Rate Histograms per Ground Speed Bin', fontsize=14)
        fig_speed_bin_baro_hist.tight_layout()

    # GPS Climb Rate per Speed Bin Histograms
    def plot_speed_bin_gps_climb_hists(self):
        """Small multiples of GPS climb-rate histograms per ground-speed bin."""
        fig_speed_bin_gps_hist, ax_speed_bin_gps_hist_climb = plt.subplots(5, 4, figsize=(20, 15))

        for i in range(len(self.speed_bins) - 1):
            ax_speed_bin_gps_hist_climb[i // 4, i % 4].hist(self.gliding[(self.gliding['ground_speed'] >= self.speed_bins[i]) & (self.gliding['ground_speed'] < self.speed_bins[i + 1])]['avg_gps_climb_rate'], bins=20, density=True)
            ax_speed_bin_gps_hist_climb[i // 4, i % 4].set_title(f'Ground Speed: {self.speed_bins[i]:.2f} - {self.speed_bins[i + 1]:.2f} m/s')
            ax_speed_bin_gps_hist_climb[i // 4, i % 4].set_ylabel('Percent (%)')
            ax_speed_bin_gps_hist_climb[i // 4, i % 4].set_xlabel('GPS Climb Rate (m/s)')
            ax_speed_bin_gps_hist_climb[i // 4, i % 4].set_xlim(-5, 5)

        fig_speed_bin_gps_hist.suptitle('GPS Climb Rate Histograms per Ground Speed Bin', fontsize=14)
        fig_speed_bin_gps_hist.tight_layout()

    # Box Charts for Glide Ratios per Ground Speed Bin
    def plot_ground_speed_bin_gr_boxes(self):
        """Box plots of glide ratio per ground-speed bin (positions at bin medians)."""
        # Plot box charts for each Ground Speed Median in the speed_bin_stats dataframe - Glide Ratio
        fig_speed_bin_gr_box, ax_speed_bin_gr_box = plt.subplots()

        ax_speed_bin_gr_box.boxplot([self.gliding[(self.gliding['ground_speed'] >= self.speed_bins[i]) & (self.gliding['ground_speed'] < self.speed_bins[i + 1])]['glide_ratio'] for i in range(len(self.speed_bins) - 1)], positions=self.speed_bin_stats['Ground Speed Median'], widths=0.5)
        ax_speed_bin_gr_box.set_ylabel('Glide Ratio (-)')
        ax_speed_bin_gr_box.set_xlabel('Ground Speed (m/s)')
        ax_speed_bin_gr_box.set_title('Glide Ratio Boxplots per Ground Speed Bin')

        # Make wider so all box plots fit
        fig_speed_bin_gr_box.set_figwidth(15)

        fig_speed_bin_gr_box.tight_layout()

    # Glide Ratio Uncertainties per Ground Speed Bin
    def plot_ground_speed_bin_gr_uncertainties(self):
        """Error-bar plot: mean ± std of glide ratio vs ground-speed bin median."""
        fig_speed_bin_gr_scatter = plt.figure(figsize=(10, 6))

        plt.errorbar(self.speed_bin_stats['Ground Speed Median'], self.speed_bin_stats['Glide Ratio Mean'], yerr=self.speed_bin_stats['Glide Ratio Std Dev'], fmt='o', capsize=5)
        plt.ylabel('Glide Ratio (-)')
        plt.xlabel('Ground Speed (m/s)')
        plt.title('Glide Ratio vs. Ground Speed with Uncertainty')
        plt.grid()

    # Baro Climb Rate Uncertainties per Ground Speed Bin
    def plot_ground_speed_bin_baro_climb_uncertainties(self):
        """Error-bar plot: mean ± std of barometric climb rate vs bin median."""
        fig_speed_bin_bcr_scatter = plt.figure(figsize=(10, 6))

        plt.errorbar(self.speed_bin_stats['Ground Speed Median'], self.speed_bin_stats['Barometric Climb Rate Mean'], yerr=self.speed_bin_stats['Barometric Climb Rate Std Dev'], fmt='o', capsize=5)
        plt.ylabel('Barometric Climb Rate (m/s)')
        plt.xlabel('Ground Speed (m/s)')
        plt.title('Barometric Climb Rate vs. Ground Speed with Uncertainty')

        plt.grid()
        plt.axis('equal')

    # GPS Climb Rate Uncertainties per Ground Speed Bin
    def plot_ground_speed_bin_gps_climb_uncertainties(self):
        """Error-bar plot: mean ± std of GPS climb rate vs ground-speed bin median."""
        fig_speed_bin_gps_scatter = plt.figure(figsize=(10, 6))

        plt.errorbar(self.speed_bin_stats['Ground Speed Median'], self.speed_bin_stats['GPS Climb Rate Mean'], yerr=self.speed_bin_stats['GPS Climb Rate Std Dev'], fmt='o', capsize=5)
        plt.ylabel('GPS Climb Rate (m/s)')
        plt.xlabel('Ground Speed (m/s)')
        plt.title('GPS Climb Rate vs. Ground Speed with Uncertainty')

        plt.grid()
        plt.axis('equal')

    # Probability Distribution Function for Glide Ratio
    def plot_PDF_glide_ratio(self):
        """Probability density function (histogram) of glide ratio in glide data."""
        fig_gr_pdf, ax_gr_pdf = plt.subplots()

        ax_gr_pdf.hist(self.gliding['glide_ratio'], bins=20, density=True)
        ax_gr_pdf.set_ylabel('Density')
        ax_gr_pdf.set_xlabel('Glide Ratio (-)')
        ax_gr_pdf.set_title('Glide Ratio Probability Density Function')

        fig_gr_pdf.tight_layout()

    # Glide Success Threshold (Inverse CDF)
    def plot_glide_success_threshold(self):
        """
        Inverse cumulative histogram (approx. 1−CDF) of glide ratio.

        The curve can be read as the probability that a randomly sampled
        glide has glide_ratio ≥ threshold.
        """
        fig_gr_icdf, ax_gr_icdf = plt.subplots()

        ax_gr_icdf.hist(self.gliding['glide_ratio'], bins=1000, density=True, cumulative=-1, histtype='step')
        ax_gr_icdf.set_ylabel('Probability of Successful Glide')
        ax_gr_icdf.set_xlabel('Instrument Glide Ratio Threshold (-)')
        ax_gr_icdf.set_title('Glide Success Threshold - Cumulative Density Function')

        plt.grid()
        fig_gr_icdf.tight_layout()

    # General parsing of the IGC file to calculate derived data series of relevance (e.g. climb rate, glide ratio etc.)
    def __general_parse(self):
        """
        Parse an IGC file into a per-fix DataFrame and compute derived signals.

        Returns
        -------
        pd.DataFrame
            Columns include: time, latitude, longitude, gps_altitude, baro_altitude,
            ground_speed, heading, heading_roc, gps_climb_rate, baro_climb_rate,
            avg_* rolling features, flight_state, glide_ratio.
        """
        # Open IGC file
        with open(self.file, 'r') as f:
            lines = f.readlines()
        
        # Read each B-record in the IGC file and extract the corresponding information
        fixes = []
        for line in lines:
            if line.startswith("B"):
                # IGC B-record slicing per spec; simplest subset used here
                timestamp = line[1:7]
                lat = self.__igc_to_decimal(line[7:14], line[14:15])
                lon = self.__igc_to_decimal(line[15:23], line[23:24])
                gps_alt = line[25:30]
                bar_alt = line[30:35]
                
                fixes.append([timestamp, lat, lon, gps_alt, bar_alt])
        
        # Create a DataFrame from the extracted raw information
        df = pd.DataFrame(fixes, columns=["time", 
                                        "latitude", 
                                        "longitude",  
                                        "gps_altitude", 
                                        "baro_altitude"
                                        ])
        # Ensure numeric dtype for downstream math
        df["latitude"] = df["latitude"].astype(float)
        df["longitude"] = df["longitude"].astype(float)
        df["gps_altitude"] = df["gps_altitude"].astype(float)
        df["baro_altitude"] = df["baro_altitude"].astype(float)

        # Calculate ground speed, heading, and rate of change of heading
        # Note: access next row via index arithmetic; last row falls back to itself
        df['ground_speed'] = df.apply(lambda row: self.__calculate_ground_speed(row['latitude'], row['longitude'],
                                                                        df.loc[row.name + 1, 'latitude'] if row.name + 1 < len(df) else row['latitude'],
                                                                        df.loc[row.name + 1, 'longitude'] if row.name + 1 < len(df) else row['longitude']
                                                                        ), axis=1
                                ).astype(float)
        df['heading'] = df.apply(lambda row: self.__calculate_heading(row['latitude'], row['longitude'],
                                                            df.loc[row.name + 1, 'latitude'] if row.name + 1 < len(df) else row['latitude'],
                                                            df.loc[row.name + 1, 'longitude'] if row.name + 1 < len(df) else row['longitude']
                                                            ), axis=1
                                ).astype(float)
        df['heading_roc'] = df['heading'].rolling(window=2).apply(self.__heading_delta,raw=False).astype(float)

        # Calculate climb rates (1 Hz assumption from diff of consecutive samples)
        df['gps_climb_rate'] = df['gps_altitude'].diff().astype(float)
        df['baro_climb_rate'] = df['baro_altitude'].diff().astype(float)
        
        # Calculate rolling averages and flight-state/glide-ratio using configured window
        df = pd.concat([df, 
                        pd.DataFrame([self.__calculate_rolling_subparams(df[['ground_speed','heading_roc','gps_climb_rate','baro_climb_rate']].iloc[max(0, i - self.avg_window + 1):i + 1])
                                    if i >= self.avg_window - 1 else (np.nan, np.nan, np.nan)
                                    for i in range(len(df))], 
                        columns=['avg_heading_roc', 'avg_ground_speed', 'avg_gps_climb_rate', 'avg_baro_climb_rate', 'flight_state','glide_ratio'])], 
                        axis=1)

        return df

    # Convert IGC DDDMMmmm or DDMMmmm format to decimal degrees.
    def __igc_to_decimal(self, igc_coordinate, direction):
        """
        Convert IGC coordinate fields to decimal degrees.

        Parameters
        ----------
        igc_coordinate : str
            'DDMMmmm' for latitude or 'DDDMMmmm' for longitude.
        direction : str
            'N'/'S' or 'E'/'W'; used to set sign.

        Returns
        -------
        float
            Decimal degrees with appropriate sign.
        """
        # Extract degrees, minutes, and thousandths of a minute
        if len(igc_coordinate) == 7:  # Latitude: DDMMmmm
            degrees = int(igc_coordinate[:2])
            minutes = int(igc_coordinate[2:4])
            thousandths = int(igc_coordinate[4:])
        elif len(igc_coordinate) == 8:  # Longitude: DDDMMmmm
            degrees = int(igc_coordinate[:3])
            minutes = int(igc_coordinate[3:5])
            thousandths = int(igc_coordinate[5:])
        else:
            raise ValueError("Invalid IGC coordinate format. Must be 7 or 8 characters.")

        # Convert to decimal degrees
        decimal_degrees = float(degrees + (minutes + thousandths / 1000) / 60)

        # Apply negative sign for South or West directions
        if direction in ['S', 'W']:
            decimal_degrees *= -1

        return decimal_degrees
    
    # Heading calculation between two points
    def __calculate_heading(self, lat1, lon1, lat2, lon2):
        """
        Compute initial bearing from (lat1, lon1) to (lat2, lon2) in degrees [0, 360).

        Notes
        -----
        Uses spherical trigonometry; inputs expected in degrees.
        """
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Difference in longitude
        delta_lon = lon2 - lon1
        
        # Calculate initial bearing
        x = math.sin(delta_lon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
        initial_bearing = math.atan2(x, y)
        
        # Convert from radians to degrees
        initial_bearing = math.degrees(initial_bearing)
        
        # Normalize to 0-360
        compass_bearing = (initial_bearing + 360) % 360
        
        return compass_bearing

    # Rate of change of heading calculation
    def __heading_delta(self, rolling_input):
        """
        Two-sample delta heading, handling 0/360° wrap-around in both turn directions.

        Parameters
        ----------
        rolling_input : sequence[float]
            Two headings in degrees (previous, current).

        Returns
        -------
        float
            Signed delta in degrees per sample.
        """
        rolling_input = list(rolling_input)

        # Adjust for 360 degree rollover, left turn
        if abs(rolling_input[0]>300) & abs(rolling_input[1]<60):
            return  (360+rolling_input[1]) - rolling_input[0]
        
        # Adjust for 360 degree rollover, right turn
        elif abs(rolling_input[0]<60) & abs(rolling_input[1]>300):
            return  (rolling_input[1]-360) - rolling_input[0]
        
        else :
            return rolling_input[1] - rolling_input[0]
    
    # Ground speed based on lat/lon
    def __calculate_ground_speed(self, lat1, lon1, lat2, lon2, time_diff_seconds=1):
        """
        Compute ground speed (m/s) from two positions using the haversine distance.

        Parameters
        ----------
        lat1, lon1, lat2, lon2 : float
            Coordinates in degrees.
        time_diff_seconds : float, default 1
            Time between samples in seconds.

        Returns
        -------
        float
            Ground speed in m/s (0 if non-positive time interval).
        """
        # Radius of Earth in meters
        R = 6371000  

        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # Calculate differences
        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1

        # Haversine formula
        a = math.sin(delta_lat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance_meters = R * c  # Distance in meters

        # Calculate ground speed (distance / time)
        ground_speed = distance_meters / time_diff_seconds if time_diff_seconds > 0 else 0

        return ground_speed

    # Calculate subparameters with the rolling average window (average climb rates, glide ratio, and flight states)
    def __calculate_rolling_subparams(self, window_data):
        """
        Compute rolling averages and classify flight state; derive glide ratio.

        Parameters
        ----------
        window_data : pd.DataFrame
            Window slice with columns: ground_speed, heading_roc, gps_climb_rate,
            baro_climb_rate.

        Returns
        -------
        tuple
            (avg_heading_roc, avg_ground_speed, avg_gps_climb_rate,
             avg_baro_climb_rate, flight_state, glide_ratio)

        Notes
        -----
        Flight state:
          - 2 (Thermal): avg_baro_climb_rate > 0 and |avg_heading_roc| >= 10
          - 1 (Glide):   |avg_heading_roc| < 10 and avg_ground_speed > 3
          - 0 (Other):   otherwise
        Glide ratio is |Vg / Vz| for avg_gps_climb_rate < 0; else a large sentinel.
        """
        # Average climb rates
        avg_gps_climb_rate = window_data['gps_climb_rate'].mean()
        avg_baro_climb_rate = window_data['baro_climb_rate'].mean()

        # Flight state calculations and conditional logic
        avg_ground_speed = window_data['ground_speed'].mean()
        avg_heading_roc = window_data['heading_roc'].mean()
        abs_avg_heading_roc = abs(avg_heading_roc)

        if avg_baro_climb_rate > 0 and abs_avg_heading_roc >= 10 :
            flight_state = 2.0  # Thermal
        elif abs_avg_heading_roc < 10 and avg_ground_speed > 3 :
            flight_state = 1.0  # Glide
        else:
            flight_state = 0.0  # Undefined
        
        # Calculate glide ratio
        glide_ratio = abs(avg_ground_speed / avg_gps_climb_rate) if avg_gps_climb_rate < 0 else 1000000

        return avg_heading_roc, avg_ground_speed, avg_gps_climb_rate, avg_baro_climb_rate, flight_state, glide_ratio

    # Thermal Dataframe Parsing
    def __thermal_parsing (self) :
        """
        Aggregate consecutive thermal-state fixes into per-thermal rows.

        Rules
        -----
        - Rows are grouped while indices remain consecutive and turn direction
          (sign of heading_roc) does not flip.
        - A completed thermal row is recorded only if:
            height_gain > 0, turn_count > 1 (≥360°), and |avg_turn_rate| > 10.
        """
        df = self.general

        # id, height_gain, turn count, avg_turn_rate, turn direction, avg_climb_rate, 
        thermal_calc_df = df[df['flight_state']==2][['gps_altitude','baro_altitude', 'heading','heading_roc','gps_climb_rate','baro_climb_rate',]]

        # Initialize empty dataframe & initialize running variables
        thermal_df = pd.DataFrame(columns=['thermal_id','height_gain','turn_count','avg_turn_rate','turn_direction','avg_gps_climb_rate','avg_baro_climb_rate'])
        thermal_id = 0
        thermal_ctr = 0
        running_sum_baro_climb_rate_thermal = 0
        running_sum_gps_climb_rate_thermal = 0
        running_sum_hdg_thermal = 0
        running_sum_hdg_roc_thermal = 0
        alt_init_thermal = thermal_calc_df['gps_altitude'].iloc[0]
        thermal_rows = []

        # Iterate through rows in thermal_calc_df 
        for current_row, next_row in zip(thermal_calc_df.itertuples(index=True), thermal_calc_df.iloc[1:].itertuples(index=True, name='NextRow')):
            # Check if the next row is part of the same thermal, otherwise start new thermal
            if (next_row.Index - current_row.Index == 1) :
                # Check if the next row is part of the same turn direction, otherwise start new thermal
                if (current_row.heading_roc * next_row.heading_roc > 0) :
                    # Running sum updates
                    running_sum_baro_climb_rate_thermal += current_row.baro_climb_rate
                    running_sum_gps_climb_rate_thermal += current_row.gps_climb_rate
                    running_sum_hdg_thermal += current_row.heading
                    running_sum_hdg_roc_thermal += current_row.heading_roc 
                    thermal_ctr += 1

                    # Continue to the next iteration of the loop
                    continue

            # New thermal is started if neither of the above conditions is met
            # Running sum updates
            running_sum_baro_climb_rate_thermal += current_row.baro_climb_rate
            running_sum_gps_climb_rate_thermal += current_row.gps_climb_rate
            running_sum_hdg_thermal += current_row.heading
            running_sum_hdg_roc_thermal += current_row.heading_roc 
            thermal_ctr += 1

            # Value calculations
            height_gain = current_row.gps_altitude - alt_init_thermal
            turn_count = running_sum_hdg_thermal / 360
            avg_turn_rate = running_sum_hdg_roc_thermal / thermal_ctr
            turn_direction = np.sign(avg_turn_rate)
            avg_gps_climb_rate = running_sum_gps_climb_rate_thermal / thermal_ctr
            avg_baro_climb_rate = running_sum_baro_climb_rate_thermal / thermal_ctr

            # Check if thermal gain is positive and more than on turn to concatenate thermal row
            if height_gain > 0 and turn_count > 1 and abs(avg_turn_rate) > 10:
                thermal_rows.append({
                    'thermal_id': thermal_id,
                    'height_gain': height_gain,
                    'turn_count': turn_count,
                    'avg_turn_rate': avg_turn_rate,
                    'turn_direction': turn_direction,
                    'avg_gps_climb_rate': avg_gps_climb_rate,
                    'avg_baro_climb_rate': avg_baro_climb_rate
                })    
            thermal_df = pd.DataFrame(thermal_rows)

            # Counter updates and resets
            thermal_id += 1
            thermal_ctr = 0
            alt_init_thermal = next_row.gps_altitude
            running_sum_hdg_thermal = 0
            running_sum_hdg_roc_thermal = 0
            running_sum_baro_climb_rate_thermal = 0
            running_sum_gps_climb_rate_thermal = 0

        return thermal_df
    
    # Speed-bin Dataframe Parsing
    def __glide_polar_parsing (self) :
        """
        Build glide-state subset and compute per-bin statistics/counters.

        Returns
        -------
        tuple
            (gliding_df, speed_bin_stats_df, glide_ratio_counts, climb_rate_counts)

        Notes
        -----
        The temporary `gliding_df` is filtered by:
          - flight_state == 1
          - glide_ratio < self.LD_max
          - ground_speed < 20 m/s
        """
        df = self.general
        speed_bins = self.speed_bins

        # Temporary DF for glide ratio and climb rate corresponding to glide state
        temp_df =  df[(df['flight_state']==1) & (df['glide_ratio']<self.LD_max) & (df['ground_speed']<20)][['ground_speed','glide_ratio','avg_baro_climb_rate','avg_gps_climb_rate']]

        # Create a new dataframe where each row is a speed bin and the columns are the means, medians, standard deviations, and IQRs for: glide ratio, barometric climb rate, and GPS climb rate
        speed_bin_stats = pd.DataFrame(index=range(len(speed_bins) - 1), columns=['Speed Bin', 
                                                                                'Glide Ratio Mean', 
                                                                                'Glide Ratio Median',
                                                                                'Glide Ratio Std Dev',
                                                                                'Glide Ratio Q1',
                                                                                'Glide Ratio Q3', 
                                                                                'Glide Ratio Q3',
                                                                                'Glide Ratio IQR', 
                                                                                'Barometric Climb Rate Mean', 
                                                                                'Barometric Climb Rate Median', 
                                                                                'Barometric Climb Rate Std Dev',
                                                                                'Barometric Climb Rate Q1',
                                                                                'Barometric Climb Rate Q3',
                                                                                'Barometric Climb Rate IQR', 
                                                                                'GPS Climb Rate Mean', 
                                                                                'GPS Climb Rate Median', 
                                                                                'GPS Climb Rate Std Dev',
                                                                                'GPS Climb Rate Q1',
                                                                                'GPS Climb Rate Q3',
                                                                                'GPS Climb Rate IQR'])

        for i in range(len(speed_bins) - 1):
            glide_ratio_data = temp_df[(temp_df['ground_speed'] >= speed_bins[i]) & (temp_df['ground_speed'] < speed_bins[i + 1])]['glide_ratio']
            baro_climb_rate_data = temp_df[(temp_df['ground_speed'] >= speed_bins[i]) & (temp_df['ground_speed'] < speed_bins[i + 1])]['avg_baro_climb_rate']
            gps_climb_rate_data = temp_df[(temp_df['ground_speed'] >= speed_bins[i]) & (temp_df['ground_speed'] < speed_bins[i + 1])]['avg_gps_climb_rate']

            speed_bin_stats.loc[i, 'Speed Bin'] = f'{speed_bins[i]:.2f} - {speed_bins[i + 1]:.2f} m/s'
            speed_bin_stats.loc[i, 'Glide Ratio Mean'] = glide_ratio_data.mean()
            speed_bin_stats.loc[i, 'Glide Ratio Median'] = glide_ratio_data.median()
            speed_bin_stats.loc[i, 'Glide Ratio Std Dev'] = glide_ratio_data.std()
            speed_bin_stats.loc[i, 'Glide Ratio Q1'] = glide_ratio_data.quantile(0.25)
            speed_bin_stats.loc[i, 'Glide Ratio Q3'] = glide_ratio_data.quantile(0.75)
            speed_bin_stats.loc[i, 'Glide Ratio IQR'] = glide_ratio_data.quantile(0.75) - glide_ratio_data.quantile(0.25)
            speed_bin_stats.loc[i, 'Barometric Climb Rate Mean'] = baro_climb_rate_data.mean()
            speed_bin_stats.loc[i, 'Barometric Climb Rate Median'] = baro_climb_rate_data.median()
            speed_bin_stats.loc[i, 'Barometric Climb Rate Std Dev'] = baro_climb_rate_data.std()
            speed_bin_stats.loc[i, 'Barometric Climb Rate Q1'] = baro_climb_rate_data.quantile(0.25)
            speed_bin_stats.loc[i, 'Barometric Climb Rate Q3'] = baro_climb_rate_data.quantile(0.75)
            speed_bin_stats.loc[i, 'Barometric Climb Rate IQR'] = baro_climb_rate_data.quantile(0.75) - baro_climb_rate_data.quantile(0.25)
            speed_bin_stats.loc[i, 'GPS Climb Rate Mean'] = gps_climb_rate_data.mean()
            speed_bin_stats.loc[i, 'GPS Climb Rate Median'] = gps_climb_rate_data.median()
            speed_bin_stats.loc[i, 'GPS Climb Rate Std Dev'] = gps_climb_rate_data.std()
            speed_bin_stats.loc[i, 'GPS Climb Rate Q1'] = gps_climb_rate_data.quantile(0.25)
            speed_bin_stats.loc[i, 'GPS Climb Rate Q3'] = gps_climb_rate_data.quantile(0.75)
            speed_bin_stats.loc[i, 'GPS Climb Rate IQR'] = gps_climb_rate_data.quantile(0.75) - gps_climb_rate_data.quantile(0.25)

        # Add a column for the speed bin values which corresponds to the middle of each bin
        speed_bin_stats['Ground Speed Median'] = (speed_bins[:-1] + speed_bins
                                        [1:]) / 2
        
        # For each speed bin between 6.5 and 17.5 m/s, extract the number of non-NaN data points for glide ratio and climb rate
        glide_ratio_counts = []
        climb_rate_counts = []

        for i in range(len(speed_bins) - 1):
            glide_ratio_counts.append(temp_df[(temp_df['ground_speed'] >= speed_bins[i]) & (temp_df['ground_speed'] < speed_bins[i + 1])]['glide_ratio'].count())
            climb_rate_counts.append(temp_df[(temp_df['ground_speed'] >= speed_bins[i]) & (temp_df['ground_speed'] < speed_bins[i + 1])]['avg_baro_climb_rate'].count())

        return temp_df, speed_bin_stats, glide_ratio_counts, climb_rate_counts