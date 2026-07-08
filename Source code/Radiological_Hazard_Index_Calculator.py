"""
Radiological Hazard Index Calculator
=====================================

A comprehensive tool for assessing radiological hazard indices from natural radionuclides
(Ra-226, Th-232, K-40) in environmental samples with full uncertainty propagation.


Key Indices Computed:
    - Radium Equivalent (Rad_Eq): Ra + 1.43*Th + 0.077*K [Bq/kg]
    - External Hazard Index (Hex): Ra/370 + Th/259 + K/4810 [unitless]
    - Internal Hazard Index (Hin): Ra/185 + Th/259 + K/4810 [unitless]
    - Dose Rate: 0.462*Ra + 0.604*Th + 0.0417*K [nGy/h]
    - Annual Effective Dose Equivalent (AEDE): Dose_Rate × 8760 × 0.2 × 10⁻⁶ [mSv/y]

Safety Thresholds (WHO/IAEA Standards):
    - Dose Rate: ≤ 59 nGy/h
    - Radium Equivalent: ≤ 370 Bq/kg
    - External Hazard Index: ≤ 1
    - Internal Hazard Index: ≤ 1
    - Annual Effective Dose: ≤ 1 mSv/y

Author: Tumusiime Hamphrey
Institution: Kyambogo University
Date: 20/06/2026
"""
#import all the necessary libraries
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout,
    QFileDialog, QTableWidget, QTableWidgetItem,
    QGroupBox, QProgressBar, QSplitter,
    QHeaderView, QComboBox, QTabWidget
)

# COMPUTATION ENGINE (carries all the necessary calculations for the indices)

def compute_indices(df):
    """
    Compute hazard indices with uncertainty propagation.
    Returns DataFrame with value and uncertainty in separate columns.
    """
    df = df.copy()
    
    # Extract uncertainties (default 0 if not present for better error handling)
    dRa = df.get("dRa", pd.Series([0] * len(df)))
    dTh = df.get("dTh", pd.Series([0] * len(df)))
    dK = df.get("dK", pd.Series([0] * len(df)))
    
    # Calculate values
    rad_eq = df["Ra"] + 1.43 * df["Th"] + 0.077 * df["K"]
    hex_val = df["Ra"]/370 + df["Th"]/259 + df["K"]/4810
    hin_val = df["Ra"]/185 + df["Th"]/259 + df["K"]/4810
    dose_rate = 0.462*df["Ra"] + 0.604*df["Th"] + 0.0417*df["K"]
    aede = dose_rate * 8760 * 0.2 * 1e-6
    
    # Propagate uncertainties
    dRadEq = np.sqrt((1*dRa)**2 + (1.43*dTh)**2 + (0.077*dK)**2)
    dHex = np.sqrt((dRa/370)**2 + (dTh/259)**2 + (dK/4810)**2)
    dHin = np.sqrt((dRa/185)**2 + (dTh/259)**2 + (dK/4810)**2)
    dDose = np.sqrt((0.462*dRa)**2 + (0.604*dTh)**2 + (0.0417*dK)**2)
    dAEDE = dDose * 8760 * 0.2 * 1e-6
    
    return pd.DataFrame({
        "Rad_Eq": rad_eq, "dRad_Eq": dRadEq,
        "Hex": hex_val, "dHex": dHex,
        "Hin": hin_val, "dHin": dHin,
        "Dose_Rate": dose_rate, "dDose_Rate": dDose,
        "AEDE": aede, "dAEDE": dAEDE
    })

# =========================================================
# MAIN APPLICATION - BATCH PROCESSING ONLY
# =========================================================

class MainApp(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Radiological Hazard Assessment Tool")
        self.resize(1400, 800)

        self.data = None
        self.results = None

        # Main layout
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)

        # ---------------------------------------------
        # LEFT PANEL - CONTROLS
        # ---------------------------------------------
        left_panel = QWidget()
        left_panel.setFixedWidth(350)
        left_layout = QVBoxLayout(left_panel)

        # Title
        title = QLabel("Batch Processing")
        title.setObjectName("title")
        left_layout.addWidget(title)

        # ---------------------------------------------
        # DATA INPUT
        # ---------------------------------------------
        input_group = QGroupBox("Data Input")
        input_layout = QVBoxLayout()
        input_group.setLayout(input_layout)

        file_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load CSV")
        self.load_btn.clicked.connect(self.load_csv)
        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.load_btn)
        file_layout.addWidget(self.file_label)
        input_layout.addLayout(file_layout)

        self.sample_info_label = QLabel("Samples: 0")
        input_layout.addWidget(self.sample_info_label)

        left_layout.addWidget(input_group)

        # ---------------------------------------------
        # CSV FORMAT INFO
        # ---------------------------------------------
        info_group = QGroupBox("CSV Format")
        info_layout = QVBoxLayout()
        info_group.setLayout(info_layout)

        info_text = QLabel(
            "Required columns:\n"
            "Ra, Th, K (activity concentrations)\n"
            "dRa, dTh, dK (uncertainties)\n\n"
            "If uncertainties are not provided,\n"
            "they will be set to zero."
        )
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)

        left_layout.addWidget(info_group)

        # ---------------------------------------------
        # ACTIONS
        # ---------------------------------------------
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout()
        action_group.setLayout(action_layout)

        self.process_btn = QPushButton("⚡ Process Samples")
        self.process_btn.clicked.connect(self.compute_all)
        self.process_btn.setEnabled(False)
        action_layout.addWidget(self.process_btn)

        self.progress_bar = QProgressBar()
        action_layout.addWidget(self.progress_bar)

        # Export buttons
        export_layout = QHBoxLayout()
        self.export_btn = QPushButton("Export CSV")
        self.export_btn.clicked.connect(self.save_results)
        self.export_btn.setEnabled(False)

        self.save_plot_btn = QPushButton("Save Plot")
        self.save_plot_btn.clicked.connect(self.save_plot)
        self.save_plot_btn.setEnabled(False)

        export_layout.addWidget(self.export_btn)
        export_layout.addWidget(self.save_plot_btn)
        action_layout.addLayout(export_layout)

        left_layout.addWidget(action_group)

        # ---------------------------------------------
        # PLOT OPTIONS
        # ---------------------------------------------
        plot_group = QGroupBox("Plot Options")
        plot_layout = QVBoxLayout()
        plot_group.setLayout(plot_layout)

        # Plot view selector
        view_layout = QHBoxLayout()
        view_layout.addWidget(QLabel("View:"))
        self.plot_view = QComboBox()
        self.plot_view.addItems([
            "Combined (Ra + Th + K)",
            "Ra Only",
            "Th Only",
            "K Only"
        ])
        self.plot_view.currentTextChanged.connect(self.update_plot)
        view_layout.addWidget(self.plot_view)
        plot_layout.addLayout(view_layout)

        self.plot_btn = QPushButton("Generate Plot")
        self.plot_btn.clicked.connect(self.update_plot)
        self.plot_btn.setEnabled(False)
        plot_layout.addWidget(self.plot_btn)

        left_layout.addWidget(plot_group)

        # ---------------------------------------------
        # THRESHOLD INFORMATION
        # ---------------------------------------------
        info_group = QGroupBox("Safety Thresholds")
        info_layout = QVBoxLayout()
        info_group.setLayout(info_layout)

        thresholds_text = QLabel(
            "Radium Equivalent: ≤ 370 Bq/kg\n"
            "External Hazard (Hex): ≤ 1\n"
            "Internal Hazard (Hin): ≤ 1\n"
            "Dose Rate: ≤ 59 nGy/h\n"
            "Annual Effective Dose: ≤ 1 mSv/y"
        )
        thresholds_text.setWordWrap(True)
        info_layout.addWidget(thresholds_text)

        left_layout.addWidget(info_group)

        # Status bar at bottom of left panel
        self.status = QLabel("Ready - Load CSV data to begin")
        self.status.setObjectName("status")
        left_layout.addWidget(self.status)

        left_layout.addStretch()

        # ---------------------------------------------
        # RIGHT PANEL - RESULTS WITH TABS
        # ---------------------------------------------
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Create tab widget for plot and table
        self.tabs = QTabWidget()

        # ----- PLOT TAB -----
        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)
        
        self.figure = Figure(figsize=(10, 7))
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        
        self.tabs.addTab(plot_tab, "Correlation Plot")

        # ----- TABLE TAB -----
        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)
        
        self.table = QTableWidget()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table_layout.addWidget(self.table)
        
        self.tabs.addTab(table_tab, "Results Table")

        right_layout.addWidget(self.tabs)

        # Set splitter sizes
        self.layout.addWidget(left_panel)
        self.layout.addWidget(right_panel)

        # Apply styles
        self.apply_style()

    # LOAD DATA
    
    def load_csv(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV File",
            "",
            "CSV Files (*.csv)"
        )

        if not path:
            return

        try:
            df = pd.read_csv(path)

            required = ["Ra", "Th", "K", "dRa", "dTh", "dK"]

            # Check for required columns, add zeros if missing
            missing = []
            for col in required:
                if col not in df.columns:
                    df[col] = 0.0
                    missing.append(col)

            if missing:
                self.status.setText(f" Columns not found, set to zero: {', '.join(missing)}")
                self.status.setStyleSheet("color: orange;")
            else:
                self.status.setText(f"Loaded {len(df)} samples")
                self.status.setStyleSheet("color: lightgreen;")

            self.data = df[required]
            self.file_label.setText(f"Loaded: {path.split('/')[-1]}")
            self.sample_info_label.setText(f"Samples: {len(self.data)}")
            self.process_btn.setEnabled(True)

        except Exception as e:
            self.status.setText(" Error loading CSV")
            self.status.setStyleSheet("color: red;")
            print(e)

    # COMPUTE ALL
    # =====================================================
    def compute_all(self):

        if self.data is None:
            self.status.setText("No data loaded")
            return

        try:
            self.process_btn.setEnabled(False)
            self.progress_bar.setValue(0)

            df = self.data.copy() #obtaining a copy of the loadd data

            df["Sample"] = [
                f"SAMPLE_{i+1:03d}"
                for i in range(len(df))
            ]

            indices = compute_indices(df[["Ra", "Th", "K", "dRa", "dTh", "dK"]])

            # Round values (AEDE gets 6 dp, others 3 dp)
            for col in indices.columns:
                if "AEDE" in col:
                    indices[col] = indices[col].round(6)
                else:
                    indices[col] = indices[col].round(3)

            indices.insert(0, "Sample", df["Sample"])

            indices["Status"] = indices.apply(
                lambda row:
                "Safe"
                if (
                    row["Rad_Eq"] <= 370 and
                    row["Hex"] <= 1 and
                    row["Hin"] <= 1 and
                    row["Dose_Rate"] <= 59 and
                    row["AEDE"] <= 1
                )
                else "Unsafe",
                axis=1
            )

            self.results = indices

            # -----------------------------------------
            # TABLE DISPLAY
            # -----------------------------------------
            self.table.setRowCount(len(indices))
            self.table.setColumnCount(len(indices.columns))
            self.table.setHorizontalHeaderLabels(indices.columns)
        #giving green and red colurs to cels with safe/ unsafe status
            for i in range(len(indices)):
                for j in range(len(indices.columns)):

                    item = QTableWidgetItem(str(indices.iloc[i, j]))

                    if indices.columns[j] == "Status":
                        if indices.iloc[i, j] == "Safe":
                            item.setBackground(Qt.green)
                        else:
                            item.setBackground(Qt.red)

                    self.table.setItem(i, j, item)

            # Enable buttons
            self.export_btn.setEnabled(True)
            self.save_plot_btn.setEnabled(True)
            self.plot_btn.setEnabled(True)
            self.process_btn.setEnabled(True)
            self.progress_bar.setValue(100)
        #counting the safe and unsafe
            safe_count = (indices["Status"] == "Safe").sum()
            unsafe_count = (indices["Status"] == "Unsafe").sum()

            self.status.setText(
                f" Processing complete! {safe_count} safe, {unsafe_count} unsafe samples"
            )
            self.status.setStyleSheet("color: lightgreen;")

            # Switch to table tab to show results
            self.tabs.setCurrentIndex(1)

            # Generate initial plot
            self.update_plot()

        except Exception as e:
            self.status.setText(" Computation Error")
            self.status.setStyleSheet("color: red;")
            self.process_btn.setEnabled(True)
            print(e)

    # UPDATE PLOT (Combined or Single)
    # =====================================================
    def update_plot(self):
        """Update the plot based on selected view"""
        
        if self.data is None or self.results is None:
            self.status.setText("No data to plot")
            return
        
        view = self.plot_view.currentText()
        
        if view == "Combined (Ra + Th + K)":
            self.plot_combined()
        elif view == "Ra Only":
            self.plot_single("Ra", "blue", "Ra-226")
        elif view == "Th Only":
            self.plot_single("Th", "red", "Th-232")
        elif view == "K Only":
            self.plot_single("K", "green", "K-40")
        
        self.canvas.draw()
        self.status.setText(f"Plot updated: {view}")
        self.status.setStyleSheet("color: cyan;")

    # PLOT COMBINED (All Three Nuclides)
    # =====================================================
    def plot_combined(self):
        """Plot all three nuclides on one graph"""
        
        self.figure.clear()
        #1 column, 1 row and all that..
        ax = self.figure.add_subplot(111)

        # Extract data
        Ra = self.data["Ra"].values   #.values turns list to numpy array
        Th = self.data["Th"].values
        K = self.data["K"].values

        dRa = self.data["dRa"].values
        dTh = self.data["dTh"].values
        dK = self.data["dK"].values

        Dose = self.results["Dose_Rate"].values
        dDose = self.results["dDose_Rate"].values

        # Sort for clean plotting (maintaining order of ra,th,and k  values)
        ra_idx = np.argsort(Ra)
        th_idx = np.argsort(Th)
        k_idx = np.argsort(K)

        #---- Ra ---- #apply the sorting order to the plottiing
        ax.errorbar(
            Ra[ra_idx],
            Dose[ra_idx],
            xerr=dRa[ra_idx],
            yerr=dDose[ra_idx],
            fmt='o',
            color='blue',
            ecolor='lightblue',
            elinewidth=1,
            capsize=3,
            label='Ra-226'
        )
        ra_fit = np.polyfit(Ra, Dose, 1) #finds the coefficients of a ploynomial
        ra_line = np.poly1d(ra_fit)
        ax.plot(Ra[ra_idx], ra_line(Ra[ra_idx]), "--", color="blue", linewidth=2)

        # ---- Th ----
        ax.errorbar(
            Th[th_idx],
            Dose[th_idx],
            xerr=dTh[th_idx],
            yerr=dDose[th_idx],
            fmt='s',
            color='red',
            ecolor='pink',
            elinewidth=1,
            capsize=3,
            label='Th-232'
        )
        th_fit = np.polyfit(Th, Dose, 1)
        th_line = np.poly1d(th_fit)
        ax.plot(Th[th_idx], th_line(Th[th_idx]), "--", color="red", linewidth=2)

        # ---- K ----
        ax.errorbar(
            K[k_idx],
            Dose[k_idx],
            xerr=dK[k_idx],
            yerr=dDose[k_idx],
            fmt='^',
            color='green',
            ecolor='lightgreen',
            elinewidth=1,
            capsize=3,
            label='K-40'
        )
        k_fit = np.polyfit(K, Dose, 1)
        k_line = np.poly1d(k_fit)
        ax.plot(K[k_idx], k_line(K[k_idx]), "--", color="green", linewidth=2)

        # ---- R² Values ----
        ra_r2 = np.corrcoef(Ra, Dose)[0,1]**2
        th_r2 = np.corrcoef(Th, Dose)[0,1]**2
        k_r2 = np.corrcoef(K, Dose)[0,1]**2

        ax.text(0.05, 0.95, f'Ra-226: R² = {ra_r2:.3f}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                color='blue')
        ax.text(0.05, 0.90, f'Th-232: R² = {th_r2:.3f}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                color='red')
        ax.text(0.05, 0.85, f'K-40: R² = {k_r2:.3f}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                color='green')

        # ---- Formatting ----
        ax.set_ylim(bottom=0, top=max(Dose) * 1.1)
        ax.set_xlim(left=0)

        ax.set_title("Dose Rate Correlation with All Radionuclides", fontsize=12, fontweight='bold')
        ax.set_xlabel("Radionuclide Concentration (Bq/kg)", fontsize=11)
        ax.set_ylabel("Dose Rate (nGy/h)", fontsize=11)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        # ---- Statistics Box ----
        stats_text = f"""
        Total Samples: {len(Dose)}
        Mean Dose Rate: {np.mean(Dose):.2f} ± {np.std(Dose):.2f} nGy/h
        Dose Rate Range: {np.min(Dose):.2f} - {np.max(Dose):.2f} nGy/h
        """
        ax.text(0.97, 0.03, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        self.figure.tight_layout()

    # =====================================================
    # PLOT SINGLE NUCLIDE
    # =====================================================
    def plot_single(self, nuclide, color, label):
        """Plot dose rate vs a single radionuclide"""
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Get data for specific nuclide
        if nuclide == "Ra":
            x = self.data["Ra"].values
            dx = self.data["dRa"].values
            x_label = "Ra-226 Concentration (Bq/kg)"
        elif nuclide == "Th":
            x = self.data["Th"].values
            dx = self.data["dTh"].values
            x_label = "Th-232 Concentration (Bq/kg)"
        else:  # K
            x = self.data["K"].values
            dx = self.data["dK"].values
            x_label = "K-40 Concentration (Bq/kg)"

        dose = self.results["Dose_Rate"].values
        ddose = self.results["dDose_Rate"].values

        # Sort for clean plotting
        idx = np.argsort(x)
        x_sorted = x[idx]
        dose_sorted = dose[idx]
        dx_sorted = dx[idx]
        ddose_sorted = ddose[idx]

        # Plot with error bars
        ax.errorbar(x_sorted, dose_sorted, 
                    xerr=dx_sorted, yerr=ddose_sorted,
                    fmt='o', color=color, ecolor='lightblue',
                    elinewidth=1, capsize=3, label=label)

        # Regression line
        fit = np.polyfit(x, dose, 1)
        line = np.poly1d(fit)
        ax.plot(x_sorted, line(x_sorted), '--', color=color, linewidth=2)

        # R² value
        r2 = np.corrcoef(x, dose)[0,1]**2
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Formatting
        ax.set_ylim(bottom=0, top=max(dose) * 1.1)
        ax.set_xlim(left=0)

        ax.set_title(f'Dose Rate vs {label} Concentration', fontsize=12, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel('Dose Rate (nGy/h)', fontsize=11)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        # Statistics box
        stats_text = f"""
        Samples: {len(dose)}
        Mean {label}: {np.mean(x):.2f} ± {np.std(x):.2f} Bq/kg
        Mean Dose: {np.mean(dose):.2f} ± {np.std(dose):.2f} nGy/h
        Correlation: R² = {r2:.3f}
        """
        ax.text(0.97, 0.03, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        self.figure.tight_layout()

    # =====================================================
    # SAVE PLOT
    # =====================================================
    def save_plot(self):

        if self.figure is None:
            self.status.setText("No plot available")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plot",
            "",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )

        if path:
            self.figure.savefig(
                path,
                dpi=300,
                bbox_inches="tight"
            )

            self.status.setText("Plot saved successfully")
            self.status.setStyleSheet("color: lightgreen;")

    # =====================================================
    # SAVE RESULTS
    # =====================================================
    def save_results(self):

        if self.results is None:
            self.status.setText("Nothing to save")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Results",
            "",
            "CSV Files (*.csv)"
        )

        if path:
            self.results.to_csv(path, index=False)
            self.status.setText("Results exported")
            self.status.setStyleSheet("color: lightgreen;")

    # =====================================================
    # STYLE
    # =====================================================
    def apply_style(self):

        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e2f;
                color: white;
                font-size: 14px;
            }

            QLabel#title {
                font-size: 24px;
                font-weight: bold;
                color: #00d4ff;
                padding: 10px;
            }

            QLabel#status {
                padding: 10px;
                background-color: #2b2b3d;
                border-radius: 6px;
                font-weight: bold;
            }

            QGroupBox {
                font-weight: bold;
                border: 2px solid #444;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #00d4ff;
            }

            QPushButton {
                background-color: #2d89ef;
                border: none;
                border-radius: 8px;
                padding: 10px;
                font-weight: bold;
                color: white;
            }

            QPushButton:hover {
                background-color: #45a1ff;
            }

            QPushButton:disabled {
                background-color: #444;
                color: #888;
            }

            QComboBox {
                padding: 8px;
                border-radius: 6px;
                background-color: #2b2b3d;
                border: 2px solid #555;
                color: white;
                min-width: 150px;
            }

            QComboBox::drop-down {
                border: none;
            }

            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid white;
                margin-right: 5px;
            }

            QTableWidget {
                background-color: #2b2b3d;
                gridline-color: #444;
                alternate-background-color: #333344;
            }

            QTableWidget::item {
                padding: 5px;
            }

            QHeaderView::section {
                background-color: #444;
                padding: 8px;
                font-weight: bold;
                border: 1px solid #555;
            }

            QProgressBar {
                border: 2px solid #444;
                border-radius: 6px;
                text-align: center;
                background-color: #2b2b3d;
            }

            QProgressBar::chunk {
                background-color: #2d89ef;
                border-radius: 4px;
            }

            QTabWidget::pane {
                border: 1px solid #444;
                border-radius: 8px;
                background-color: #1e1e2f;
            }

            QTabBar::tab {
                background-color: #333;
                padding: 10px 20px;
                margin: 2px;
                border-radius: 6px;
            }

            QTabBar::tab:selected {
                background-color: #2d89ef;
            }

            QTabBar::tab:hover:!selected {
                background-color: #444;
            }

            QLabel {
                color: white;
            }
        """)


# RUN APP
#run the application if this script is executed directly
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainApp()
    window.show()
    sys.exit(app.exec_())