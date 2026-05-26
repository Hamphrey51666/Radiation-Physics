import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QLineEdit,
    QListWidget, QComboBox, QCheckBox,
    QFileDialog, QTabWidget, QTableWidget,
    QTableWidgetItem, QTextEdit, QGroupBox
)

# =========================================================
# COMPUTATION ENGINE
# =========================================================

import numpy as np

def compute_indices(df):

    df = df.copy()

    # =====================================================
    # BASE COMPUTATIONS
    # =====================================================

    rad_eq = (
        df["Ra"] +
        1.43 * df["Th"] +
        0.077 * df["K"]
    )

    hex_val = (
        (df["Ra"]/370) +
        (df["Th"]/259) +
        (df["K"]/4810)
    )

    hin_val = (
        (df["Ra"]/185) +
        (df["Th"]/259) +
        (df["K"]/4810)
    )

    dose_rate = (
        (df["Ra"]*0.462) +
        (df["Th"]*0.604) +
        (df["K"]*0.0417)
    )

    aede = (
        dose_rate *
        8760 *
        0.2 *
        1e-6
    )

    # =====================================================
    # UNCERTAINTY EXTRACTION
    # =====================================================

    dRa = df.get("dRa", pd.Series([0]*len(df)))
    dTh = df.get("dTh", pd.Series([0]*len(df)))
    dK = df.get("dK", pd.Series([0]*len(df)))

    # =====================================================
    # UNCERTAINTY PROPAGATION
    # =====================================================

    dRadEq = np.sqrt(
        (1*dRa)**2 +
        (1.43*dTh)**2 +
        (0.077*dK)**2
    )

    dHex = np.sqrt(
        (dRa/370)**2 +
        (dTh/259)**2 +
        (dK/4810)**2
    )

    dHin = np.sqrt(
        (dRa/185)**2 +
        (dTh/259)**2 +
        (dK/4810)**2
    )

    dDose = np.sqrt(
        (0.462*dRa)**2 +
        (0.604*dTh)**2 +
        (0.0417*dK)**2
    )

    dAEDE = (
        dDose *
        8760 *
        0.2 *
        1e-6
    )

    # =====================================================
    # RETURN DATAFRAME
    # =====================================================

    return pd.DataFrame({

        "Rad_Eq": rad_eq,
        "dRad_Eq": dRadEq,

        "Hex": hex_val,
        "dHex": dHex,

        "Hin": hin_val,
        "dHin": dHin,

        "Dose_Rate": dose_rate,
        "dDose_Rate": dDose,

        "AEDE": aede,
        "dAEDE": dAEDE
    })

# =========================================================
# MANUAL INPUT TAB
# =========================================================

class ManualInputTab(QWidget):

    def __init__(self, parent):
        super().__init__()

        self.parent_app = parent
        self.sample_count = 0

        self.layout = QVBoxLayout()

        # ---------------------------------------------
        # TITLE
        # ---------------------------------------------

        title = QLabel("Manual Sample Input")
        title.setObjectName("title")
        self.layout.addWidget(title)

        # ---------------------------------------------
        # INPUTS
        # ---------------------------------------------
         # validators
        validator = QDoubleValidator()
        self.ra = QLineEdit()
        self.ra.setPlaceholderText("Enter Ra value")

        self.th = QLineEdit()
        self.th.setPlaceholderText("Enter Th value")

        self.k = QLineEdit()
        self.k.setPlaceholderText("Enter K value")

        self.dra = QLineEdit()
        self.dra.setPlaceholderText("Enter Ra uncertainty")

        self.dth = QLineEdit()
        self.dth.setPlaceholderText("Enter Th uncertainty")

        self.dk = QLineEdit()
        self.dk.setPlaceholderText("Enter K uncertainty")

        self.dra.setValidator(validator)
        self.dth.setValidator(validator)
        self.dk.setValidator(validator)

        self.layout.addWidget(self.dra)
        self.layout.addWidget(self.dth)
        self.layout.addWidget(self.dk)

       

        self.ra.setValidator(validator)
        self.th.setValidator(validator)
        self.k.setValidator(validator)

        self.layout.addWidget(self.ra)
        self.layout.addWidget(self.th)
        self.layout.addWidget(self.k)

        # ---------------------------------------------
        # BUTTONS
        # ---------------------------------------------

        button_layout = QHBoxLayout()

        self.add_btn = QPushButton("Add Sample")
        self.clear_btn = QPushButton("Clear")
        self.save_btn = QPushButton("Save Dataset")

        button_layout.addWidget(self.add_btn)
        button_layout.addWidget(self.clear_btn)
        button_layout.addWidget(self.save_btn)

        self.layout.addLayout(button_layout)

        # ---------------------------------------------
        # SAMPLE LIST
        # ---------------------------------------------

        self.sample_list = QListWidget()
        self.layout.addWidget(self.sample_list)

        # ---------------------------------------------
        # STATUS
        # ---------------------------------------------

        self.counter = QLabel("Total Samples: 0")
        self.output = QLabel("Ready")

        self.layout.addWidget(self.counter)
        self.layout.addWidget(self.output)

        self.setLayout(self.layout)

        # ---------------------------------------------
        # CONNECTIONS
        # ---------------------------------------------

        self.add_btn.clicked.connect(self.add_sample)
        self.clear_btn.clicked.connect(self.clear_fields)
        self.save_btn.clicked.connect(self.save_dataset)

    # =====================================================

    def add_sample(self):

        try:
            ra = float(self.ra.text())
            th = float(self.th.text())
            k = float(self.k.text())

            dra = float(self.dra.text())
            dth = float(self.dth.text())
            dk = float(self.dk.text())

            self.sample_count += 1

            name = f"Sample{self.sample_count}"

            df = pd.DataFrame({
                "Ra": [ra],
                "Th": [th],
                "K": [k],
                "dRa": [dra],
                "dTh": [dth],
                "dK": [dk]  
            })

            self.parent_app.shared_samples[name] = df

            self.sample_list.addItem(
                f"{name} | "
                f"Ra={ra}±{dra}  "
                f"Th={th}±{dth}  "
                f"K={k}±{dk}"
            )

            self.parent_app.csv_tab.refresh_samples()

            self.counter.setText(
                f"Total Samples: {self.sample_count}"
            )

            self.output.setText(f"{name} added successfully")

            self.output.setStyleSheet("color: lightgreen;")

            self.clear_fields()

        except:
            self.output.setText("Invalid numerical input")
            self.output.setStyleSheet("color: red;")

    # =====================================================

    def clear_fields(self):

        self.ra.clear()
        self.th.clear()
        self.k.clear()
        self.dra.clear()
        self.dth.clear()
        self.dk.clear()

    # =====================================================

    def save_dataset(self):

        if not self.parent_app.shared_samples:
            self.output.setText("No samples available")
            self.output.setStyleSheet("color: orange;")
            return

        rows = []

        for name, df in self.parent_app.shared_samples.items():

            rows.append([
                name,
                df["Ra"][0],
                df["Th"][0],
                df["K"][0],
                df["dRa"][0],
                df["dTh"][0],
                df["dK"][0]
                ])

        final_df = pd.DataFrame(
            rows,
            columns=   [
                        "Sample",
                        "Ra", "Th", "K",
                        "dRa", "dTh", "dK"
                        ]
        )

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Dataset",
            "",
            "CSV Files (*.csv)"
        )

        if path:

            final_df.to_csv(path, index=False)

            self.output.setText("Dataset saved")
            self.output.setStyleSheet("color: cyan;")



# CSV ANALYSIS TAB

class CSVAnalysisTab(QWidget):

    def __init__(self, parent):
        super().__init__()

        self.parent_app = parent
        self.samples = {}
        self.last_results = None

        self.layout = QVBoxLayout()

        # TITLE
        

        title = QLabel("Single Sample Analysis")
        title.setObjectName("title")

        self.layout.addWidget(title)

       
        # LOAD BUTTON
      

        self.load_btn = QPushButton("Load CSV File")
        self.layout.addWidget(self.load_btn)

        
        # SAMPLE BOX
      

        self.sample_box = QComboBox()
        self.layout.addWidget(self.sample_box)

        
        # CHECKBOXES

        check_group = QGroupBox("Select Hazard Indices")

        check_layout = QVBoxLayout()

        self.cb_raeq = QCheckBox("Radium Equivalent")
        self.cb_hex = QCheckBox("External Hazard")
        self.cb_hin = QCheckBox("Internal Hazard")
        self.cb_dose = QCheckBox("Dose Rate")
        self.cb_aede = QCheckBox("AEDE")

        check_layout.addWidget(self.cb_raeq)
        check_layout.addWidget(self.cb_hex)
        check_layout.addWidget(self.cb_hin)
        check_layout.addWidget(self.cb_dose)
        check_layout.addWidget(self.cb_aede)

        check_group.setLayout(check_layout)

        self.layout.addWidget(check_group)

        # BUTTONS
        

        button_layout = QHBoxLayout()

        self.compute_btn = QPushButton("Compute")
        self.save_btn = QPushButton("Save Results")

        button_layout.addWidget(self.compute_btn)
        button_layout.addWidget(self.save_btn)

        self.layout.addLayout(button_layout)

        # RESULT TABLE

        self.result_table = QTableWidget()
        self.layout.addWidget(self.result_table)

        # OUTPUT

        self.output = QLabel("Ready")
        self.layout.addWidget(self.output)

        self.setLayout(self.layout)

        # ---------------------------------------------
        # CONNECTIONS
        # ---------------------------------------------

        self.load_btn.clicked.connect(self.load_csv)
        self.compute_btn.clicked.connect(self.compute)
        self.save_btn.clicked.connect(self.save)

    # =====================================================

    def refresh_samples(self):

        self.sample_box.clear()

        combined = {}

        combined.update(self.parent_app.shared_samples)
        combined.update(self.samples)

        self.sample_box.addItems(combined.keys())

    # =====================================================

    def load_csv(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV",
            "",
            "CSV Files (*.csv)"
        )

        if not path:
            return

        try:

            df = pd.read_csv(path)

            data = df.iloc[2:].apply(
                pd.to_numeric,
                errors='coerce'
            )

            self.samples = {}

            n = len(data.columns)//3

            for i in range(n):

                s = data.iloc[:, i*3:(i+1)*3]

                s.columns = ["Ra", "Th", "K"]

                self.samples[f"CSV_Sample{i+1}"] = s

            self.refresh_samples()

            self.output.setText("CSV Loaded Successfully")
            self.output.setStyleSheet("color: lightgreen;")

        except Exception as e:

            self.output.setText("Error Loading CSV")
            self.output.setStyleSheet("color: red;")

            print(e)

    # =====================================================

    def compute(self):

        name = self.sample_box.currentText()

        if not name:
            self.output.setText("No sample selected")
            return

        df = None

        if name in self.parent_app.shared_samples:
            df = self.parent_app.shared_samples[name]

        elif name in self.samples:
            df = self.samples[name]

        if df is None:
            return

        res = compute_indices(df).iloc[0]

        filtered = {}

        if self.cb_raeq.isChecked():
            filtered["Rad_Eq"] = (
            f"{round(res['Rad_Eq'],2)} ± "
            f"{round(res['dRad_Eq'],2)}"
            )

        if self.cb_hex.isChecked():
            filtered["Hex"] = (
            f"{round(res['Hex'],4)} ± "
            f"{round(res['dHex'],4)}"
)

        if self.cb_hin.isChecked():
            filtered["Hin"] = (
            f"{round(res['Hin'],4)} ± "
            f"{round(res['dHin'],4)}"
)

        if self.cb_dose.isChecked():
            filtered["Dose_Rate"] = (
            f"{round(res['Dose_Rate'], 2)} ± "
            f"{round(res['dDose_Rate'], 2)}"
            )

        if self.cb_aede.isChecked():
            filtered["AEDE"] = (
            f"{round(res['AEDE'],6)} ± "
            f"{round(res['dAEDE'],6)}"
)

        self.last_results = filtered

        self.result_table.setRowCount(len(filtered))
        self.result_table.setColumnCount(2)

        self.result_table.setHorizontalHeaderLabels(
            ["Index", "Value"]
        )

        for row, (key, value) in enumerate(filtered.items()):

            self.result_table.setItem(
                row,
                0,
                QTableWidgetItem(key)
            )

            self.result_table.setItem(
                row,
                1,
                QTableWidgetItem(str(value))
            )

        self.output.setText("Computation complete")
        self.output.setStyleSheet("color: cyan;")

    # =====================================================

    def save(self):

        if not self.last_results:
            self.output.setText("Nothing to save")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Results",
            "",
            "CSV Files (*.csv)"
        )

        if path:

            pd.DataFrame(
                [self.last_results]
            ).to_csv(path, index=False)

            self.output.setText("Results saved")
            self.output.setStyleSheet("color: lightgreen;")


# =========================================================
# BATCH PROCESSING TAB
# =========================================================

class BatchProcessingTab(QWidget):

    def __init__(self, parent):
        super().__init__()

        self.parent_app = parent

        self.data = None
        self.results = None

        self.layout = QVBoxLayout()

        # ---------------------------------------------
        # TITLE
        # ---------------------------------------------
        title = QLabel("Batch Processing")
        title.setObjectName("title")
        self.layout.addWidget(title)

        # ---------------------------------------------
        # PLOT AREA
        # ---------------------------------------------
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        # ---------------------------------------------
        # BUTTONS
        # ---------------------------------------------
        button_layout = QHBoxLayout()

        self.load_btn = QPushButton("Load Samples")
        self.compute_btn = QPushButton("Compute All")
        self.plot_btn = QPushButton("Plot Dose Rate Correlation")
        self.save_plot_btn = QPushButton("Save Plot")
        self.save_btn = QPushButton("Export Results")

        button_layout.addWidget(self.load_btn)
        button_layout.addWidget(self.compute_btn)
        button_layout.addWidget(self.plot_btn)
        button_layout.addWidget(self.save_plot_btn)
        button_layout.addWidget(self.save_btn)

        self.layout.addLayout(button_layout)

        # ---------------------------------------------
        # TABLE
        # ---------------------------------------------
        self.table = QTableWidget()
        self.layout.addWidget(self.table)

        # ---------------------------------------------
        # OUTPUT
        # ---------------------------------------------
        self.output = QLabel("Ready")
        self.layout.addWidget(self.output)

        self.setLayout(self.layout)

        # ---------------------------------------------
        # CONNECTIONS
        # ---------------------------------------------
        self.load_btn.clicked.connect(self.load_csv)
        self.compute_btn.clicked.connect(self.compute_all)
        self.plot_btn.clicked.connect(self.plot_correlation)
        self.save_plot_btn.clicked.connect(self.save_plot)
        self.save_btn.clicked.connect(self.save_results)

    # =====================================================
    # LOAD DATA
    # =====================================================
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

            required = [
                        "Ra", "Th", "K",
                        "dRa", "dTh", "dK"
                        ]

            for col in required:
                if col not in df.columns:
                    self.output.setText("CSV must contain Ra, Th, K")
                    return

            self.data = df

            self.output.setText(f"{len(df)} samples loaded")
            self.output.setStyleSheet("color: lightgreen;")

        except Exception as e:
            self.output.setText("Error loading CSV")
            self.output.setStyleSheet("color: red;")
            print(e)

    # =====================================================
    # COMPUTE ALL
    # =====================================================
    def compute_all(self):

        if self.data is None:
            self.output.setText("No data loaded")
            return

        try:
            df = self.data.copy()

            df["Sample"] = [
                f"SAMPLE_{i+1:03d}"
                for i in range(len(df))
            ]

            indices = compute_indices(df[["Ra", "Th", "K", "dRa", "dTh", "dK"]])

# round to 3 dp (AEDE gets 6 dp)
            for col in ["Rad_Eq", "dRad_Eq", "Hex", "dHex", "Hin", "dHin", "Dose_Rate", "dDose_Rate"]:
                indices[col] = indices[col].round(3)
            indices["AEDE"]  = indices["AEDE"].round(6)
            indices["dAEDE"] = indices["dAEDE"].round(6)

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

            for i in range(len(indices)):
                for j in range(len(indices.columns)):

                    item = QTableWidgetItem(str(indices.iloc[i, j]))

                    if indices.columns[j] == "Status":
                        if indices.iloc[i, j] == "Safe":
                            item.setBackground(Qt.green)
                        else:
                            item.setBackground(Qt.red)

                    self.table.setItem(i, j, item)

            self.output.setText("Batch computation completed")
            self.output.setStyleSheet("color: cyan;")

        except Exception as e:
            self.output.setText("Computation Error")
            self.output.setStyleSheet("color: red;")
            print(e)

    # =====================================================
    # PLOT CORRELATION
    # =====================================================
    # =====================================================
# PLOT CORRELATION WITH UNCERTAINTY BARS
# =====================================================
    def plot_correlation(self):

        if self.data is None or self.results is None:
            self.output.setText("No data to plot")
            return

        import numpy as np

    # -------------------------------------------------
    # CLEAR OLD FIGURE
    # -------------------------------------------------
        self.figure.clear()

        ax = self.figure.add_subplot(111)

    # -------------------------------------------------
    # EXTRACT DATA
    # -------------------------------------------------
        Ra = self.data["Ra"].values
        Th = self.data["Th"].values
        K = self.data["K"].values

        dRa = self.data["dRa"].values
        dTh = self.data["dTh"].values
        dK = self.data["dK"].values

        Dose = self.results["Dose_Rate"].values
        dDose = self.results["dDose_Rate"].values

    # -------------------------------------------------
    # SORT FOR CLEAN PLOTTING
    # -------------------------------------------------
        ra_idx = np.argsort(Ra)
        th_idx = np.argsort(Th)
        k_idx = np.argsort(K)

    # -------------------------------------------------
    # SORTED VALUES
    # -------------------------------------------------
        Ra_s = Ra[ra_idx]
        Th_s = Th[th_idx]
        K_s = K[k_idx]

        dRa_s = dRa[ra_idx]
        dTh_s = dTh[th_idx]
        dK_s = dK[k_idx]

        Dose_ra = Dose[ra_idx]
        Dose_th = Dose[th_idx]
        Dose_k = Dose[k_idx]

        dDose_ra = dDose[ra_idx]
        dDose_th = dDose[th_idx]
        dDose_k = dDose[k_idx]

    # -------------------------------------------------
    # ERROR BAR PLOTS
    # -------------------------------------------------

    # ----- Ra -----
        ax.errorbar(
            Ra_s,
            Dose_ra,
            xerr=dRa_s,
            yerr=dDose_ra,
            fmt='o',
            color='blue',
            ecolor='lightblue',
            elinewidth=1,
            capsize=3,
            label='Ra'
        )

    # ----- Th -----
        ax.errorbar(
            Th_s,
            Dose_th,
            xerr=dTh_s,
            yerr=dDose_th,
            fmt='o',
            color='red',
            ecolor='pink',
            elinewidth=1,
            capsize=3,
            label='Th'
        )

    # ----- K -----
        ax.errorbar(
            K_s,
            Dose_k,
            xerr=dK_s,
            yerr=dDose_k,
            fmt='o',
            color='green',
            ecolor='lightgreen',
            elinewidth=1,
            capsize=3,
            label='K'
        )

    # -------------------------------------------------
    # LINEAR REGRESSION
    # -------------------------------------------------

        ra_fit = np.polyfit(Ra, Dose, 1)
        th_fit = np.polyfit(Th, Dose, 1)
        k_fit = np.polyfit(K, Dose, 1)

        ra_line = np.poly1d(ra_fit)
        th_line = np.poly1d(th_fit)
        k_line = np.poly1d(k_fit)

        ax.plot(
            Ra_s,
            ra_line(Ra_s),
            "--",
            color="blue"
        )

        ax.plot(
            Th_s,
            th_line(Th_s),
            "--",
            color="red"
        )

        ax.plot(
            K_s,
            k_line(K_s),
            "--",
            color="green"
        )

    # -------------------------------------------------
    # AXIS SETTINGS
    # -------------------------------------------------

        ax.set_ylim(
            min(Dose) * 0.95,
            max(Dose) * 1.05
        )

        ax.set_title(
            "Dose Rate Correlation Analysis\nwith Uncertainty Bars"
     )

        ax.set_xlabel(
            "Radionuclide Concentration (Bq/kg)"
        )

        ax.set_ylabel(
            "Dose Rate (nGy/h)"
     )

        ax.legend()

        ax.grid(True)

    # -------------------------------------------------
    # DRAW CANVAS
    # -------------------------------------------------

        self.canvas.draw()

        self.output.setText(
         "Correlation plot with uncertainty bars updated"
        )

        self.output.setStyleSheet(
            "color: cyan;"
        )
    # =====================================================
    # SAVE PLOT
    # =====================================================
    def save_plot(self):

        if self.figure is None:
            self.output.setText("No plot available")
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

            self.output.setText("Plot saved successfully")
            self.output.setStyleSheet("color: lightgreen;")

    # =====================================================
    # SAVE RESULTS
    # =====================================================
    def save_results(self):

        if self.results is None:
            self.output.setText("Nothing to save")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Results",
            "",
            "CSV Files (*.csv)"
        )

        if path:
            self.results.to_csv(path, index=False)
            self.output.setText("Results exported")
            self.output.setStyleSheet("color: lightgreen;")
# =========================================================
# MAIN APP
# =========================================================

class MainApp(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "Radiological Hazard Assessment Tool"
        )

        self.resize(1200, 750)

        self.shared_samples = {}

        self.layout = QVBoxLayout()

        # ---------------------------------------------
        # HEADER
        # ---------------------------------------------

        header = QLabel(
            "Radiological Hazard Assessment Tool"
        )

        header.setObjectName("main_header")

        sub = QLabel(
            "Environmental Radiation Analysis System"
        )

        sub.setObjectName("sub_header")

        self.layout.addWidget(header)
        self.layout.addWidget(sub)

        # ---------------------------------------------
        # TABS
        # ---------------------------------------------

        self.tabs = QTabWidget()

        self.manual_tab = ManualInputTab(self)
        self.csv_tab = CSVAnalysisTab(self)
        self.batch_tab = BatchProcessingTab(self)

        self.tabs.addTab(
            self.manual_tab,
            "Manual Input"
        )

        self.tabs.addTab(
            self.csv_tab,
            "CSV Analysis"
        )

        self.tabs.addTab(
            self.batch_tab,
            "Batch Processing"
        )

        self.layout.addWidget(self.tabs)

        # ---------------------------------------------
        # THRESHOLD NOTES
        # ---------------------------------------------

        limits = QLabel(
            "Thresholds  |  "
            "Rad_Eq ≤ 370  |  "
            "Hex ≤ 1  |  "
            "Hin ≤ 1  |  "
            "Dose Rate ≤ 59  |  "
            "AEDE ≤ 1"
        )

        limits.setObjectName("limits")

        self.layout.addWidget(limits)

        # ---------------------------------------------
        # FOOTER
        # ---------------------------------------------

        footer = QLabel(
            "Developed by Hamphrey Tumusiime"
        )

        footer.setObjectName("footer")

        self.layout.addWidget(footer)

        self.setLayout(self.layout)

        # STYLE

        self.setStyleSheet("""

            QWidget {
                background-color: #1e1e2f;
                color: white;
                font-size: 14px;
            }

            QLabel#main_header {
                font-size: 28px;
                font-weight: bold;
                color: cyan;
                padding: 10px;
            }

            QLabel#sub_header {
                font-size: 16px;
                color: lightgray;
                padding-bottom: 10px;
            }

            QLabel#title {
                font-size: 20px;
                font-weight: bold;
                color: #00d4ff;
                padding: 5px;
            }

            QLabel#footer {
                color: gray;
                padding-top: 10px;
            }

            QLabel#limits {
                color: orange;
                font-weight: bold;
                padding: 8px;
            }

            QPushButton {
                background-color: #2d89ef;
                border-radius: 8px;
                padding: 10px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #45a1ff;
            }

            QLineEdit {
                padding: 8px;
                border: 2px solid #555;
                border-radius: 6px;
                background-color: #2b2b3d;
            }

            QComboBox {
                padding: 8px;
                border-radius: 6px;
                background-color: #2b2b3d;
            }

            QListWidget {
                background-color: #2b2b3d;
                border-radius: 6px;
            }

            QTableWidget {
                background-color: #2b2b3d;
                gridline-color: gray;
            }

            QHeaderView::section {
                background-color: #444;
                padding: 5px;
                font-weight: bold;
            }

            QTabWidget::pane {
                border: 1px solid gray;
            }

            QTabBar::tab {
                background: #333;
                padding: 10px;
                margin: 2px;
            }

            QTabBar::tab:selected {
                background: #2d89ef;
            }

        """)


# =========================================================
# RUN APP
# =========================================================

app = QApplication(sys.argv)

window = MainApp()

window.show()

sys.exit(app.exec_())