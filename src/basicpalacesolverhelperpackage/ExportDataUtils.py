import numpy as np
import pandas as pd
import skrf as rf

class ExportDataUtils:

    def exportTouchstone(self, inputFile:str = "port-S.csv", outFile:str = "output.s2p", Z0:float=50):
        """Export touchstone file - STILL EXPERIMENTAL
        It can be used just to export data in touchstone format but to fuly characterize some part there is needed to
        make multiple simulation when ports are excited and then some impedance load connected to them.
        """
        # 1. Load CSV
        df = pd.read_csv(inputFile)
        df.columns = df.columns.str.strip()

        print("Columns found:", list(df.columns))

        # 2. Extract frequency (Palace outputs GHz)
        freqs = df.iloc[:, 0].values

        # 3. Build complex S-matrix  [nfreqs, nports, nports]
        #    Palace column order: Re[S11], Im[S11], Re[S21], Im[S21], Re[S12], Im[S12], Re[S22], Im[S22]
        n_data_cols = len(df.columns) - 1

        #
        # TODO: Port count needs to be corrected as fully characterized part there needs to be multiple
        #       measurements done when output ports are excited and then there is load resistor used
        #       on them!!!
        #
        n_ports = int(np.sqrt(n_data_cols / 2))

        print(f"Detected {n_ports}-port network, {len(freqs)} frequency points")

        S = np.zeros((len(freqs), n_ports, n_ports), dtype=complex)

        col = 1
        for j in range(n_ports):      # source port
            for i in range(n_ports):  # receiving port
                re = df.iloc[:, col].values
                im = df.iloc[:, col + 1].values
                S[:, i, j] = re + 1j * im
                col += 2

        # 4. Create scikit-rf Network and write Touchstone
        #    rf.Frequency expects Hz by default
        freq_obj = rf.Frequency.from_f(freqs, unit="GHz")
        network = rf.Network(frequency=freq_obj, s=S, z0=Z0, name="palace_sim")

        network.write_touchstone(outFile)
        print(f"Saved: {outFile}")

        # 5. Quick sanity check plot (optional)
        import matplotlib.pyplot as plt
        network.plot_s_db()
        plt.title("S-parameters from Palace FEM")
        plt.tight_layout()
        plt.savefig("s_params.png", dpi=150)
        plt.show()
