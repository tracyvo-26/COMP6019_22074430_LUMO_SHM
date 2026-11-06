"""
Lists the sessions where SSI-COV found no mode frequency at all.

Needs: ssi_modal_frequencies.csv (from ssi_cov_batch.py).
"""
import pandas as pd
import os
f = pd.read_csv(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "02_modal_identification", "SSI-COV", "output_batch", "blockrows20_final", "ssi_modal_frequencies.csv")))
cols = [c for c in f.columns if c.startswith("freq_")]
print(f[f[cols].isna().all(axis=1)][["session", "damage_label", "temperature_degC"]])