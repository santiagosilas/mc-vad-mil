"""
## Statistics

Datasets:
- P: Pets-2009 Dataset (I3D-RGB) - MB-WAN - PAIR C2,C3
- H: HQFS Dataset (I3D-RGB) - MB-WAN - PAIR C2,C4
- U: Up-Fall Dataset (I3D-FLOW) - MB-SULT - PAIR C1,C2

Schemes:
- M1: MC-MIL-1
- M2: MC-MIL-2
- FC: MC-MIL-FC


IF p > 0.05 THEN Probably the same distribution!

"""

from scipy.stats import ttest_rel
from scipy import stats
import numpy as np
import pandas as pd

# Obtained Results from Experiments
R = {"P":{},"U":{},"H":{}}
R["P"]["M1"] = np.array([74.24, 71.75, 75.91, 73.53, 74.24])
R["P"]["M2"] = np.array([78.86, 71.68, 77.88, 77.05, 77.44])
R["P"]["FC"] = np.array([73.53, 74.40, 73.53, 72.66, 73.53])
R["H"]["M1"] = np.array([87.40, 88.92, 90.42, 91.24, 91.27])
R["H"]["M2"] = np.array([84.16, 85.82, 87.29, 89.92, 87.21])
R["H"]["FC"] = np.array([80.34, 85.28, 82.57, 84.43, 86.68])
R["U"]["M1"] = np.array([78.68, 77.81, 76.50, 76.12, 79.21])
R["U"]["M2"] = np.array([83.50, 83.69, 83.31, 82.37, 83.38])
R["U"]["FC"] = np.array([79.97, 80.68, 77.42, 80.33, 77.52])

print(pd.DataFrame(np.vstack([
    R["P"]["M1"], R["P"]["M2"], R["P"]["FC"],
    R["H"]["M1"], R["H"]["M2"], R["H"]["FC"],
    R["U"]["M1"], R["U"]["M2"], R["U"]["FC"],
]).T, columns = ["P-M1","P-M2","P-FC","H-M1","H-M2","H-FC","U-M1","U-M2","U-FC"], index=["E1","E2","E3","E4","E5"]))


pvalues = dict()
pvalues["P"] = {"M1": {"M2":None,"FC":None},"M2": {"M1":None,"FC":None},"FC": {"M1":None,"M2":None}}
pvalues["H"] = {"M1": {"M2":None,"FC":None},"M2": {"M1":None,"FC":None},"FC": {"M1":None,"M2":None}}
pvalues["U"] = {"M1": {"M2":None,"FC":None},"M2": {"M1":None,"FC":None},"FC": {"M1":None,"M2":None}}



for ds in ["P","H","U"]:
  for t1 in ["FC"]:
    print(ds, f"{t1}: {np.mean(R[ds][t1]):.2f}% {np.std(R[ds][t1]):.2f}")
    for t2 in ["M1","M2"]:
      if t1 != t2:
        stat, pvalue = ttest_rel(R[ds][t1], R[ds][t2])
        pvalues[ds][t1][t2] = pvalue
        cod = "✔️" if pvalue > 0.05 else "×"
        print("\t", ds, f"{t2}: {np.mean(R[ds][t2]):.2f}% {np.std(R[ds][t2]):.2f} {cod}({pvalue:.2f})")

"""
P FC: 73.53% 0.55
	 P M1: 73.93% 1.34 ✔️(0.65)
	 P M2: 76.58% 2.52 ✔️(0.11)
H FC: 83.86% 2.21
	 H M1: 89.85% 1.49 ×(0.00)
	 H M2: 86.88% 1.90 ×(0.04)
U FC: 79.18% 1.42
	 U M1: 77.66% 1.20 ✔️(0.20)
	 U M2: 83.25% 0.46 ×(0.01)
"""

from scipy.stats import t
from scipy.stats import wilcoxon

for ds in ["P","H","U"]:
  for t1 in ["FC"]:
    print(ds, f"{t1}: {np.mean(R[ds][t1]):.2f}% {np.std(R[ds][t1]):.2f}")
    for t2 in ["M1","M2"]:
      if t1 != t2:
        stat, pvalue = wilcoxon(R[ds][t1], R[ds][t2])
        pvalues[ds][t1][t2] = pvalue
        cod = "✔️" if pvalue > 0.05 else "×"
        print("\t", ds, f"{t2}: {np.mean(R[ds][t2]):.2f}% {np.std(R[ds][t2]):.2f} {cod}({pvalue:.2f})")

"""
P FC: 73.53% 0.55
	 P M1: 73.93% 1.34 ✔️(0.62)
	 P M2: 76.58% 2.52 ✔️(0.12)
H FC: 83.86% 2.21
	 H M1: 89.85% 1.49 ✔️(0.06)
	 H M2: 86.88% 1.90 ✔️(0.06)
U FC: 79.18% 1.42
	 U M1: 77.66% 1.20 ✔️(0.31)
	 U M2: 83.25% 0.46 ✔️(0.06)
"""