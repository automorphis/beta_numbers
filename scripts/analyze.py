import logging
from pathlib import Path
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np

from beta_numbers.data.registers import Pickle_Register
from beta_numbers.data.states import Save_State_Type
from beta_numbers.salem_numbers import Salem_Number
from beta_numbers.utilities import Int_Polynomial

logging.basicConfig(filename="logs/analyze.log")

data_root = Path("D:/beta_expansions")
saves_directory = data_root / "ivU4QAnanCq3ms2bdmBx"
register_filename = saves_directory / "register.pkl"

starting_dps = 128

beta = Salem_Number(Int_Polynomial((1,-10,-40,-59,-40,-10,1), starting_dps))

logging.info("Loading register from %s" % register_filename)
try:
    with register_filename.open("rb") as fh:
        register = pkl.load(fh)
except FileNotFoundError:
    logging.warning("Could not find register... discovering....")
    register = Pickle_Register.discover(saves_directory)
    with register_filename.open("wb") as fh:
        pkl.dump(register, fh)
logging.info("Register loaded.")

big_plot_sample_size = 10**6
curr_max = -1
maxs = []

for i,B in enumerate(register.get_all(Save_State_Type.BS, beta)):
    if i % big_plot_sample_size == 0 and i > 0:
        maxs.append(curr_max)
        curr_max = -1
    curr_max = max(np.max(np.abs(B.ndarray_coefs())), curr_max)

with (Path.home() / "big_plot.pkl").open("wb") as fh:
    pkl.dump(maxs, fh)

plt.plot(list(range(5000)),maxs)

