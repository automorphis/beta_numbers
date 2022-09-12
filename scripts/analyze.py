import logging
import time
from pathlib import Path
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np

from beta_numbers.data.registers import Pickle_Register
from beta_numbers.data.states import Save_State_Type
from beta_numbers.perron_numbers import Salem_Number
from beta_numbers.utilities import Int_Polynomial

logging.basicConfig(filename="logs/analyze.log", level = logging.INFO)

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
start = time.time()

i = 0
for save_state in register.get_all_save_states(Save_State_Type.BS, beta, 100000):
    data = save_state.data
    if i % big_plot_sample_size == 0 and i > 0:
        logging.info(f"i = {i:12}, elapsed: {(time.time() - start):2.4f}")
        start = time.time()
        maxs.append(curr_max)
        curr_max = -1
        with (Path.home() / "big_plot2.pkl").open("wb") as fh:
            pkl.dump(maxs, fh)
    curr_max = max( np.max(data), curr_max )
    i += 100000

# for i,B in enumerate(register.get_all(Save_State_Type.BS, beta)):
#     if i % big_plot_sample_size == 0 and i > 0:
#         logging.info(f"i = {i:12}, elapsed: {(time.time() - start):2.4f}")
#         start = time.time()
#         maxs.append(curr_max)
#         curr_max = -1
#         with (Path.home() / "big_plot.pkl").open("wb") as fh:
#             pkl.dump(maxs, fh)
#     curr_max = max(B.max_abs_coef(), curr_max)

with (Path.home() / "big_plot.pkl").open("wb") as fh:
    pkl.dump(maxs, fh)

plt.plot(list(range(5000)),maxs)

