import pickle as pkl

from pathlib import Path

import numpy as np

from src.save_states import Pickle_Register, Save_State_Type
from src.utility import random_filename

def convert_native_data_to_current_format(read_register, write_directory):
    for metadata,filename in read_register.metadatas.items():
        old_save_state = Pickle_Register.load_save_state(filename,False)
        # old_data = old_save_state.data
        # if old_save_state.type == Save_State_Type.BS:
        #     new_data = np.empty((len(old_data), old_save_state.beta.degree()), dtype = int)
        #     for i in range(len(old_data)):
        #         new_data[i,:] = old_data[i].coef
        # else:
        #     new_data = old_data
        # new_save_state = old_save_state.get_metadata()
        # new_save_state.data = new_data
        Pickle_Register.dump_save_state(old_save_state, write_directory / filename.name)

data_root = Path.home() / "beta_expansions"

write_directory = random_filename(data_root, None)
read_directory = data_root / "YTnjZNENzEUanVhGvZKF"

# read_register = Pickle_Register.discover(read_directory)

with (read_directory / "register.pkl").open("rb") as fh:
    read_register = pkl.load(fh)


# read_register_filename = read_directory / "register.pkl"
# with read_register_filename.open("rb") as fh:
#     read_register = Pickle_Register(read_directory, pkl.load(fh))
#
# for metadata,filename in read_register.metadatas.items():

# with (read_directory / "register.pkl").open("wb") as fh:
#     pkl.dump(read_register, fh)

Path.mkdir(write_directory)
#
convert_native_data_to_current_format(read_register, write_directory)
