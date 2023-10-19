import datetime
import os
import re
import shutil
import unittest
import subprocess
from pathlib import Path
import time


test_home_dir = Path.home() / "betanumbers_slurm_testcases"
python_command = "sage -python"
error_filename = test_home_dir / 'test_slurm_error.txt'
sbatch_filename = test_home_dir / 'test.sbatch'
slurm_tests_filename = Path(__file__).parent / "slurm_tests"
allocation_query_sec = 0.5
running_query_sec = 0.5
allocation_max_sec = 60
timeout_extra_wait_sec = 30

def write_batch_file(time_sec, slurm_test_main_filename, num_processes, args):

    with sbatch_filename.open("w") as fh:
        fh.write(
f"""#!/usr/bin/env bash

#SBATCH --job-name=corniferslurmtests
#SBATCH --time={datetime.timedelta(seconds = time_sec)}
#SBATCH --ntasks={num_processes}
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --error={error_filename}

{python_command} {slurm_test_main_filename} {num_processes} {test_home_dir} {args}
""")
#SBATCH --output=/dev/null
#SBATCH --mail-user=lane.662@osu.edu
#SBATCH --mail-type=all

class TestSlurm(unittest.TestCase):

    def setUp(self):
        self.job_id = None

    def tearDown(self):

        subprocess.run(["scancel", self.job_id])
        time.sleep(2)

    @classmethod
    def setUpClass(cls):

        if test_home_dir.exists():
            shutil.rmtree(test_home_dir)

        test_home_dir.mkdir(parents = True, exist_ok = False)

    @classmethod
    def tearDownClass(cls):

        if test_home_dir.exists():
            shutil.rmtree(test_home_dir)

    def check_empty_error_file(self):

        error_filename.exists()

        with error_filename.open("r") as fh:

            contents = ""

            for line in fh:
                contents += line

        if len(contents) > 0:
            self.fail(f"Must be empty error file! Contents: {contents}")

    def check_timeout_error_file(self):

        error_filename.exists()

        with error_filename.open("r") as fh:

            contents = ""

            for line in fh:
                contents += line

        if re.match(r"^slurmstepd: error: \*\*\* JOB.*ON.*CANCELLED AT.*DUE TO TIME LIMIT \*\*\*$", contents) is None:
            self.fail(f"Invalid error file. Contents: {contents}")

    def wait_till_running(self, max_sec, query_sec):

        querying = True
        start = time.time()

        while querying:

            if time.time() - start >= max_sec + timeout_extra_wait_sec:
                raise Exception("Ran out of time!")

            time.sleep(query_sec)
            squeue_process = subprocess.run(
                ["squeue", "-j", self.job_id, "-o", "%.2t"], capture_output = True, text = True
            )
            querying = "PD" in squeue_process.stdout

    def wait_till_not_running(self, max_sec, query_sec):

        querying = True
        start = time.time()

        while querying:

            if time.time() - start >= max_sec:
                raise Exception("Ran out of time!")

            time.sleep(query_sec)
            squeue_process = subprocess.run(
                ["squeue", "-j", self.job_id, "-o", "%.2t"], capture_output = True, text = True
            )
            querying = squeue_process.stdout != "ST\n"

        time.sleep(query_sec)

    def submit_batch(self):

        sbatch_process = subprocess.run(
            ["sbatch", str(sbatch_filename)], capture_output = True, text = True
        )
        self.job_id = sbatch_process.stdout[20:-1]
        print(self.job_id)

    def test_slurm_1(self):

        slurm_test_main_filename = slurm_tests_filename / 'test1.py'
        running_max_sec = 120
        slurm_time = running_max_sec + 1
        num_processes = 5
        blk_size = 10
        max_sum_abs_coef = {2: 10, 3: 7, 4: 5}
        max_sum_abs_coef_str = str(max_sum_abs_coef).replace(":", "").replace(",", "").replace("{", "").replace("}", "")
        write_batch_file(
            slurm_time, slurm_test_main_filename, num_processes, f"{blk_size} {slurm_time - 10} {max_sum_abs_coef_str}"
        )
        print("Submitting test batch #1...")
        self.submit_batch()
        self.wait_till_running(allocation_max_sec, allocation_query_sec)
        print(f"Running test #1 (running_max_sec = {running_max_sec})...")
        self.wait_till_not_running(running_max_sec, running_query_sec)
        print("Checking test #1...")
        self.check_empty_error_file()


