# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from mutate import mutate
import subprocess
import os
import glob
import pytest
import logging

logging.basicConfig(level=logging.INFO)

# ----------------------------------------
# - Execute Simulation
# ----------------------------------------

def execute_sim (folder = "", cmd_args = "", clean = False):

    cwd = os.getcwd()

    if (os.path.isdir(folder)) and clean:
        logging.info(f"Cleaning existing {folder} folder...")
        files = glob.glob(f"{folder}/*")

        for file in files:
            os.remove(file)

    elif (os.path.isdir(folder) == 0):
        # Creating answer folder
        logging.info(f"mkdir -p {folder}")
        os.system   (f"mkdir -p {folder}")

    logging.info("Compiling...")
    logging.info(f"iverilog {cmd_args}")
    os.system   (f"iverilog {cmd_args}")

    logging.info("Executing...")
    logging.info(f"cd {folder}; vvp a.out")

    os.chdir(folder)
    result = subprocess.run(["vvp", "a.out", "-lsim.log"])
    os.chdir(cwd)

    logging.info(f"Simulation return: {result.returncode}")
    return result.returncode

# ----------------------------------------
# - Test Encapsulation Class
# ----------------------------------------

class testRunner:

    def __init__(self):
        self.uut      = os.getenv("UUT")
        self.mutants  = os.getenv("MUTANTS").split()
        self.args     = os.getenv("ARGS")
        self.vcd_name = os.getenv("VCD_NAME")
        self.bugs     = []

        # Configure Golden Vector folder naming
        self.golden   = "golden"

        # Configure Bugs and Report
        self.fail     = 0

        print      ("----------------------------------------")
        print      ("- Evaluating golden...")
        print      ("----------------------------------------")

        self.golden_sim(self.mutants)

        self.n_bugs = 0
        for mut in self.mutants:

            mutation = mutate(mut, "gpt-4o")

            for i in range(mutation):

                self.n_bugs += 1
                self.bugs.append({
                    "file" : os.path.basename(mut),
                    "id"   : i + 1,
                })

    # ----------------------------------------
    # - Simulate Golden Answer
    # ----------------------------------------

    def golden_sim(self, files):

        cmd_args  = " ".join(files)
        cmd_args  = f"{self.uut} {cmd_args} {self.args} -o {self.golden}/a.out"
        error = execute_sim(folder = self.golden, cmd_args = cmd_args)

    # ----------------------------------------
    # - Simulate Bug Answer
    # ----------------------------------------

    def bug_sim(self, mutant = str, bug = int, vcd : bool = False):

        folder   = f"BUG_{bug}"
        args     = self.args + f"-DBUG_{bug}=1"
        cmd_args = f"{self.uut} {mutant} {args} -o {folder}/a.out"

        return self.evaluate_bug(id = bug, cmd_args = cmd_args, vcd = vcd)

    # ----------------------------------------
    # - VCD Comparison
    # ----------------------------------------

    def evaluate_vcd (self, bug_folder):

        print    (f'vcddiff {self.golden}/{self.vcd_name} {bug_folder}/{self.vcd_name} > {bug_folder}/report.log')
        os.system(f'vcddiff {self.golden}/{self.vcd_name} {bug_folder}/{self.vcd_name} > {bug_folder}/report.log')

        if (os.stat(f"{bug_folder}/report.log").st_size != 0):
            return True
        else:
            print (f"vcddiff of golden vector and {bug_folder} are equal! Testbench was unable to detect the bug!")
            return False

    # ----------------------------------------
    # - Encapsulate Bug Evaluation
    # ----------------------------------------

    def evaluate_bug (self, id = int, cmd_args = "", vcd : bool = False):

        error = execute_sim (folder = f"BUG_{id}", cmd_args = cmd_args, clean = True)

        if not vcd:
            return error
        else:
            return self.evaluate_vcd(bug_folder=f"BUG_{id}")

    # ----------------------------------------
    # - Encapsulates Bugs Resolution Test
    # ----------------------------------------

    def test_bug(resolution):
        assert resolution == True 

# ----------------------------------------
# - Test Infrascture
# ----------------------------------------

test = testRunner()

@pytest.mark.parametrize('bug', test.bugs)
def test_runner(bug):

    r = test.bug_sim (bug['file'], bug['id'])
    assert r != 0

# ----------------------------------------
# - Clean Python
# ----------------------------------------

if __name__ == "__main__":

    # ----------------------------------------
    # - Evaluate Bugs
    # ----------------------------------------

    print      ("----------------------------------------")
    print      ("- Running bugs...")
    print      ("----------------------------------------")

    fail = 0

    for bug in test.bugs:
        sim_res = test.bug_sim (bug['file'], bug['id'])
        fail   += sim_res
        assert (sim_res == True)

    # ----------------------------------------
    # - Print Resolution
    # ----------------------------------------

    print      ("----------------------------------------")
    print      ("- Final Evaluation ...")
    print      ("----------------------------------------")

    if (fail == test.n_bugs):
        print (f"Successful harness! All {fail} tests failed!")
    else:
        print (f"Harness failed to detect all bugs: {fail} / {test.n_bugs}")