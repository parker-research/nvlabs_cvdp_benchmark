# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import os

from dotenv import load_dotenv
from src.config_manager import config

from openai_mutant_gen import OpenAI_Mutant

logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

def mutate(filename, model = None):
    if model is None:
        model = config.get("DEFAULT_MODEL")

    evaluator = "openai"
    logging.info(f"Using evaluator: {evaluator}")

    with open(filename, 'r') as f:
        eval_file = f.read()

    logging.info(f"Running mutate.py for the provided text:{filename}")
    logging.info(f"Using model ({model}) to generate the mutations...")

    openai_model = OpenAI_Mutant(model)
    error        = 3

    while True:

        try:
            mutations    = openai_model.mutate(eval_file)

        except Exception as e:

            if (error > 0):
                print(f"Error in json decoding... Trying again...")
                error -= 1
            else:
                raise Exception(e)

        else:
            break

    # Split into lines
    mutations    = mutations.split("\n")

    # Extract number of mutations
    n_mutations  = mutations[-2].split(":")[1].strip()
    n_mutations  = int(n_mutations)

    # Remove first and last line
    mutations    = mutations[1:-2]
    mutations    = "\n".join(mutations)

    # Write new generated file
    new_filename = os.path.basename(filename)
    with open(new_filename, 'w+') as f:
        f.write(mutations)
    
    logging.info(f"Wrote mutated file in: {new_filename}")
    return n_mutations

if __name__ == "__main__":

    parser = argparse.ArgumentParser("evaluator")
    parser.add_argument("-f", '--file', help="Evaluating file.", required=True, type=str)
    args = vars(parser.parse_args())

    logging.info(mutate(args['file']))