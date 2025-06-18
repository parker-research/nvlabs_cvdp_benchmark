# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import logging
import pytest
import os

from models import TextEvaluationInput
from openai_evaluator import OpenAI_Evaluator

logging.basicConfig(level=logging.INFO)

# ----------------------------------------
# - Tokenizer
# ----------------------------------------

def count_tokens(text: str):
    import tiktoken

    # Initialize the tokenizer for OpenAI's GPT-3 models (specifically the tokenizer used in GPT-3/4)
    encoding_name = "cl100k_base"  # This encoding is used for models like GPT-4
    tokenizer = tiktoken.get_encoding(encoding_name)
    tokens = tokenizer.encode(text)
    return len(tokens)

evaluators = {
    "rtl"   : "RTL Code",
    "verif" : "Testbench Code",
    "docs"  : "Documentation"
}

mapping = {

    "1"  : "docs",         # LLM consistency check between two documents, combine with
                           # entity extraction for a composite score. Also, ask LLM if detailed
                           # spec matches high-level.

    "2"  : "rtl",          # run against testbench, classify errors by iverilog
    "3"  : "rtl",          # run against testbench, classify errors by iverilog
    "4"  : "rtl",          # run against testbench, classify errors by iverilog
    "5"  : "rtl",          # run against testbench, classify errors by iverilog

    "6"  : "docs + rtl",   # Ranked scoring of snippet (Microsoft paper) + how much
                           # overlap composite score. LLM based likely again.

    "7"  : "rtl",          # LLM consistency with golden answer, and
                           # correlation with question (have a list of potential problems with code and score will be
                           # based on how well response overlaps with golden reference list)

    "8"  : "docs",         # Ranked scoring of snippet (Microsoft paper) + how
                           # much overlap composite score. LLM based likely again.

    "9"  : "docs",         # LLM consistency with golden answer, and correlation with question
    "10" : "docs",         # LLM consistency with golden answer, and correlation with question

    "11" : "docs + verif", # LLM consistency check between two
                           # documents, combine with entity extraction for a composite score. Also, ask LLM if detailed
                           # spec matches high-level.

    "12" : "verif",        # Composite score of: Run testbench stimulus and coverage results.

    "13" : "verif",        # Run testbench stimulus and hand-crafted bugs into Verilog RTL to verify checker
                           # works. (This is similar to assertion generation, but in testbench instead of in design)

    "14" : "verif",        # Hand-craft bugs to check assertion. Have a placeholder in correct and 
                           # non-correct RTL for where assertion should go. Run simulation that is hand-
                           # crafted to exercise.

    "15" : "docs",         # LLM to check consistency if found bug matches reference found
                           # bug. This identifies what the problem is and localizes where it is.

    "16" : "rtl"           # Should take CID015 as input. Creates a fix and modifies RTL with fix. 
                           # Simulate RTL for fix.

}

current_folder = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------
# - Test Class
# ----------------------------------------

class testRunner:

    def __init__(self):

        evaluator = "openai"
        logging.info(f"Using evaluator: {evaluator}")

        # ----------------------------------------
        # - Get Generic Criteria
        # ----------------------------------------

        with open(os.path.join(current_folder, "generic.json"), 'r') as f:
            self.data = json.load(f)

        # ----------------------------------------
        # - Get File from Environment Variable
        # ----------------------------------------

        uut = os.getenv("UUT")
        cat = os.getenv("CATEGORY")

        # If execution without datapoint
        if cat == None:
            types = uut.split("/")[-2]
            logging.info(f"Couldn't detect category from environment. Setting type as: {types} : {uut}")

        else:
            types = mapping[cat]

        self.type = types.split("+")[0].strip()

        # ----------------------------------------
        # - Setup Criterias
        # ----------------------------------------

        evaluation = {}

        for type in types.split("+"):

            # Remove whitespaces
            type = type.strip()

            for k in self.data [type]:

                if k in evaluation:
                    evaluation [k]['criteria'].extend(self.data [type][k]['criteria'])
                    evaluation [k]['scoring'] .extend(self.data [type][k]['scoring'])
                else:
                    evaluation [k] = self.data [type][k]

        if os.path.exists("/code/src"):

            for file in os.listdir("/code/src"):

                if file.endswith(".json"):
                    with open(os.path.join("/code/src", file), 'r') as f:
                        data = json.load(f)

                    # ----------------------------------------
                    # - Increment by criterias
                    # ----------------------------------------

                    for k in data:

                        if k in evaluation:
                            evaluation [k]['criteria'].extend(data [k]['criteria'])
                            evaluation [k]['scoring'] .extend(data [k]['scoring'])
                        else:
                            evaluation [k] = data [k]
    
        # ----------------------------------------
        # - Formatting Internal Variables
        # ----------------------------------------

        self.openai = OpenAI_Evaluator()
        self.evals  = evaluation

        if uut:
            self.set_file(uut, evaluation)

    def set_file(self, filename, evaluation : dict = {}):

        with open(filename, 'r') as f:
            eval_file = f.read()

        self.model         = {}
        self.model["text"] = eval_file
        self.model["type"] = evaluators[self.type]
        self.criterias     = evaluation.keys()

    def evaluate(self, id : int = 0, criteria : str = "", scale = [], scoring = []):

        text = f"Evaluate the {criteria} of the File by evaluating the following criteria:\n"

        for i, value in enumerate(scale):

            text += f"\t- {value} The scoring for this criteria should be done accordingly to the scale:\n"

            for j, k in enumerate(scoring[i]):
                size  = len(scoring[i])
                text += f"\t\t- {(j + 1) / size * 10} : {k}\n"

        self.model["criteria"] = text
        input_data        = TextEvaluationInput(**self.model)

        # ----------------------------------------
        # - Prompt Requesting
        # ----------------------------------------

        n_criteria = len(scale)
        logging.info(f"Running {n_criteria} evaluations for the provided text.")

        evaluations = self.openai.evaluation_loop(input_data)

        # ----------------------------------------
        # - Averaging and Comparison w/ Threshold
        # ----------------------------------------

        result = 0
        for res in evaluations:
            result += res["score"]

        result    = result / int(n_criteria)
        threshold = self.evals[criteria]['threshold']

        message   = f"OpenAI result - Criterion: {criteria}\n"\
                    f"Average score: {result} - "\
                    f"Threshold:     {threshold}"

        if (result < threshold):
            logging.error(message)

        else:
            logging.info (message)

        # ----------------------------------------
        # - Report in CLI
        # ----------------------------------------

        for i, res in enumerate(evaluations):

            message = f"Criteria : {scale[i]}\n"     \
                      f"Grade    : {res['score']}\n" \
                      f"Comments : {res['comments']}\n"

            if res['score'] < threshold:
                logging.warn(message)
            else:
                logging.info(message)

        # ----------------------------------------
        # - Reporting
        # ----------------------------------------

        with open(f'reports_{id}.json', 'w') as f:
            json.dump(evaluations, f, indent=4)

        return (result >= threshold)

# ----------------------------------------
# - Test Infrastructure
# ----------------------------------------

test = testRunner()

@pytest.mark.parametrize('id, criteria', enumerate(test.criterias))
def test_runner(id, criteria):

    scale = test.evals[criteria]['criteria']
    score = test.evals[criteria]['scoring']

    r = test.evaluate(id = id, criteria = criteria, scale = scale, scoring = score)
    assert r == True

# ----------------------------------------
# - CLI
# ----------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser("evaluator")
    parser.add_argument("-f", '--file', help="Evaluating file(s).",
                        required=True)

    args = vars(parser.parse_args())
    test.set_file(filename = args['file'])
    
    for id, criteria in enumerate(test.criterias):

        scale = test.evals[criteria]['criteria']
        score = test.evals[criteria]['scoring']

        r = test.evaluate(id = id, criteria = criteria, scale = scale, scoring = score)
        assert r == True
