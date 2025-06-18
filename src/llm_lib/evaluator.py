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

current_folder = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------
# - Test Class
# ----------------------------------------

class Evaluator:

    def __init__(self, criteria_files : list = [], uut : str = None):

        # ----------------------------------------
        # - Setup Criterias
        # ----------------------------------------

        if criteria_files:
            evaluation = self.set_evaluator(criteria_files)
        else:
            evaluation = {}
    
        # ----------------------------------------
        # - Formatting Internal Variables
        # ----------------------------------------

        self.model     = {}
        self.evals     = evaluation
        self.criterias = evaluation.keys()

        try:
            self.openai    = OpenAI_Evaluator()
            evaluator      = "openai"
            logging.info(f"Using evaluator: {evaluator}")

        except:
            self.openai    = None
            pass

        if uut:
            self.set_file(uut, evaluation)

    def set_evaluator(self, criteria_files : list = []):

        evaluation = {}

        for file in criteria_files:

            if os.path.exists(file):

                with open(file, 'r') as f:
                    data = json.load(f)

                print(data)

                # ----------------------------------------
                # - Increment by criterias
                # ----------------------------------------

                for k in data:

                    if k in evaluation:
                        evaluation [k]['criteria'].extend(data [k]['criteria'])
                        evaluation [k]['scoring'] .extend(data [k]['scoring'])
                    else:
                        evaluation [k] = data [k]

        return evaluation

    def set_file(self, filename, evaluation : dict = {}):

        if not evaluation:
            evaluation = self.evals

        with open(filename, 'r') as f:
            eval_file = f.read()

        self.model["text"] = eval_file
        self.model["type"] = "Response"
        self.criterias     = evaluation.keys()

    def evaluate(self, id : int = 0):

        criteria = list(self.evals.keys())[0]
        scale    = self.evals[criteria]['criteria']
        score    = self.evals[criteria]['scoring']

        return self.specific_evaluate(id = id, criteria = criteria, scale = scale, scoring = score)

    def specific_evaluate(self, id : int = 0, criteria : str = "", scale = [], scoring = []):

        text = f"Evaluate the quality of the response by evaluating the following criteria:\n"

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

        if self.openai:
            evaluations = self.openai.evaluation_loop(input_data)

        else:
            evaluations = [{ "score" : 0.0, "comments" : "Failed to execute OpenAI Model." }]

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

test = Evaluator()

@pytest.mark.parametrize('id, criteria', enumerate(test.criterias))
def test_runner(id, criteria):

    scale = test.evals[criteria]['criteria']
    score = test.evals[criteria]['scoring']

    r = test.specific_evaluate(id = id, criteria = criteria, scale = scale, scoring = score)
    assert r == True

# ----------------------------------------
# - CLI
# ----------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser("evaluator")
    parser.add_argument("-f", '--file', help="Evaluating file(s).", required=True)
    parser.add_argument("-u", '--uut',  help="Unit under test.",    required=True)

    args = vars(parser.parse_args())

    test.set_evaluator([args['file']])
    test.model["type"] = "Response"
    test.model["text"] = args['uut']
    
    for id, criteria in enumerate(test.criterias):

        scale = test.evals[criteria]['criteria']
        score = test.evals[criteria]['scoring']

        r = test.specific_evaluate(id = id, criteria = criteria, scale = scale, scoring = score)
        assert r == True