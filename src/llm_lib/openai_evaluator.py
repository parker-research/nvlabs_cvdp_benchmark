# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import re

from constants import PROMPT_TEMPLATE
from models import TextEvaluationInput, EvaluationResult
from openai_llm import OpenAI_Instance


class OpenAI_Evaluator (OpenAI_Instance):

    def __init__(self):
        super().__init__()

    def prompt (self, text : str, evaluate_type : str, criteria : str) -> EvaluationResult:

        response_text = ""

        try:
            prompt        = PROMPT_TEMPLATE.format(code=text, text=criteria, evaluate_type=evaluate_type)
            response      = super().prompt(prompt)
            response_text = re.sub(r'[\x00-\x1F\x7F]', '', response.strip())

            start         = response_text.find('[')
            end           = response_text[::-1].find(']')
            response_text = response_text[start:-end if end else None]
            output        = json.loads(response_text)

            return output #EvaluationResult(score=float(output['score']), description=output["description"])

        except Exception as e:
            logging.info(response_text)
            raise e

    def evaluation_loop(self, input_data: TextEvaluationInput) -> list[TextEvaluationInput]:

        return self.prompt(input_data.text, input_data.type, input_data.criteria)