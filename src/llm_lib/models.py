# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List

from pydantic import BaseModel, Field


class TextEvaluationInput(BaseModel):
    text     : str = Field(..., description="The text to be evaluated")
    type     : str = Field(..., description="The type of the text that we're analyzing")
    criteria : str = Field(..., description="The text describing the criteria that will be analyzed")

class EvaluationResult(BaseModel):
    score: float = Field(..., description="Score from 0 to 10 indicating the fulfillment of the criterion")
    description: str = Field(..., description="Description of why the text passed or failed the criterion")