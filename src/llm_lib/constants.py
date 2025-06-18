# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

PROMPT_TEMPLATE = """Consider the following {evaluate_type}:

{code}

{text}

Scores that are not part of the scale given above aren't considered valid.
The description should be very useful, detailed, insightful, and properly formatted.
Provide some example of lines that could be improoved and provide the necessary information for improoving it.

The output should be in the following JSON format without any extra formatting or characters.
Consider one entry for each analyzed criteria.
[
    {{
        "score": <score>,
        "description": "<description>",
        "comments" : "<example_lines>"
    }}
]
Here is an example of a valid JSON output:
[
    {{
        "score": 85,
        "description": "The RTL code meets most of the criteria, but there are minor issues with overflow handling."
        "comments" : " Line 62: always @( * ) begin and Line 63: if (!nreset) begin - Combinational logic doesn't have to handle the reset behavior. That should be done by the flip-flops."
    }}
]"""