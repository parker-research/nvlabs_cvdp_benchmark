# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ----------------------------------------
# - JSONL Generation
# ----------------------------------------

import json

def create_jsonl(filename : str, content : []):

    value = ""

    # Convert to Lines
    for i in range(len(content)):
        value += json.dumps(content [i]) + "\n"

    with open(filename, "w+") as f:
        f.write(value)
