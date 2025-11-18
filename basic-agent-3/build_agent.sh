#!/bin/sh 

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

docker build -f Dockerfile-agent -t cvdp-basic-agent-3 --no-cache .
