#!/bin/sh 

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

docker build -f Dockerfile-base -t cvdp-example-agent-base .
docker build -f Dockerfile-agent -t cvdp-example-agent --no-cache .
