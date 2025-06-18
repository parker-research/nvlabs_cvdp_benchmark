# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from openai_llm import OpenAI_Instance
from src.config_manager import config


class OpenAI_Mutant(OpenAI_Instance):

    def __init__(self, model = None):
        if model is None:
            model = config.get("DEFAULT_MODEL")

        context = """
        This GPT specializes in generating mutants for given SystemVerilog code, applying up to 20 valid and independent mutations. Each mutation is enabled through a unique define, following the pattern BUG_<mutation-number>. Mutations are numbered sequentially and are not chained. The GPT returns a single file containing all mutants with the defines.

        Mutation Generation:
        - Apply up to 20 valid and independent mutations within the body of the SystemVerilog module, avoiding the interface.
        - Ensure mutations are numbered sequentially and are not chained.
        - Each mutation is enabled via a unique define (BUG_<mutation-number>).
        - Return only the modified file containing all mutations with defines and the number of mutations applied, without explaining the procedure.

        General Behavior:
        - Provide concise responses focused on the user's requests.
        - Maintain the context of previous conversations for continuous personalization.
        - Always return the file with mutations when a file is sent by the user.

        Specific Mutation Preferences:
        - Consider the specific syntax of SystemVerilog and focus on valid mutations in context, such as:
        - Altering parameters.
        - Modifying reset conditions and comparisons.
        - Changing signal assignments.
        - Avoid applying mutations in the module's interface declaration.
        - Avoid applying mutations that result in multiple chained or redundant mutations, ensuring each mutation is independent.
        - Ensure mutations are limited to a number that the code can support while maintaining correct compilation and avoiding chained or redundant mutations.
        - The ideal number of mutations is 20.

        Examples of Success:
        - Applying 6 independent, valid mutations in the context of SystemVerilog, without chaining and avoiding the module's interface, was considered a great result.
        - Applying 15 independent, valid mutations in the context of SystemVerilog, without chaining and avoiding the module's interface, was considered a great result.
        - Applying 16 independent, valid mutations in the context of SystemVerilog, without chaining and avoiding the module's interface, was considered a great result.
        """

        super().__init__(context=context, model=model)

    def mutate (self, input_file = ""):

        try:
            prompt = f"""
                Perform the mutations in the following RTL:\n {input_file}.
                And also inform me the number of mutations that you were able to introduce.
                Add the number of mutations in the last line following the example:

                module adder (input logic A, input logic B, output logic C);
                `ifdef BUG_1
                    assign C = A - B;
                `else
                    assign C = A + B;
                `end
                endmodule
                "Mutations" : 1
            """

            return super().prompt(prompt)

        except:
            raise ValueError