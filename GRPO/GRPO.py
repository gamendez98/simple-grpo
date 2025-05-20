import re
from dataclasses import dataclass
from typing import Optional

import torch
from einops import rearrange
from transformers import AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, \
    PreTrainedTokenizerFast

from GRPO.shorthand_operations import pad_and_concat

FUNCTION_STOP_REGEX = r'`(\s*\w+\([^()]+?\))`$'
ANSWER_REGEX = r'<answer>(.+?)</answer>'


@dataclass
class FunctionCall:
    name: str
    args: list[str]
    result: Optional[str] = None
    exception: Optional[str] = None

    def format_result(self) -> str:
        if self.result is not None:
            return f' <result>{self.result}</result>'
        else:
            return f' <exception>{self.exception}</exception>'


class FunctionStopCriteria(StoppingCriteria):

    def __init__(self, tokenizer, prompt_size):
        self.regex_pattern = re.compile(FUNCTION_STOP_REGEX)
        self.tokenizer = tokenizer
        self.prompt_size: int = prompt_size
        self.function_call: Optional[FunctionCall] = None

    def __call__(self, input_ids, scores, **kwargs):
        decoded = self.tokenizer.decode(input_ids[0, self.prompt_size:], skip_special_tokens=True)
        match = self.regex_pattern.search(decoded)
        if match:
            function_string = match.group(0)
            function_string = (
                function_string.replace('`', '')
                .replace('`', '')
                .strip()
            )
            function_name, function_args = function_string.split('(', 1)
            function_args = function_args.rsplit(')', 1)[0].split(',')
            function_args = [a.strip() for a in function_args]
            self.function_call = FunctionCall(
                name=function_name,
                args=function_args
            )
            return True
        return False


class ToolGRPOTrainer:
    def __init__(self, model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizerFast):
        self.model = model
        self.tokenizer = tokenizer
        self.answer_regex = re.compile(ANSWER_REGEX)
        self.call_reward = 1
        self.successful_call_reward = 1
        self.output_format_reward = 1
        self.correctness_reward = 2

    @staticmethod
    def tool_multiply(number_1, number_2):
        return number_1 * number_2

    def call_function(self, function_call: FunctionCall):
        try:
            if function_call.name == 'multiply':
                result = self.tool_multiply(*[eval(a) for a in function_call.args])
                function_call.result = str(result)
        except Exception as e:
            function_call.exception = str(e)

    @staticmethod
    def create_prompts(tokenizer: PreTrainedTokenizerFast, number_inputs: list[tuple[int]]) -> list[str]:
        all_prompt_parts = []
        for number_1, number_2 in number_inputs:
            user_prompt = (
                f"What is the result of {number_1} times {number_2}."
                "You can use the calculator. By using a tool. like so <function>multiply(2, 5)</function> <result>10</result>"
                " Provide the final result in an answer tag <answer>final answer</answer>."
            )
            prompt_parts = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_prompt}
            ]
            all_prompt_parts.append(prompt_parts)
        return tokenizer.apply_chat_template(
            all_prompt_parts,
            tokenize=False,
            add_generation_prompt=True,
        )

    @staticmethod
    def calculate_advantages(batch_group_rewards: torch.Tensor, drgrpo=False) -> torch.Tensor:
        group_dim = 1
        advantage = batch_group_rewards - batch_group_rewards.mean(dim=group_dim, keepdim=True)
        if drgrpo:
            return advantage
        advantage = advantage / advantage.std(dim=group_dim, keepdim=True)
        return advantage

    def produce_groups(self, number_inputs: list[tuple[int]], group_size: int) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        prompts = self.create_prompts(self.tokenizer, number_inputs)
        batch_group_generations = []
        batch_group_generation_masks = []
        batch_group_rewards = []
        for prompt, n_input in zip(prompts, number_inputs):
            group_generations = []
            group_generation_masks = []
            group_rewards = []
            for i in range(group_size):
                tokenized_prompt = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
                generation, function_calls, generation_mask = self.generate(tokenized_prompt["input_ids"])
                generation_text = self.tokenizer.decode(generation[0])
                reward = self.reward_function(generation_text, n_input, function_calls)
                group_generations.append(generation)
                group_generation_masks.append(generation_mask)
                group_rewards.append(reward)
            # shape (group, sequence)
            group_generations = pad_and_concat(group_generations)
            # shape (group, sequence)
            group_generation_masks = pad_and_concat(group_generation_masks)
            # (group)
            group_rewards = torch.tensor(group_rewards, dtype=torch.float32)
            batch_group_generations.append(group_generations)
            batch_group_generation_masks.append(group_generation_masks)
            batch_group_rewards.append(group_rewards)
        # (batch, group, sequence)
        batch_group_generations = [rearrange(gg, 'group sequence -> 1 group sequence') for gg in
                                   batch_group_generations]
        batch_group_generations = pad_and_concat(batch_group_generations)
        # (batch, group, sequence)
        batch_group_generation_masks = [rearrange(ggm, 'group sequence -> 1 group sequence') for ggm in
                                        batch_group_generation_masks]
        batch_group_generation_masks = pad_and_concat(batch_group_generation_masks)
        # (batch, group)
        batch_group_rewards = [rearrange(gr, 'group -> 1 group') for gr in batch_group_rewards]
        batch_group_rewards = pad_and_concat(batch_group_rewards)
        return batch_group_generations, batch_group_generation_masks, batch_group_rewards

    @torch.inference_mode()
    def generate(self, tokenized_prompt: torch.Tensor) -> tuple[torch.Tensor, list[FunctionCall], torch.Tensor]:
        '''
            Note to self: if you want to apply the tool stop to a batch what you do is the following:
            you can either let it all run and then replace or
            take out the tool call generate the rest and then complete the prompt with the tool call
        '''
        model_inputs = tokenized_prompt.clone()
        function_calls = []
        generation_mask = [0] * tokenized_prompt.shape[1]
        while True:
            function_criteria = FunctionStopCriteria(self.tokenizer, tokenized_prompt.shape[1])
            output = self.model.generate(
                model_inputs,
                max_length=500,
                stopping_criteria=StoppingCriteriaList([function_criteria])
            )
            generation_mask += [1] * output.shape[1]
            if function_criteria.function_call is None:
                # exit once the model exits without using a tool
                break
            model_inputs = torch.cat([model_inputs, output], dim=1)
            function_call = function_criteria.function_call
            self.call_function(function_call)
            function_calls.append(function_call)
            call_result_tokens = self.tokenizer(function_call.format_result(), return_tensors='pt')
            result_ids = call_result_tokens.input_ids.to(output.device)
            generation_mask += [0] * result_ids.shape[1]
            model_inputs = torch.cat([model_inputs, result_ids], dim=1)
        generation_mask = torch.tensor(generation_mask, dtype=torch.bool)
        return output, function_calls, rearrange(generation_mask, 'sequence -> 1 sequence')

    def reward_function(self, generation_text: str, number_inputs: tuple[int], function_calls: list[FunctionCall]):
        total_reward = 0
        generation_output = generation_text.rsplit('<|im_start|>assistant\n', 1)[-1]
        answer = self.answer_regex.search(generation_output)
        if answer:
            total_reward += self.output_format_reward
            answer = answer.group(1)
            answer = answer.replace('<answer>', '').replace('</answer>', '').strip()
            if str(self.tool_multiply(*number_inputs)) == answer:
                total_reward += self.correctness_reward
        if function_calls:
            total_reward += self.call_reward
            function_call = function_calls[-1]
            if function_call.result is not None:
                total_reward += self.successful_call_reward
        return total_reward

    def objective_function(self, model, old_model, batch_group_generation, batch_group_generation_mask,
                           batch_group_rewards, shortest_prompt_size=0, drgrpo=False, ref_model=None):
        advantages = self.calculate_advantages(batch_group_rewards, drgrpo)

        batch_group_flat_generation = rearrange(batch_group_generation, 'batch group sequence -> (batch group) sequence')
        logit = model(batch_group_flat_generation)
        old_logit = old_model(batch_group_flat_generation)
