import copy
import random
import re
from dataclasses import dataclass
from typing import Protocol, Generator, Counter

import numpy as np
import torch
from einops import rearrange, reduce
from torch import nn
from tqdm import tqdm
from transformers import StoppingCriteriaList, \
    PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast

from grpo.cosine_annealing_with_warmup import CosineAnnealingWithWarmup
from grpo.function_calling import FunctionCall, FunctionStopCriteria
from grpo.shorthand_operations import pad_and_concat

ANSWER_REGEX = r'<answer>(.+?)</answer>'


def calculate_entropy(arr):
    """Calculate the Shannon entropy of an array."""
    # Convert the array to a NumPy array
    arr = np.array(arr)

    # Get the unique elements and their counts
    values, counts = np.unique(arr, return_counts=True)

    # Compute the probabilities
    probabilities = counts / counts.sum()

    # Compute entropy: sum(-p * log2(p))
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy

def most_common_element_index(arr):
    arr = np.asarray(arr)
    counts = Counter(arr)
    most_common_element, _ = counts.most_common(1)[0]
    first_index = np.where(arr == most_common_element)[0][0]
    return first_index

class GRPOTrainableModelInterface(Protocol):
    def parameters(self) -> Generator[nn.Parameter, None, None]: ...

    def __call__(self, *args, **kwargs) -> CausalLMOutputWithPast: ...

    def eval(self) -> 'GRPOTrainableModelInterface': ...

    @property
    def device(self) -> torch.device: ...

    def generate(self, *args, **kwargs) -> torch.Tensor: ...


@dataclass
class ToolGRPOTrainerConfig:
    group_size: int = 4
    batch_size: int = 4
    minibatch_size: int = 1
    numbers_range: tuple[int, int] = (10, 100)
    call_reward: float = 0
    successful_call_reward: float = 0
    output_format_reward: float = 1
    correctness_reward: float = 10
    dkl_weight: float = 0.01
    gradient_clipping: float = 0.1
    policy_lr: float = 5e-6
    warmup_steps: int = 25
    max_steps: int = 250 * 8
    use_divergence_kl: bool = False
    adam_betas: tuple[float, float] = (0.9, 0.99)
    adam_weight_decay: float = 0.1
    epochs_per_step: int = 4
    objective_clip_band: float = 0.2
    drgrpo: bool = True
    advantage_std_epsilon: float = 1e-8
    generation_max_tokens: int = 128
    min_group_entropy: float = 0.5
    entropy_retries = 20


class ToolGRPOTrainer:
    def __init__(
            self,
            model: GRPOTrainableModelInterface,
            tokenizer: PreTrainedTokenizerFast,
            config: ToolGRPOTrainerConfig = None,
    ):
        self.config = config or ToolGRPOTrainerConfig()

        self.rl_steps = self.config.max_steps // self.config.group_size
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.policy_lr,
            betas=self.config.adam_betas,
            weight_decay=self.config.adam_weight_decay,
        )
        self.scheduler = CosineAnnealingWithWarmup(self.optimizer, self.config.warmup_steps, self.rl_steps)
        self.model = model
        self.old_model = copy.deepcopy(model).eval()
        self.ref_model = None
        if self.config.use_divergence_kl:
            self.ref_model = copy.deepcopy(model).eval()
        self.tokenizer = tokenizer
        self.answer_regex = re.compile(ANSWER_REGEX)

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
    def create_prompts(tokenizer: PreTrainedTokenizerFast, number_inputs: list[tuple[int, int]]) -> list[str]:
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

    def calculate_advantages(self, batch_group_rewards: torch.Tensor) -> torch.Tensor:
        group_dim = 1
        advantage = batch_group_rewards - batch_group_rewards.mean(dim=group_dim, keepdim=True)
        if self.config.drgrpo:
            return advantage
        advantage = advantage / (advantage.std(dim=group_dim, keepdim=True) + self.config.advantage_std_epsilon)
        return advantage

    def produce_groups(self, number_inputs: list[tuple[int, int]]) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Produces groups of generated sequences, their corresponding masks, and
        reward values based on a list of input tuples. This function performs
        batch processing, involving the generation of prompts, tokenization,
        sequence generation, reward computation, and tensor concatenation.
        Outputs are returned as batched tensors.

        :param number_inputs: List of tuples, where each tuple contains two
            integers used to generate prompts and calculate rewards.
        :type number_inputs: List[tuple[int, int]]
        :return: A tuple containing three tensors:
            - batch_group_generations: Batched tensor of generated sequences
              with dimensions (batch, group, sequence).
            - batch_group_generation_masks: Batched tensor of masks indicating
              valid parts of generated sequences with dimensions
              (batch, group, sequence).
            - batch_group_rewards: Batched tensor of rewards corresponding
              to each generated sequence group, with dimensions
              (batch, group).
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        prompts = self.create_prompts(self.tokenizer, number_inputs)
        batch_group_generations = []
        batch_group_generation_masks = []
        batch_group_rewards = []
        batch_group_reward_entropy = []
        for prompt, n_input in zip(prompts, number_inputs):
            group_generations = []
            group_generation_masks = []
            group_rewards = []
            group_reward_entropy = 0.0
            attempt = 0
            while len(group_generations) < self.config.group_size or group_reward_entropy < self.config.min_group_entropy:
                tokenized_prompt = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
                generation, function_calls, generation_mask = self.generate(tokenized_prompt["input_ids"], use_old_model=True)
                generation_text = self.tokenizer.decode(generation[0])
                reward = self.reward_function(generation_text, n_input, function_calls)
                if len(group_generations) == self.config.group_size:
                    index_to_replace = most_common_element_index(group_rewards)
                    group_generations[index_to_replace] = generation
                    group_generation_masks[index_to_replace] = generation_mask
                    group_rewards[index_to_replace] = reward
                    print(f'replacement due to low entropy attempt {attempt}!!!!')
                    attempt += 1
                    if attempt > self.config.entropy_retries:
                        break
                else:
                    group_generations.append(generation)
                    group_generation_masks.append(generation_mask)
                    group_rewards.append(reward)
                group_reward_entropy = calculate_entropy(group_rewards)
            batch_group_reward_entropy.append(group_reward_entropy)
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
        batch_group_reward_entropy = torch.tensor(batch_group_reward_entropy, dtype=torch.float32)
        return batch_group_generations, batch_group_generation_masks, batch_group_rewards, batch_group_reward_entropy

    @torch.inference_mode()
    def generate(self, tokenized_prompt: torch.Tensor, use_old_model: bool = False) -> tuple[torch.Tensor, list[FunctionCall], torch.Tensor]:
        """
            Note to self: if you want to apply the tool stop to a batch, what you do is the following:
            you can either let it all run and then replace or
            take out the tool call, generate the rest and then complete the prompt with the tool call
        """
        model_inputs = tokenized_prompt.clone()
        function_calls = []
        generation_mask = [0] * tokenized_prompt.shape[1]
        while True:
            function_criteria = FunctionStopCriteria(self.tokenizer, tokenized_prompt.shape[1])
            model = self.old_model if use_old_model else self.model
            output = model.generate(
                model_inputs,
                max_length=self.config.generation_max_tokens,
                stopping_criteria=StoppingCriteriaList([function_criteria])
            )
            tokens_generated = output.shape[1] - model_inputs.shape[1]
            generation_mask += [1] * tokens_generated
            if function_criteria.function_call is None:
                # exit once the model exits without using a tool
                break
            model_inputs = torch.cat([model_inputs, output[:, -tokens_generated:]], dim=1)
            function_call = function_criteria.function_call
            self.call_function(function_call)
            function_calls.append(function_call)
            call_result_tokens = self.tokenizer(function_call.format_result(), return_tensors='pt')
            result_ids = call_result_tokens.input_ids.to(output.device)
            generation_mask += [0] * result_ids.shape[1]
            model_inputs = torch.cat([model_inputs, result_ids], dim=1)
        generation_mask = torch.tensor(generation_mask, dtype=torch.bool)
        return output, function_calls, rearrange(generation_mask, 'sequence -> 1 sequence')

    def get_answer(self, generation_text) -> str | None:
        generation_output = generation_text.rsplit('<|im_start|>assistant\n', 1)[-1]
        answer = self.answer_regex.search(generation_output)
        if answer:
            answer = answer.group(1)
            answer = answer.replace('<answer>', '').replace('</answer>', '').strip()
            return answer
        return None

    def reward_function(self, generation_text: str, number_inputs: tuple[int, int], function_calls: list[FunctionCall]):
        total_reward = 0
        answer = self.get_answer(generation_text)
        if answer is not None and answer.isnumeric():
            total_reward += self.config.output_format_reward
        if str(self.tool_multiply(*number_inputs)) == answer:
            total_reward += self.config.correctness_reward
        if function_calls:
            total_reward += self.config.call_reward
            function_call = function_calls[-1]
            if function_call.result is not None:
                total_reward += self.config.successful_call_reward
        return total_reward

    def objective_function(self, batch_group_generation, batch_group_generation_mask,
                           batch_group_rewards, shortest_prompt_size=0) -> torch.Tensor:
        batch_size = batch_group_generation.shape[0]

        advantages = self.calculate_advantages(batch_group_rewards)
        advantages = rearrange(advantages, "b g -> b g 1")
        batch_group_flat_generation = rearrange(batch_group_generation,
                                                'batch group sequence -> (batch group) sequence')

        logits = self.model(batch_group_flat_generation).logits
        logits = rearrange(logits, '(batch group) sequence vocab -> batch group sequence vocab', batch=batch_size)
        with torch.no_grad():
            old_logits = self.old_model(batch_group_flat_generation).logits
        old_logits = rearrange(old_logits, '(batch group) sequence vocab -> batch group sequence vocab',
                               batch=batch_size)

        logits = logits[:, :, shortest_prompt_size:, :]
        old_logits = old_logits[:, :, shortest_prompt_size:, :]
        batch_group_generation_mask = batch_group_generation_mask[:, :, shortest_prompt_size:]

        sequence_dim = -1
        batch_group_generation_unsqueezed = rearrange(batch_group_generation,
                                                      'batch group sequence -> batch group sequence 1')
        log_probs = torch.nn.functional.log_softmax(logits, dim=sequence_dim)
        log_probs = torch.gather(log_probs, dim=sequence_dim, index=batch_group_generation_unsqueezed)
        log_probs = rearrange(log_probs, 'batch group sequence 1 -> batch group sequence')
        old_log_probs = torch.nn.functional.log_softmax(old_logits, dim=sequence_dim)
        old_log_probs = torch.gather(old_log_probs, dim=sequence_dim, index=batch_group_generation_unsqueezed)
        old_log_probs = rearrange(old_log_probs, 'batch group sequence 1 -> batch group sequence')

        new_old_prob_ratio = torch.exp(log_probs - old_log_probs)
        if self.config.objective_clip_band is not None:
            unclipped_objective = new_old_prob_ratio * advantages
            clipped_objective = torch.clamp(
                unclipped_objective, 1 - self.config.objective_clip_band, 1 + self.config.objective_clip_band
            )
            unaggregated_objective = torch.min(unclipped_objective, clipped_objective)
        else:
            unaggregated_objective = new_old_prob_ratio * advantages

        if self.ref_model is not None:
            ref_logits = self.ref_model(batch_group_flat_generation).logits
            ref_logits = rearrange(ref_logits, '(batch group) sequence vocab -> batch group sequence vocab',
                                   batch=batch_size)
            ref_logits = ref_logits[:, :, shortest_prompt_size:, :]
            ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=sequence_dim)
            ref_log_probs = torch.gather(ref_log_probs, dim=sequence_dim, index=batch_group_generation_unsqueezed)
            ref_log_probs = rearrange(ref_log_probs, 'batch group sequence 1 -> batch group sequence')
            new_ref_prob_ratio = torch.exp(log_probs - ref_log_probs)
            divergence_kl = new_ref_prob_ratio - torch.log(new_ref_prob_ratio)

            unaggregated_objective -= self.config.dkl_weight * divergence_kl

        sequence_aggregated_objective = reduce(unaggregated_objective * batch_group_generation_mask,
                                               'batch group sequence -> batch group', 'sum')
        if not self.config.drgrpo:
            batch_group_generation_count = reduce(batch_group_generation_mask, 'batch group sequence -> batch group',
                                                  'sum')
            sequence_aggregated_objective /= batch_group_generation_count

        objective = sequence_aggregated_objective.mean()
        return objective

    def train(self):
        device = self.model.device
        batch_dim = 0
        for i in tqdm(range(self.rl_steps)):
            number_inputs = [
                (random.randint(*self.config.numbers_range), random.randint(*self.config.numbers_range))
                for _ in range(self.config.batch_size)
            ]
            (
                batch_group_generations,
                batch_group_generation_masks,
                batch_group_rewards,
                batch_group_reward_entropy,
            ) = self.produce_groups(number_inputs)

            print(f"Average reward: {batch_group_rewards.mean()}; batch_group_reward_entropy: {batch_group_reward_entropy}")
            if batch_group_reward_entropy.sum() < self.config.min_group_entropy:
                print("Skipping due to low entropy.")
                continue

            for epoch in range(self.config.epochs_per_step):
                index_permutation = torch.randperm(batch_group_generations.shape[batch_dim])
                batch_group_generations = batch_group_generations[index_permutation, :, :]
                batch_group_generation_masks = batch_group_generation_masks[index_permutation, :, :]
                batch_group_rewards = batch_group_rewards[index_permutation, :]

                for minibatch_start in range(0, batch_group_rewards.shape[batch_dim], self.config.minibatch_size):
                    minibatch_end = minibatch_start + self.config.minibatch_size
                    mini_batch_index = index_permutation[minibatch_start:minibatch_end]
                    minibatch_group_generations = batch_group_generations[mini_batch_index, :, :]
                    minibatch_group_generation_masks = batch_group_generation_masks[mini_batch_index, :, :].to(device)
                    minibatch_group_rewards = batch_group_rewards[mini_batch_index, :].to(device)

                    # This should be `-objective_function` but for some reason that makes the model not learn(?) there is
                    # some random minus flipping the `objective_function` somewhere
                    loss = self.objective_function(
                        minibatch_group_generations,
                        minibatch_group_generation_masks,
                        minibatch_group_rewards,
                    )

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clipping
                    )
                    self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()
            self.old_model = copy.deepcopy(self.model).eval()
            if i % 10 == 0:
                self.evaluate()

    @torch.inference_mode()
    def evaluate(self):
        number_inputs = [
            (
                random.randint(*self.config.numbers_range),
                random.randint(*self.config.numbers_range),
            )
            for _ in range(20)
        ]
        prompts = self.create_prompts(self.tokenizer, number_inputs)
        correct_predictions = 0
        accumulated_score = 0
        generation_text = ""
        for inputs, prompt in zip(number_inputs, prompts):
            tokenized_prompt = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
            generation, function_calls, _ = self.generate(tokenized_prompt.input_ids)
            generation_text = self.tokenizer.decode(generation[0])
            answer = self.get_answer(generation_text)
            accumulated_score += self.reward_function(generation_text, inputs, function_calls)
            if str(self.tool_multiply(*inputs)) == answer:
                correct_predictions += 1
        print(f"Sample Generation: {generation_text}")
        print(f"Accuracy: {correct_predictions / len(number_inputs)}")
        print(f"Average Reward: {accumulated_score / len(number_inputs)}")
        print("=" * 100)
