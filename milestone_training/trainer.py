from dataclasses import dataclass
from pathlib import Path

import einops
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup

from milestone_training.data_loader import MilestoneDataLoader


@dataclass
class SimpleMilestoneTrainerConfig:
    epochs: int = 10
    policy_lr: float = 1e-4
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_weight_decay: float = 0.01
    gradient_clipping: float = 0.1


class SimpleMilestoneTrainer:

    def __init__(
            self,
            model,
            data_loader: MilestoneDataLoader,
            config=None
    ):
        self.model = model
        self.data_loader = data_loader
        self.config = config or SimpleMilestoneTrainerConfig()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.policy_lr,
            betas=self.config.adam_betas,
            weight_decay=self.config.adam_weight_decay,
        )
        self.gradient_clipping = 0.5
        num_training_steps = len(self.data_loader) * self.config.epochs
        num_warmup_steps = int(0.05 * num_training_steps)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def calculate_loss(self, input_ids: torch.Tensor, logits: torch.Tensor, conversation_scores: torch.Tensor):
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        input_ids = einops.rearrange(input_ids, 'b t -> b t 1')
        token_log_probs = log_probs.gather(dim=-1, index=input_ids)
        token_log_probs = einops.rearrange(token_log_probs, 'b t 1 -> b t')
        loss = -torch.sum(token_log_probs * conversation_scores) / conversation_scores.sum()
        return loss

    def train(self):
        self.model.train()
        for i in range(self.config.epochs):
            for tokens, conversation_scores in tqdm(self.data_loader.batch_generator(), total=len(self.data_loader) // self.data_loader.batch_size):
                conversation_scores = conversation_scores.to(self.model.device)
                tokens = tokens.to(self.model.device)
                input_ids_x = tokens.input_ids[:,:-1]
                atta_mask_x = tokens.attention_mask[:,:-1]
                input_ids_y = tokens.input_ids[:, 1:]
                logits = self.model(input_ids = input_ids_x, attention_mask = atta_mask_x).logits

                self.optimizer.zero_grad()
                loss = self.calculate_loss(input_ids_y, logits, conversation_scores[:, 1:])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
                self.optimizer.step()
                self.scheduler.step()

            if i % 10 == 0:
                self.evaluate()

    def test(self, conversation: list[dict]):
        model = self.model
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        model.eval()
        tokenizer = self.data_loader.tokenizer
        text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        print("=" * 100)
        print("=" * 100)
        print(f"```{text}```")
        print("=" * 50 + "GENERATION" + "=" * 50)
        model_inputs = tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=300
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        output_text = tokenizer.decode(output_ids).strip()
        print(f"```{output_text}```")
        print("=" * 100)
        print("=" * 100)
        return output_text

    def evaluate(self):
        pass


def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map={"":2}
    )
    data_loader = MilestoneDataLoader(
        path=Path('data/full_scenarios_tools'),
        model_name=model_name,
        batch_size=2
    )
    smt = SimpleMilestoneTrainer(
        model=model,
        data_loader=data_loader,
        config=SimpleMilestoneTrainerConfig(epochs=1)
    )
    test_messages = [
        {
            "role": "system",
            "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."
        },
        {
            "role": "user",
            "content": "What is the address of lattitude: 37.334606, longitude: -122.009102"
        }
    ]
    smt.test(test_messages)
    smt.train()
    smt.test(test_messages)


if __name__ == "__main__":
    main()
