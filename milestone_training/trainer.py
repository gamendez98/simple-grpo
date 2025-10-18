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
    gradient_clipping: float = 0.5


class SimpleMilestoneTrainer:

    def __init__(
            self,
            model,
            data_loader: MilestoneDataLoader,
            config=SimpleMilestoneTrainerConfig()
    ):
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.optimizer = self.optimizer = torch.optim.AdamW(
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

    def train(self):
        self.model.train()
        for i in range(self.config.epochs):
            for tokens, conversation_scores in tqdm(self.data_loader.batch_generator(), total=len(self.data_loader)):
                conversation_scores = conversation_scores.to(self.model.device)
                conversation_scores = einops.rearrange(conversation_scores, 'b t -> b t 1')
                tokens = tokens.to(self.model.device)
                logits = self.model(**tokens).logits
                loss = - (logits * conversation_scores).sum()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            if self.scheduler:
                self.scheduler.step()
            if i % 10 == 0:
                self.evaluate()

    def evaluate(self):
        pass


def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    data_loader = MilestoneDataLoader(
        path=Path('data'),
        model_name=model_name,
        batch_size=1
    )
    smt = SimpleMilestoneTrainer(model=model, data_loader=data_loader)
    smt.train()


if __name__ == "__main__":
    main()
