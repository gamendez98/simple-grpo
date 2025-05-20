from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from GRPO.GRPO import ToolGRPOTrainer
import einops


#%%

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

#%%

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    device_map="auto"
)

#%%

tgrpo_trainer = ToolGRPOTrainer(model, tokenizer)

#%%

import random

inputs = [(random.randint(10, 100), random.randint(10, 100)) for _ in range(2)]

all_group_generations, all_group_generation_masks, all_group_rewards = tgrpo_trainer.produce_groups(inputs, 4)

#%%

print(all_group_generations.shape)
print(all_group_generation_masks.shape)
print(all_group_rewards.shape)

#%%

model(einops.rearrange(all_group_generations, 'batch group sequence -> (batch group) sequence'))


