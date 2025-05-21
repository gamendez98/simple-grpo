from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from grpo.GRPO import ToolGRPOTrainer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        device_map="auto"
    )


    tgrpo_trainer = ToolGRPOTrainer(model, tokenizer)

    try:
        tgrpo_trainer.train()
    except KeyboardInterrupt:
        pass



if __name__ == "__main__":
    main()