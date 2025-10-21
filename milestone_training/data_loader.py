# %%
import json
import os
from itertools import batched
from pathlib import Path
from typing import Any, Generator

import einops
import torch
from transformers import AutoTokenizer

# %%%
ConversationType = list[list[dict[str, Any]]]


def load_json(file_path: Path) -> dict[str, Any]:
    with open(file_path) as f:
        return json.load(f)


# %%

class MilestoneDataLoader:

    def __init__(
            self,
            path: Path,
            expected_file_name='annotated_conversation.json',
            batch_size: int = 4,
            model_name: str = 'Qwen/Qwen2.5-0.5B-Instruct',
            message_start_token: str = '<|im_start|>'
    ):
        self.batch_size = batch_size
        self.path = path
        self.expected_file_name = expected_file_name
        self.message_start_token = message_start_token
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.message_start_token_id = self.tokenizer.convert_tokens_to_ids(self.message_start_token)
        self.length = None

    def prepare_annotated_conversations(self, annotated_conversations: tuple[ConversationType]) -> tuple[
        torch.Tensor, torch.Tensor]:
        conversation_scores = []
        texts = self.tokenizer.apply_chat_template(
            annotated_conversations,
            tokenize=False
        )
        tokens = self.tokenizer(
            texts,
            return_tensors='pt',  # <- this makes it return PyTorch tensors
            padding=True,  # optional
            truncation=True  # optional
        )
        tensor_size = tokens['input_ids'].shape[1]
        for index, conversation in enumerate(annotated_conversations):
            # noinspection PyTypeChecker
            message_start_indices = einops.rearrange(
                torch.nonzero(tokens['input_ids'][index] == self.message_start_token_id, as_tuple=False),
                'm 1 -> m'
            )
            message_roles = [messages.get('role') for messages in annotated_conversations[index]]
            message_scores = [message.get('milestone_gain', 0) for message in annotated_conversations[index]]
            clean_scores = [s * (r == 'assistant') for s, r in zip(message_scores, message_roles)]
            clean_scores += [0]
            message_lengths = [i for i in torch.diff(message_start_indices, dim=0).tolist()]
            message_lengths += [tensor_size - message_start_indices[-1]]
            extended_clean_scores = []
            for cs, ml in zip(clean_scores, message_lengths):
                extended_clean_scores += [0, 0, 0] + [cs] * (ml-5) + [0, 0] # The 0 padding is for eliminating the prompt conversation formating
            conversation_scores.append(extended_clean_scores)

        return tokens, torch.tensor(conversation_scores) * tokens['attention_mask']

    def file_generator(self, entry_path: Path = None) -> Generator[Path, None, None]:
        """
        Recursively yields paths of files in `root_path` (and subdirectories)
        that have the given `target_name`.
        """
        entry_path = entry_path or self.path
        for entry in os.scandir(entry_path):
            if entry.is_file() and entry.name == self.expected_file_name:
                yield Path(entry.path)
            elif entry.is_dir():
                yield from self.file_generator(Path(entry.path))

    def __len__(self):
        if self.length is None:
            self.length = sum(1 for _ in self.file_generator())
        return self.length


    def batch_generator(self) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        for annotated_chat_batch in batched(map(load_json, self.file_generator()), self.batch_size):
            tokens, conversation_scores = self.prepare_annotated_conversations(annotated_chat_batch)
            yield tokens, conversation_scores


def main():
    import pandas as pd
    mdl = MilestoneDataLoader(path=Path('data'))

    for i, batch in zip(range(4), mdl.batch_generator()):
        df = pd.DataFrame({
            'input_ids': batch[0]['input_ids'][0].tolist(),
            'conversation_scores': batch[1].tolist()
        })
        print(df)


if __name__ == "__main__":
    main()
