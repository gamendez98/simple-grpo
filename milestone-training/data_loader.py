# %%
import json
import os
from itertools import batched
from pathlib import Path
from typing import Any, Generator

import einops
import pandas as pd
import torch
from transformers import AutoTokenizer

# %%%
test_conversations = [
    [
        {
            "role": "system",
            "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."
        },
        {
            "role": "user",
            "content": "Add Stephen Sondheim to my contact, his phone_number is +19876543210"
        },
        {
            "role": "assistant",
            "content": "Do you want to specify a relationship for Stephen Sondheim in your contacts, or should I assume it's not necessary? Additionally, should I mark this contact as yourself (is_self: True)?",
            "milestone_gain": 0.9308869304625594,
            "milestone_similarity": 0.2924017592713677
        },
        {
            "role": "user",
            "content": "No need for a relationship or marking it as myself. Just add him."
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0sUUstevyx44EDs4fjS3ci8Q",
                    "function": {
                        "arguments": "{\"name\": \"Stephen Sondheim\", \"phone_number\": \"+19876543210\"}",
                        "name": "add_contact"
                    },
                    "type": "function"
                }
            ],
            "milestone_gain": 0.6384851711911917,
            "milestone_similarity": 0.2924017592713677
        },
    ],
    [
        {
            "role": "system",
            "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."
        },
        {
            "role": "user",
            "content": "Add a to do to buy chocolate milk 3/22/2024 5PM"
        },
        {
            "role": "assistant",
            "content": "Just to clarify, would you like me to set a reminder for this task at 5 PM on March 22, 2024?",
            "milestone_gain": 1.0,
            "milestone_similarity": 0.0
        },
        {
            "role": "user",
            "content": "Yes, exactly!"
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_2W2jgxf8BrudZ64JqUNl8JnQ",
                    "function": {
                        "arguments": "{\"year\": 2024, \"month\": 3, \"day\": 22, \"hour\": 17, \"minute\": 0, \"second\": 0}",
                        "name": "datetime_info_to_timestamp"
                    },
                    "type": "function"
                }
            ],
            "milestone_gain": 1.0,
            "milestone_similarity": 0.0
        },
        {
            "tool_call_id": "call_2W2jgxf8BrudZ64JqUNl8JnQ",
            "role": "tool",
            "name": "datetime_info_to_timestamp",
            "content": "1711144800.0",
            "tool_details": {
                "milestone_matches": [
                    {
                        "milestone_index": 0,
                        "milestone": {
                            "snapshot_constraints": [
                                {
                                    "database_namespace": "SANDBOX",
                                    "snapshot_constraint": "snapshot_similarity",
                                    "reference_milestone_node_index": None,
                                    "target_dataframe": [
                                        {
                                            "sender": "EXECUTION_ENVIRONMENT",
                                            "recipient": "AGENT",
                                            "tool_trace": "{\"tool_name\": \"datetime_info_to_timestamp\", \"arguments\": {}}"
                                        }
                                    ],
                                    "column_similarity_measure": {
                                        "sandbox_message_index": "column_exact_match_similarity",
                                        "sender": "column_exact_match_similarity",
                                        "recipient": "column_exact_match_similarity",
                                        "content": "column_rouge_l_similarity",
                                        "openai_tool_call_id": "column_one_similarity",
                                        "openai_function_name": "column_one_similarity",
                                        "conversation_active": "column_exact_match_similarity",
                                        "tool_call_exception": "column_one_similarity",
                                        "tool_trace": "column_tool_trace_exact_match_similarity",
                                        "visible_to": "column_exact_match_similarity"
                                    }
                                }
                            ],
                            "guardrail_database_list": [
                                "SANDBOX",
                                "SETTING",
                                "CONTACT",
                                "MESSAGING",
                                "REMINDER"
                            ],
                            "guardrail_database_exclusion_list": None
                        },
                        "milestone_similarity": 1.0
                    }
                ]
            }
        },
    ]
]

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

    def prepare_annotated_conversations(self, annotated_conversations: tuple[ConversationType]) -> tuple[
        torch.Tensor, torch.Tensor]:
        conversation_scores = []
        texts = self.tokenizer.apply_chat_template(
            annotated_conversations,
            tokenize=False
        )
        tokens = self.tokenizer(
            texts,
            return_offsets_mapping=True,
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
            df = pd.DataFrame({
                'token_ids': tokens['input_ids'][index].tolist(),
                'conversation': self.tokenizer.convert_ids_to_tokens(tokens['input_ids'][index]),
                'conversation_score': conversation_scores[-1]
            })
            print(df)

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

    def batch_generator(self) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        for annotated_chat_batch in batched(map(load_json, self.file_generator()), self.batch_size):
            tokens, conversation_scores = self.prepare_annotated_conversations(annotated_chat_batch)
            yield tokens, conversation_scores


def main():
    mdl = MilestoneDataLoader(path=Path('data'))

    for i, batch in zip(range(4), mdl.batch_generator()):
        df = pd.DataFrame({
            'input_ids': batch[0]['input_ids'][0].tolist(),
            'conversation_scores': batch[1].tolist()
        })
        print(df)


if __name__ == "__main__":
    main()
