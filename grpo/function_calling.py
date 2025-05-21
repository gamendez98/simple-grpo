import re
from dataclasses import dataclass
from typing import Optional

from transformers import StoppingCriteria


FUNCTION_STOP_REGEX = r'`(\s*\w+\([^()]+?\))`$'

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
    BATCH_DIM = 0
    GROUP_DIM = 1
    SEQUENCE_DIM = 2
    TOKEN_DIM = -1

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