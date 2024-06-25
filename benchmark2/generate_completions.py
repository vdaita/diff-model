import datasets
from pathlib import Path
from typing import List, Callable, TypeVar
from tqdm import tqdm
import torch
from typing import List, Literal, Optional, TypedDict, Callable, Union
import gzip
import json
import os
import uuid

def gunzip_json_write(path: Path, data: dict) -> None:
    with gzip.open(path, "wt") as f:
        json.dump(data, f)


T = TypeVar("T")


def batch_prompts_from_example(example: T, batch_size: int, completion_limit: int) -> List[List[T]]:
    prompts = [example] * completion_limit
    num_batches = completion_limit // batch_size
    batches = [prompts[i * batch_size: (i + 1) * batch_size]
               for i in range(num_batches)]
    # the last batch may be smaller
    if len(prompts) % batch_size != 0:
        batches.append(prompts[num_batches * batch_size:])

    return batches


Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str
    # the prefix for the assistant's response. this is only used for the
    # last message in a conversation, and is ignored otherwise.
    # NOTE: leaving commented for python 3.10 compatibility
    #  prefix_after: NotRequired[str]


class EditCommand(TypedDict):
    instruction: Optional[str]
    content: str


class EditResponse(TypedDict):
    instruction: Optional[str]
    content: str


# (old, instr, new) -> prompt
PromptFormatFunction = Callable[[str, str, str], str]

# (old, instr) -> [messages]
MessagesFormatFunction = Callable[[str, str], List[Message]]

# (old, new) -> response
PostProcessFunction = Callable[[str, str], str]


CompletionEngine = Literal["vllm", "transformers"]


def starcoder_edit_prompt(old, instr, new):
    # starcoder tokens
    OLD_CODE_TOKEN = "<commit_before>"
    REFLECTION_TOKEN = "<commit_msg>"
    NEW_CODE_TOKEN = "<commit_after>"
    return OLD_CODE_TOKEN + old + REFLECTION_TOKEN + instr + NEW_CODE_TOKEN + new


def direct_edit_prompt(
    old,
    instr,
    new,
    codeblock_before: Optional[str] = None,
    codeblock_after: Optional[str] = None,
):
    """
    The codeblock_before and codeblock_after arguments are used to specify
    if there should be a codeblock surrounding the code before and after
    the instruction. If None, then no codeblock is used. The string is the
    extension of the codeblock, e.g. "py" or "md".
    """
    if codeblock_before is not None:
        old = f"```{codeblock_before}\n{old}\n```"
    if codeblock_after is not None:
        new = f"```{codeblock_after}\n{new}\n```"
    before = f"""## Code Before:\n{old}\n"""
    instr = f"""## Instruction:\n{instr}\n"""
    after = f"""## Code After:\n{new}"""
    return before + instr + after


def openai_edit_prompt_1shot(old: str, instr: str) -> List[Message]:
    return [
        {
            "role": "system",
            "content": """
You are PythonEditGPT. You will be provided the original code snippet and an instruction that specifies the changes you need to make. You will produce the changed code, based on the original code and the instruction given. Only produce the code, do not include any additional prose.
             """.strip(),
        },
        {
            "role": "user",
            "content": """
## Code Before
```py
def add(a, b):
    return a + b
```

## Instruction
Add a "sub" function that subtracts two numbers. Also write docstrings for both functions and change a,b to x,y.
""".strip(),
        },
        {
            "role": "assistant",
            "content": """
## Code After
```py
def add(x, y):
    \"\"\"Adds two numbers.\"\"\"
    return x + y

def sub(x, y):
    \"\"\"Subtracts two numbers.\"\"\"
    return x - y
```
         """.strip(),
        },
        {
            "role": "user",
            "content": f"""
## Code Before
```py
{old}
```
## Instruction
{instr}
""".strip(),
        },
    ]


def direct_edit_prompt_1shot(
    old,
    instr,
    new,
):
    p = direct_edit_prompt(old, instr, new)
    shot = """## Code Before:
def add(a, b):
    return a + b
## Instruction:
Add a "sub" function that subtracts two numbers. Also write docstrings for both functions and change a,b to x,y.
## Code After:
def add(x, y):
    \"\"\"Adds two numbers.\"\"\"
    return x + y

def sub(x, y):
    \"\"\"Subtracts two numbers.\"\"\"
    return x - y"""
    p = shot + "\n" + p
    return p


def starcoder2_edit_prompt_1shot(old: str, instr: str, _: str) -> str:
    return f"""<issue_start>username_0: I have a program in Python that I'd like to change.

Here is the code for the program:
```py
def add(a, b):
    return a + b
```

The change I'd like to make is:
Add a "sub" function that subtracts two numbers. Also write docstrings for both functions and change a,b to x,y.

Please someone help me. Can you also provide the full code with the change?<issue_comment>username_1: Sure, no problem. I will be able to help. I am an expert in editing Python code.

Here is the full code with the change:
```py
def add(x, y):
    \"\"\"Adds two numbers.\"\"\"
    return x + y

    def sub(x, y):
    \"\"\"Subtracts two numbers.\"\"\"
    return x - y
```
Upvotes: 200<issue_comment>username_0: Thank you so much! I have another program in Python that I'd like to change.

Here is the code for the program:
```py
{old}
```

The change I'd like to make is:
{instr}

Please someone help me. Can you also provide the full code with the change?
Upvotes: 100<issue_comment>username_1: Sure, no problem. I will be able to help. I am an expert in editing Python code.

Here is the full code with the change:
```py"""


def python_markdown_codeblock_extract(_: str, new: str) -> str:
    lines = new.split("\n")
    buf = ""
    in_codeblock = False
    for ln in lines:
        if ln.startswith("```"):
            if in_codeblock:
                break
            else:
                in_codeblock = True
        elif in_codeblock:
            buf += ln + "\n"
    return buf


def autodetect_dtype() -> str:
    if torch.cuda.is_bf16_supported():
        return "bfloat16"
    else:
        return "auto"


def vllm_get_tokenizer(model):
    tokenizer = model.get_tokenizer()
    # oh vLLM... how you have fallen..
    if tokenizer.__class__.__name__ == "TokenizerGroup":
        tokenizer = tokenizer.tokenizer

    return tokenizer

class ChatModel:
    def generate(self, messages: List[List[Message]], **kwargs) -> List[str]:
        raise NotImplementedError


def chatmodel_factory(model_type, model_name, num_gpus):
    if model_type == "llama":
        return HFChatModel(model_name, num_gpus)
    else:
        raise ValueError(f"Unknown chat model type {model_type}")


class EditModel:
    def __init__(self, before_content_tok=None, instruction_tok=None, after_content_tok=None):
        self.before_content_tok = before_content_tok
        self.instruction_tok = instruction_tok
        self.after_content_tok = after_content_tok

    def edit_model_generate(
        self,
        model: Union[LLM, TransformersVLLMAdapter],
        str_prompts: List[str], **kwargs
    ) -> List[RequestOutput]:
        kwargs_gen = kwargs.copy()
        if "declaration" in kwargs_gen:
            del kwargs_gen["declaration"]
        use_tqdm = kwargs_gen.pop("use_tqdm", False)
        gens = model.generate(
            prompts=str_prompts,
            sampling_params=SamplingParams(
                top_p=kwargs_gen.pop("top_p", 0.95),
                temperature=kwargs_gen.pop("temperature", 0.2),
                max_tokens=kwargs_gen.pop("max_tokens", 1024),
                **kwargs_gen,
            ),
            use_tqdm=use_tqdm,
        )
        return gens

    def get_before_content_tok(self) -> Optional[str]:
        return self.before_content_tok

    def get_instruction_tok(self) -> Optional[str]:
        return self.instruction_tok

    def get_after_content_tok(self) -> Optional[str]:
        return self.after_content_tok

    def generate(self, prompts: List[EditCommand], **kwargs) -> List[EditResponse]:
        raise NotImplementedError

    def bugfix_instr(self, prompt) -> Optional[str]:
        return None

    def get_prompt_format(self):
        raise NotImplementedError

    def get_tokenizer(self):
        raise NotImplementedError


def editmodel_factory(model_type, model_name, num_gpus):
    if model_type == "starcoder":
        return StarCoderCommitEditModel(model_name, num_gpus)
    else:
        raise ValueError(f"Unknown edit model type {model_type}")

def init_completion_engine(engine: CompletionEngine, **kwargs):
    if engine == "vllm":
        extra_kwargs = {}
        if "max_model_len" in kwargs:
            extra_kwargs["max_model_len"] = kwargs["max_model_len"]
        return LLM(
            kwargs["model_name"],
            dtype=autodetect_dtype(),
            tensor_parallel_size=kwargs["num_gpus"],
            gpu_memory_utilization=kwargs["gpu_util"],
            **extra_kwargs,
        )
    elif engine == "transformers":
        return TransformersVLLMAdapter(kwargs["model_name"])
    else:
        raise ValueError(f"Unknown completion engine {engine}")

from unsloth import FastLanguageModel

class DiffEditModel(EditModel):
    def __init__(
        self,
        path="vdaita/diff-starcoder-7b-rl",
        num_gpus=1,
        quantization=True
    ):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "vdaita/diff-codegemma-7b",
            max_seq_length = 8192,
            load_in_4bit = True,
        )
        FastLanguageModel.for_inference(model)

        self.model = model
        self.tokenizer = tokenizer

        # self.generator = pipeline("text-generation", model=path, device_map="auto", model_kwargs={"load_in_8bit": True})

    def generate(self, prompts: List[EditCommand], **kwargs) -> List[EditResponse]:
        import diff_utils
        kwargs = kwargs.copy()

        chat_prompts = []
        for prompt in prompts:
            chat_prompts.append(
                f"""# Filename: main.py\n# File:\n{prompt["content"]}\n#Instructions:\n{prompt["instruction"]}\n# Patch:\n"""
            )
        
        chat_prompts = self.tokenizer(chat_prompts, tokenize=True, return_tensors="pt").to("cuda")

        outputs = self.model.generate(input_ids=chat_prompts, max_new_tokens=1000, use_cache=True)
        edited_files = []

        for (ufmt_output, prompt) in zip(outputs, prompts):
            output = ufmt_output[0]["generated_text"]
            
            if not(os.path.exists("completions")):
                os.mkdir("completions")
            
            file_id = str(uuid.uuid4())
            file = open(f"completions/{file_id}.txt", "w+")
            file.write(f"# Content: {prompt['content']} \n \n # Instruction: {prompt['instruction']} \n \n # Generated output: \n {output}")
            file.close()

            sr_blocks = diff_utils.parse_diff(output)
            content = prompt["content"]
            actual_searches = [diff_utils.find_best_match(block.search_block, content).block for block in sr_blocks]
            for (block, actual_search) in zip(sr_blocks, actual_searches):
                content = content.replace(actual_search, block.replace_block)
            edited_files.append(EditResponse(instruction=prompt["instruction"], content=f"```python\n{content}\n```"))
            print("Ratio: ", len(output) / len(prompt["content"]), " | ",  len(output) / len(content))

        return edited_files

class ChatAdaptorEditModel(EditModel):
    """
    This is an adaptor class to use ChatModels as EditModels.
    NOTE: This model class is only intended for inference, not training.
    """

    # TODO: implement whole shebang for bugfix

    def __init__(
        self,
        chat_model: ChatModel,
        prompt_format: MessagesFormatFunction = openai_edit_prompt_1shot,
        post_process: PostProcessFunction = python_markdown_codeblock_extract,
    ):
        super().__init__()
        self.model = chat_model
        self.prompt_format = prompt_format
        self.post_process = post_process

    def generate(self, prompts: List[EditCommand], **kwargs) -> List[EditResponse]:
        kwargs = kwargs.copy()
        # TODO: can do something with declaration here
        kwargs.pop("declaration", None)

        kwargs.pop("use_tqdm", None)

        chat_prompts = []
        for prompt in prompts:
            assert (
                prompt["instruction"] is not None
            ), "Every command must have an instruction in ChatAdaptorEditModel"
            chat_prompts.append(
                self.prompt_format(prompt["content"], prompt["instruction"])
            )

        # generate
        gens = self.model.generate(chat_prompts, **kwargs)

        responses = []
        for prompt, gen in zip(prompts, gens):
            processed = self.post_process(prompt["content"], gen)
            resp = {"content": processed, "instruction": None}
            responses.append(resp)

        return responses


# NOTE: this is the factory for each model type. to add a new model type, add a new case here
# and implement it in models.py. Also, add a new case in the argument parser below.
def model_factory(
        model_type: str,
        quantize=False,
        num_gpus=1,
        system_supported=True,
        completion_engine: CompletionEngine = "vllm",
        max_model_len=None,
) -> Callable[[str], EditModel]:
    return (lambda path: DiffEditModel(path=path, quantization=quantize, num_gpus=num_gpus))

def complete_problem(example: EditCommand, model: EditModel, batch_size: int, completion_limit: int, **kwargs) -> List[str]:
    batches = batch_prompts_from_example(example, batch_size, completion_limit)

    completions = []
    for batch in batches:
        resps = model.generate(batch, **kwargs)
        for resp in resps:
            completions.append(resp["content"])

    return completions


def main(args):
    dataset = datasets.load_dataset(
        args.dataset, args.subset, split=args.split)
    
    model = model_factory(
        args.model_type,
        quantize=args.quantize,
        num_gpus=args.num_gpus,
        system_supported=not args.no_system,
        completion_engine=args.completion_engine,
        max_model_len=args.max_model_len,
    )(args.model)
    model_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }

    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True)

    # writing in this format such that we can use the MultiPL-E evaluation container :)
    for ex in tqdm(dataset, total=len(dataset)):  # type: ignore
        assert isinstance(ex, dict)

        if args.humanevalpack:
            instr_kinds = ['instruction']
        else:
            instr_kinds = ['instruction_descriptive', 'instruction_lazy']

        for instr_kind in instr_kinds:
            path = Path(args.output_dir) / \
                (f"{ex['full_name']}_{instr_kind}.json.gz")

            print("Checking path: ", path)
            if path.exists():
                print("Path exists: ", path)
                continue  # this pretty much resumes from where it left off

            instr = ex[instr_kind]
            example = EditCommand(
                instruction=instr,
                content=ex["before"],
            )

            if "declaration" in ex:
                model_kwargs["declaration"] = ex["declaration"]

            completions = complete_problem(
                example,
                model,
                args.batch_size,
                args.completion_limit,
                **model_kwargs,
            )

            # copy over the example
            result = {}
            for k in ex:
                result[k] = ex[k]

            result["instr_kind"] = instr_kind
            # this is for compatibility with the MultiPL-E evaluator
            result["prompt"] = ex["declaration"] if "declaration" in ex else ""
            result["completions"] = completions
            result["language"] = "py"
            result["temperature"] = args.temperature
            result["top_p"] = args.top_p
            result["max_tokens"] = args.max_tokens
            result["stop_tokens"] = []
            result["script_args"] = args.__dict__.copy()

            gunzip_json_write(path, result)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="nuprl/CanItEdit", help="dataset to use")
    parser.add_argument("--split", type=str, default="test",
                        help="split of the dataset to use")
    parser.add_argument("--subset", type=str, default=None,
                        help="subset of the split to use")
    parser.add_argument(
        "--model-type",
        type=str,
        default="direct",
        choices=["direct", "direct-1shot",
                 "openai", "chat", "octocoder", "starcoder", "starcoder2", "diffcoder"],
        help="type of model to use for completions",
    )
    parser.add_argument(
        "--completion-engine",
        type=str,
        default="vllm",
        choices=["vllm", "transformers"],
    )
    parser.add_argument("--model", type=str, required=True,
                        help="path to model or hub name")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="output directory for completions")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="batch size for completions")
    parser.add_argument("--completion-limit", type=int,
                        default=20, help="number of completions per prompt")
    parser.add_argument("--temperature", type=float,
                        default=0.2, help="sampling temperature")
    parser.add_argument("--quantize", action="store_true",
                        help="quantize the model with AWQ")
    parser.add_argument("--humanevalpack", action="store_true",
                        help="run humanevalpack instead of CanItEdit")
    parser.add_argument("--top-p", type=float,
                        default=0.95, help="top-p sampling")
    parser.add_argument("--max-tokens", type=int,
                        default=2048, help="max new tokens to generate per completion. 2048 works for CanItEdit")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="max model length for batching with vLLM. only change if getting OOM")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="number of gpus for sharded model")
    parser.add_argument("--no-system", action="store_true",
                        help="disable system prompt for chat models")
    args = parser.parse_args()
    main(args)
