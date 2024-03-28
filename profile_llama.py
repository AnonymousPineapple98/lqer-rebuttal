import torch
from awq import AutoAWQForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate.utils import BnbQuantizationConfig

from lqer_cuda_perf.models import (
    LlamaQDecoderLayer,
    QDecoder,
    LlamaQModelForCausalLM,
)
from lqer_cuda_perf.profile import profile


def create_llm_int4_model(
    model_name, num_hidden_layers: int = None, start_idx: int = 0, device="cuda"
):
    bnb_q_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_q_config
    )
    if num_hidden_layers is not None:
        layers = model.model.model.layers
        truncated_layers = [
            layers[i] for i in range(start_idx, start_idx + num_hidden_layers)
        ]
        model.model.model.layers = torch.nn.ModuleList(truncated_layers)
    return model, model.config

def create_llm_int8_model(
    model_name, num_hidden_layers: int = None, start_idx: int = 0, device="cuda"
):
    bnb_q_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_q_config
    )
    if num_hidden_layers is not None:
        layers = model.model.model.layers
        truncated_layers = [
            layers[i] for i in range(start_idx, start_idx + num_hidden_layers)
        ]
        model.model.model.layers = torch.nn.ModuleList(truncated_layers)
    return model, model.config

def create_awq_model(
    model_name, num_hidden_layers: int = None, start_idx: int = 0, device="cuda"
):
    model = AutoAWQForCausalLM.from_pretrained(model_name)

    if num_hidden_layers is not None:
        layers = model.model.model.layers
        truncated_layers = [
            layers[i] for i in range(start_idx, start_idx + num_hidden_layers)
        ]
        model.model.model.layers = torch.nn.ModuleList(truncated_layers)
    model.to(device)
    return model, model.config


def create_fp8fp16_model(
    model_name,
    num_hidden_layers: int = None,
    start_idx: int = 0,
    device="cuda",
    rank: int = 4,
    max_seq_len: int = 2048,
):
    config = AutoConfig.from_pretrained(model_name)
    config.max_seq_len = max_seq_len

    if num_hidden_layers is None:
        num_hidden_layers = config.num_hidden_layers

    decoder_kwargs_list = []
    # model arch args
    for i in range(start_idx, start_idx + num_hidden_layers):
        decoder_kwargs_list.append(
            dict(
                config=config,
                layer_idx=i,
                q_recipe=dict(name="fp8fp16"),
                rank=rank,
                device=device,
            )
        )

    decoder = QDecoder(
        torch.nn.ModuleList(
            [
                LlamaQDecoderLayer(**decoder_kwargs_list[i])
                for i in range(num_hidden_layers)
            ]
        )
    )
    model = LlamaQModelForCausalLM(
        config=config,
        decoder=decoder,
        device=device,
    )
    return model, config


def create_fp16_model(
    model_name,
    num_hidden_layers: int = None,
    start_idx: int = 0,
    device="cuda",
    max_seq_len: int = 2048,
):
    config = AutoConfig.from_pretrained(model_name)
    config.max_seq_len = max_seq_len
    if num_hidden_layers is None:
        num_hidden_layers = config.num_hidden_layers

    decoder_kwargs_list = []
    # model arch args
    for i in range(start_idx, start_idx + num_hidden_layers):
        decoder_kwargs_list.append(
            dict(
                config=config,
                layer_idx=i,
                q_recipe=dict(name="fp16"),
                rank=None,
                device=device,
            )
        )

    decoder = QDecoder(
        torch.nn.ModuleList(
            [
                LlamaQDecoderLayer(**decoder_kwargs_list[i])
                for i in range(num_hidden_layers)
            ]
        )
    )
    model = LlamaQModelForCausalLM(
        config=config,
        decoder=decoder,
        device=device,
    )
    return model, config


if __name__ == "__main__":
    from argparse import ArgumentParser
    from pathlib import Path
    import json

    parser = ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("q_recipe", choices=["fp16", "lqer-fp8fp16", "llm-int4", "llm-int8", "awq"])
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    parser.add_argument("--max-seq-len", dest="max_seq_len", type=int)
    parser.add_argument("--num-samples", dest="num_samples", type=int, default=256)
    parser.add_argument("--warmup-ratio", dest="warmup_ratio", type=float, default=0.5)
    parser.add_argument(
        "--num-hidden-layers", dest="num_hidden_layers", type=int, default=-1
    )
    parser.add_argument(
        "--layer-start-idx", dest="layer_start_idx", type=int, default=0
    )
    parser.add_argument("--rank", dest="rank", type=int, default=32)
    args = parser.parse_args()
    if args.num_hidden_layers < 1:
        args.num_hidden_layers = None
    device = "cuda"
    print("=" * 80)
    print(
        f"Profiling Starts: {args.model}, q_recipe={args.q_recipe}, batch_size={args.batch_size}, max_seq_len={args.max_seq_len}, num_layers={args.num_hidden_layers}, rank={args.rank if args.q_recipe in ['lqer-fp8fp16'] else 'NA'}"
    )
    try:
        match args.q_recipe:
            case "fp16":
                model, config = create_fp16_model(
                    args.model,
                    args.num_hidden_layers,
                    args.layer_start_idx,
                    device=device,
                    max_seq_len=args.max_seq_len,
                )
            case "llm-int4":
                model, config = create_llm_int4_model(
                    args.model,
                    args.num_hidden_layers,
                    args.layer_start_idx,
                    device=device,
                )
            case "llm-int8":
                model, config = create_llm_int8_model(
                    args.model,
                    args.num_hidden_layers,
                    args.layer_start_idx,
                    device=device,
                )
            case "awq":
                model, config = create_awq_model(
                    args.model,
                    args.num_hidden_layers,
                    args.layer_start_idx,
                    device=device,
                )
            case "lqer-fp8fp16":
                model, config = create_fp8fp16_model(
                    args.model,
                    args.num_hidden_layers,
                    args.layer_start_idx,
                    device=device,
                    rank=args.rank,
                    max_seq_len=args.max_seq_len,
                )
            case _:
                raise RuntimeError(f"Unknown q_recipe: {args.q_recipe}")

        dataloader = [
            torch.randint(
                0,
                config.vocab_size,
                (args.batch_size, args.max_seq_len),
                dtype=torch.int64,
                device="cpu",
            )
            for _ in range(args.num_samples)
        ]

        results = profile(model, dataloader, args.num_samples, args.warmup_ratio)
    except RuntimeError as ex:
        if "cuda out of memory" in str(ex).lower():
            results = {
                "batch_size": args.batch_size,
                "seq_len": args.max_seq_len,
                "token_per_second": -1,
                "total_max_vram": -1,
                "description": f"batch_size={args.batch_size}, \nseq_len={args.max_seq_len}, \ntoken_per_second=NA, \ntotal_max_vram_in_GB=OOM",
            }
        else:
            raise RuntimeError(ex)

    save_path = f"{args.model.replace('/', '-')}_{args.q_recipe}_bs-{args.batch_size}_max-seq-len-{args.max_seq_len}_num-layers-{args.num_hidden_layers}_rank-rrr.json"
    if args.q_recipe in ["lqer-fp8fp16"]:
        save_path = save_path.replace("_rank-rrr", f"_rank-{args.rank}")
    else:
        save_path = save_path.replace("_rank-rrr", "")

    save_dir = Path(f"./results/{args.q_recipe}")
    save_dir.mkdir(parents=True, exist_ok=True)

    results["model"] = args.model
    results["q_recipe"] = args.q_recipe
    results["num_hidden_layers"] = args.num_hidden_layers
    results["rank"] = args.rank if args.q_recipe in ["lqer-fp8fp16"] else -1

    with open(save_dir / save_path, "w") as f:
        json.dump(results, f)
    print(results["description"])
