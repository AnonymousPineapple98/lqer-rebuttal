import time

import tqdm
import torch


def profile(model, dataloader, num_samples, warmup_ratio=0.5):
    for device in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device)
    device = next(model.parameters()).device
    model.eval()
    generate_times = []
    bs = 0
    seq_len = 0
    with torch.inference_mode():
        for i, x in tqdm.tqdm(enumerate(dataloader), total=num_samples):
            x = x.to(device)
            torch.cuda.synchronize()
            start = time.time()
            bs = x.shape[0]
            seq_len = x.shape[1]
            out = model(x, use_cache=False)

            torch.cuda.synchronize()
            end = time.time()

            if i > num_samples * warmup_ratio:
                generate_times.append(end - start)

            if i >= num_samples - 1:
                break

    avg_gen_time = sum(generate_times) / len(generate_times)
    tokens_per_second = round(seq_len / avg_gen_time * bs, 2)

    total_memory_used = 0
    memory_pct = 100
    for device in range(torch.cuda.device_count()):
        memory_used = torch.cuda.max_memory_allocated(device) / (1024**3)
        total_memory_used += memory_used
        memory_pct = (
            memory_used
            / (torch.cuda.get_device_properties(device).total_memory / (1024**3))
            * 100
        )
        print(
            f"    Max Memory (device: {device}): {memory_used:.2f} GB ({memory_pct:.2f}%)"
        )
    print(f"Avg time per batch: {avg_gen_time} s")
    print(f"Token per second: {tokens_per_second}")

    return {
        "batch_size": bs,
        "seq_len": seq_len,
        "token_per_second": tokens_per_second,
        "total_max_vram": total_memory_used,
        "description": f"batch_size={bs}, \nseq_len={seq_len}, \ntoken_per_second={tokens_per_second}, \ntotal_max_vram_in_GB={total_memory_used}",
    }
