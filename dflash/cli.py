from __future__ import annotations

import argparse


def _add_inference_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("backend", choices=("transformers", "mlx", "openai"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft")
    parser.add_argument(
        "--secondary-model",
        help="Pretrained token-only suffix drafter (Transformers, temperature 0)",
    )
    parser.add_argument("--draft-bits", type=int, choices=[4, 8])
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="0 for greedy decoding"
    )
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling")
    parser.add_argument("--top-k", type=int, default=0, help="0 disables top-k")
    parser.add_argument(
        "--reasoning",
        help="off/on or a model-supported reasoning level",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--timeout-s", type=int, default=3600)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dflash")
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="Generate one response")
    _add_inference_arguments(generate)
    generate.add_argument("prompt")
    generate.add_argument("--system")
    generate.add_argument(
        "--check-greedy", action="store_true",
        help="Compare output IDs with independent target-only greedy generation",
    )

    benchmark = commands.add_parser("benchmark", help="Run a benchmark")
    _add_inference_arguments(benchmark)
    benchmark.add_argument("--dataset", required=True)
    benchmark.add_argument("--max-samples", type=int)
    benchmark.add_argument("--num-prompts", type=int, default=1024)
    benchmark.add_argument("--concurrency", type=int, default=1)
    return parser


def _messages(args) -> list[dict]:
    messages = []
    if args.system is not None:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    return messages


def _generate_transformers(args) -> None:
    import torch

    from .benchmark import (
        apply_chat_template,
        load_transformers_models,
        stop_token_ids,
    )
    from .model import dflash_generate
    from .secondary import load_secondary_model

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    target, draft, tokenizer = load_transformers_models(
        args.model, args.draft, device
    )
    secondary = (
        load_secondary_model(args.secondary_model, target, tokenizer)
        if args.secondary_model else None
    )
    prompt = apply_chat_template(tokenizer, _messages(args), args.reasoning)
    input_ids = tokenizer.encode(
        prompt, return_tensors="pt", add_special_tokens=False
    ).to(device)
    output_ids = dflash_generate(
        draft,
        target,
        input_ids,
        args.max_new_tokens,
        stop_token_ids(target, tokenizer),
        args.temperature,
        args.top_p,
        args.top_k,
        block_size=args.block_size,
        secondary_model=secondary,
    )
    if args.check_greedy:
        import sys

        from .validation import check_greedy_output

        check_greedy_output(
            target, input_ids, output_ids, args.max_new_tokens,
            stop_token_ids(target, tokenizer),
        )
        print("Greedy check passed: exact target-only token match.", file=sys.stderr)
    print(
        tokenizer.decode(output_ids[0, input_ids.shape[1] :], skip_special_tokens=True)
    )


def _generate_mlx(args) -> None:
    import mlx.core as mx

    from .benchmark import apply_chat_template, load_mlx_models
    from .model_mlx import stream_generate

    mx.random.seed(0)
    model, draft, tokenizer = load_mlx_models(
        args.model, args.draft, args.draft_bits
    )
    prompt = apply_chat_template(tokenizer, _messages(args), args.reasoning)
    for response in stream_generate(
        model,
        draft,
        tokenizer,
        prompt,
        block_size=args.block_size,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    ):
        print(response.text, end="", flush=True)
    print()


def _generate_openai(args) -> None:
    from .benchmark import send_openai

    output = send_openai(
        args.base_url,
        _messages(args),
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        timeout_s=args.timeout_s,
        reasoning=args.reasoning,
    )
    message = output["choices"][0]["message"]
    if reasoning := message.get("reasoning_content"):
        print(reasoning)
    if content := message.get("content"):
        print(content)


def main(argv=None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.backend != "openai" and args.draft is None:
        parser.error("--draft is required for local backends")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.block_size is not None and args.block_size < 1:
        parser.error("--block-size must be positive")
    if args.secondary_model or getattr(args, "check_greedy", False):
        if args.backend != "transformers" or args.temperature != 0:
            parser.error("--secondary-model and --check-greedy require Transformers and --temperature 0")
    if args.secondary_model and args.block_size is not None and args.block_size < 3:
        parser.error("--secondary-model needs --block-size >= 3 to replace a suffix")
    if args.command == "benchmark":
        from .benchmark import run

        run(args)
        return
    {
        "transformers": _generate_transformers,
        "mlx": _generate_mlx,
        "openai": _generate_openai,
    }[args.backend](args)


if __name__ == "__main__":
    main()
