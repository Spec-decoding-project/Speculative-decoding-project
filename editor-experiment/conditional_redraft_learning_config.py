"""Frozen scientific settings, importable without Torch or remote SDK calls."""
CONFIG = dict(seed=20260918, cut=1, split=3, block_size=16, mode="midpoint",
              objectives=["ce", "prefix", "marginal"], epochs=3, batch_anchors=32,
              learning_rate=1e-4, betas=[0.9, 0.95], weight_decay=0.01,
              adam_epsilon=1e-8, clip_norm=1.0, smoothing=0.5, decay=7.0,
              max_new_tokens=512, max_prompt_tokens=256, max_anchors_per_prompt=32,
              train_per_domain=192, development_per_domain=32,
              domains=["gsm8k", "ultrachat"], warmup_fraction=0.05,
              precision="FP32 trainable parameters; BF16 autocast; BF16 development weights",
              checkpoint_primary="fixed_epoch_3", native_greedy_labels=True,
              generation_speedup_claim=False, seed_replication=False)
