import subprocess
import sys

import pytest

from dflash.cli import main


@pytest.mark.parametrize("options", [
    ["mlx", "--secondary-model", "secondary"],
    ["openai", "--secondary-model", "secondary"],
    ["transformers", "--secondary-model", "secondary", "--temperature", "0.5"],
    ["transformers", "--secondary-model", "secondary", "--block-size", "2"],
    ["mlx", "--check-greedy"],
    ["transformers", "--check-greedy", "--temperature", "0.5"],
    ["transformers", "--max-new-tokens", "0"],
    ["transformers", "--block-size", "0"],
])
def test_invalid_experiment_options_fail_before_loading(options):
    with pytest.raises(SystemExit) as error:
        main(["generate", *options, "--model", "target", "--draft", "draft", "hello"])
    assert error.value.code == 2


def test_cli_help_does_not_import_model_libraries():
    result = subprocess.run(
        [sys.executable, "-c", "import runpy, sys; sys.argv=['dflash', 'generate', '--help']; "
         "\ntry: runpy.run_module('dflash.cli', run_name='__main__')"
         "\nexcept SystemExit as e: assert e.code == 0"
         "\nassert 'torch' not in sys.modules and 'transformers' not in sys.modules"],
        text=True, capture_output=True, check=True,
    )
    assert "--secondary-model" in result.stdout
    assert "--check-greedy" in result.stdout
