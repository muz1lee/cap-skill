from pathlib import Path

import yaml

from casco.bench.libero_configs import materialize_privileged_minimal_config


def test_materialize_privileged_minimal_config_updates_task_and_output(tmp_path: Path):
    template = tmp_path / "template.yaml"
    template.write_text(
        yaml.safe_dump(
            {
                "env": {
                    "cfg": {
                        "low_level": {"suite_name": "libero_spatial", "task_id": 0},
                        "privileged": True,
                    }
                },
                "api_servers": [{"port": 8116}],
                "record_video": False,
                "output_dir": "./outputs/old",
                "trials": 1,
                "num_workers": 1,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "generated.yaml"

    materialize_privileged_minimal_config(
        template_path=template,
        output_path=output,
        suite_name="libero_object",
        task_id=3,
        output_dir="/mnt/nas/wenqian/cap-skill-data/outputs/run_003",
    )

    generated = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert generated["env"]["cfg"]["low_level"] == {
        "suite_name": "libero_object",
        "task_id": 3,
    }
    assert generated["output_dir"] == "/mnt/nas/wenqian/cap-skill-data/outputs/run_003"
    assert generated["api_servers"] == [{"port": 8116}]
    assert generated["trials"] == 1
    assert generated["num_workers"] == 1
