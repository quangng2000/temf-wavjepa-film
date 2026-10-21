"""Execute a bounded, standard-library-only read over strictly pinned SSH."""
import json
import subprocess
from pathlib import Path

REMOTE = Path(__file__).with_name("remote")


def build_script(config):
    sources = [path.read_text() for path in sorted(REMOTE.glob("*.py"))
               if path.name != "__init__.py"]
    return ("import json\n" + "\n".join(sources)
            + f"\nprint(json.dumps(snapshot({config['run_path']!r})))\n")


def read_snapshot(config):
    ssh = config["ssh"]
    arguments = ["/usr/bin/ssh", "-i", ssh["identity_file"], "-p", str(ssh["port"]),
                 "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
                 "-o", "StrictHostKeyChecking=yes",
                 "-o", "UserKnownHostsFile=" + ssh["known_hosts"], "-o", "ConnectTimeout=8",
                 "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
                 ssh["user"] + "@" + ssh["host"], "python3 -"]
    result = subprocess.run(arguments, input=build_script(config), capture_output=True,
                            text=True, timeout=25, check=True)
    return json.loads(result.stdout)
