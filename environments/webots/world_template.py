"""Resolve resources and render the configured Webots robot and controller."""
from pathlib import Path
import re
from .network_inputs import uses_body_gyro_observation


def absolute_resources(text, template):
    def replace(match):
        value = match[1]
        if value.startswith(("http:", "https:", "webots:", "file:")):
            return match[0]
        if Path(value).suffix.lower() in (".proto", ".png", ".jpg", ".jpeg", ".obj", ".stl"):
            return '"' + (template.parent / value).resolve().as_posix() + '"'
        return match[0]
    return re.sub(r'"([^"\n]+)"', replace, text)

def render_template(template, config, controller="configured_run"):
    text = absolute_resources(template.read_text(encoding="utf-8"), template)
    text, count = re.subn(r"(\bbasicTimeStep\s+)\S+", lambda m: m[1] + str(config["webots"]["basic_time_step_ms"]), text)
    if count != 1:
        raise ValueError("World template must contain exactly one basicTimeStep")
    match = re.search(r"DEF NAO Nao\s*\{[^{}]*\}", text)
    if not match:
        raise ValueError("Expected a flat DEF NAO Nao block in world template")
    block = match[0]
    for field, values in (("translation", config["webots"]["initial_translation"]), ("rotation", config["webots"]["initial_rotation"])):
        block, count = re.subn(rf"(?m)^(\s*{field}\s+)[^\r\n]+", lambda m: m[1] + " ".join(map(str, values)), block)
        if count != 1:
            raise ValueError(f"Missing/duplicate NAO {field}")
    block, count = re.subn(r'\bcontroller\s+"[^"]*"', 'controller "' + controller + '"', block)
    if count != 1:
        raise ValueError("Missing/duplicate NAO Controller")
    if uses_body_gyro_observation(config):
        block = block[:-1].rstrip() + '\n  enableGyroZ TRUE\n}'
    return text[:match.start()] + block + text[match.end():]
