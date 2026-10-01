#!/usr/bin/env python3
"""Exercise native binding against the published descriptor inventory, without weights."""
import json
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "crates/ds4-core/tests/fixtures/iquest-main.json"
PIN_KEYS = (
    "general.source.huggingface.repository",
    "general.source.huggingface.revision",
    "iquest_q1.tensor_layout",
    "iquest_q1.quantization.minimum",
)


def main():
    fixture = json.loads(FIXTURE.read_text())
    tensors = fixture["tensors"]
    descriptors = []
    for tensor in tensors:
        name = tensor["name"]
        dims = ",".join(map(str, tensor["dims"]))
        descriptors.append(
            f'{{.name={{(const uint8_t *){json.dumps(name)},{len(name)}}},'
            f'.type={tensor["type"]},.ndim={len(tensor["dims"])},.dim={{{dims}}}' + "}"
        )
    pins = [
        f'{{{json.dumps(key)},{json.dumps(fixture["metadata"][key]["value"])}}}'
        for key in PIN_KEYS
    ]
    source = r'''
#define DS4_NO_GPU 1
#include "ds4.c"
static ds4_tensor tensors[] = { DESCRIPTORS };
static const char *pins[][2] = { PINS };
int main(int argc, char **argv) {
    uint8_t map[1024] = {0}; ds4_kv metadata[4] = {0}; uint64_t cursor = 0;
    for (unsigned i = 0; i < 4; i++) {
        const uint64_t len = strlen(pins[i][1]);
        metadata[i].key = (ds4_str){(const uint8_t *)pins[i][0],strlen(pins[i][0])};
        metadata[i].type = GGUF_VALUE_STRING; metadata[i].value_pos = cursor;
        memcpy(map + cursor, &len, sizeof(len)); cursor += sizeof(len);
        memcpy(map + cursor, pins[i][1], len); cursor += len;
    }
    ds4_model model = {.map=map,.size=sizeof(map),.n_kv=4,.kv=metadata,
        .tensors=tensors,.n_tensors=sizeof(tensors)/sizeof(tensors[0])};
    if (argc == 3) {
        const unsigned index = (unsigned)strtoul(argv[2],NULL,10);
        if (!strcmp(argv[1],"type")) { tensors[index].type = tensors[index].type == DS4_TENSOR_F32 ? DS4_TENSOR_F16 : DS4_TENSOR_F32; }
        if (!strcmp(argv[1],"promotion")) { tensors[index].type = tensors[index].type == DS4_TENSOR_IQ2_XS ? DS4_TENSOR_IQ2_XXS : DS4_TENSOR_IQ2_XS; }
        if (!strcmp(argv[1],"shape")) { tensors[index].dim[0]++; }
        if (!strcmp(argv[1],"pin")) { map[metadata[index].value_pos + sizeof(uint64_t)] ^= 1; }
        if (!strcmp(argv[1],"count")) { model.n_tensors--; }
    }
    ds4_weights weights = {0}; iquest_bind(&weights,&model); return 0;
}
'''.replace("DESCRIPTORS", ",\n".join(descriptors)).replace("PINS", ",".join(pins))
    with tempfile.TemporaryDirectory(prefix="iquest-bind-") as work:
        work = Path(work)
        code, binary = work / "check.c", work / "check"
        code.write_text(source)
        subprocess.run([
            "cc", "-D_GNU_SOURCE", "-std=c11", "-O0", "-w",
            "-ffunction-sections", "-fdata-sections", "-I", str(ROOT),
            str(code), "-Wl,--gc-sections", "-lm", "-pthread", "-o", str(binary),
        ], check=True)
        subprocess.run([str(binary)], check=True)
        # Cover every distinct role plus all promoted matrices. In particular,
        # a valid quant format at the wrong tensor is still an invalid artifact.
        names = {t["name"]: i for i, t in enumerate(tensors)}
        roles = [i for i, t in enumerate(tensors) if t["name"].startswith(("blk.0.", "blk.1.", "mtp.0."))]
        roles += [names[name] for name in ("token_embd.weight", "output.weight", "output_norm.weight")]
        cases = [("type", i) for i in roles] + [("shape", i) for i in roles]
        cases += [("promotion", i) for i, t in enumerate(tensors) if t["type"] == 17]
        cases += [("promotion", names[name]) for name in (
            "blk.61.ffn_gate_exps.weight", "blk.67.ffn_up_exps.weight",
            "blk.68.ffn_down_exps.weight", "blk.78.ffn_gate_exps.weight",
            "blk.80.ffn_gate_exps.weight", "blk.81.ffn_up_exps.weight",
        )]
        cases += [("pin", i) for i in range(len(PIN_KEYS))] + [("count", 0)]
        for mode, index in cases:
            result = subprocess.run([str(binary), mode, str(index)], capture_output=True, text=True)
            assert result.returncode == 1, f"native IQuest bind accepted invalid {mode} at {index}"
    print(f"IQuest native bind: {len(tensors)} exact descriptors, {len(cases)} rejected mutations")


if __name__ == "__main__":
    main()
