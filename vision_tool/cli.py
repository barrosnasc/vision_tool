"""CLI `vision-tool`: consulta um modelo de visão (texto+imagem -> texto).

Uso padrão (modelo já configurado por padrão — Gemma 3 4B via Hugging Face):

    uvx vision-tool ui.png "O título deve ser 'Configurações' e o botão Salvar visível."

Cada chamada carrega o modelo, processa imagem + descrição esperada, imprime a
resposta e encerra o processo — uma sessão por execução e memória liberada ao
final. Use --timeout para garantir a finalização.

Sobrescrever o modelo:
    uvx vision-tool -m modelos/qwen2.5-vl-7b.gguf --mmproj modelos/mmproj.gguf \
        ui.png "Descreva a interface."
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile

from PIL import Image, UnidentifiedImageError

from vision_tool.i18n import resolve_lang

# Modelo padrão publicado: Gemma 3 4B com visão.
# Sobrescrita local: vision_tool/local_config.py (não versionado) ou env VISION_HF_REPO.
DEFAULT_HF_REPO = "ggml-org/gemma-3-4b-it-GGUF"
DEFAULT_MAX_TOKENS = 512
DEFAULT_NGL = 99  # GPU (CUDA) como padrão; build CPU ignora com aviso


def _load_local_config():
    """Carrega sobrescritas locais (fora do pacote, não versionadas).

    Procura ~/.config/vision-tool/local_config.py — arquivo do usuário com
    DEFAULT_HF_REPO e/ou DEFAULT_NGL. Nunca vai para o GitHub nem para o
    pacote instalado.
    """
    import importlib.util
    for path in (
        os.path.expanduser("~/.config/vision-tool/local_config.py"),
    ):
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("vision_tool_local", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    return None


_LOCAL_CFG = _load_local_config()
_LOCAL_HF_REPO = getattr(_LOCAL_CFG, "DEFAULT_HF_REPO", None)
_LOCAL_NGL = getattr(_LOCAL_CFG, "DEFAULT_NGL", None)
if _LOCAL_NGL is not None:
    DEFAULT_NGL = _LOCAL_NGL
# Contexto: Gemma 3 4B suporta 128k, mas cada imagem = 256 tokens e a
# validação usa <1.5k no pior caso. 8k dá folga de ~6x e reduz a KV cache
# de ~2,5 GB (128k) para ~160 MB. Use --ctx 0 para voltar ao padrão do modelo.
DEFAULT_CTX = 8192

# Gramáticas GBNF (fonte canônica: grammars/*.gbnf neste repositório).
# Restringem a saída no nível dos tokens: o modelo escolhe entre as opções
# permitidas, mas não consegue gerar texto extra nem aprovar tudo.
# A do --check é gerada por idioma: pt -> "Sim" | "Não", en -> "Yes" | "No".
# Espelha grammars/validate-json.gbnf byte a byte (string raw, sem escapes).
GRAMMAR_VALIDATE_JSON = r'''root   ::= object
object ::= "{" ws "\"ok\"" ws ":" ws boolean ws "," ws "\"divergencias\"" ws ":" ws array ws "}"
array  ::= "[" ws (string (ws "," ws string)*)? ws "]"
string ::= "\"" char* "\""
char   ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
boolean ::= "true" | "false"
ws     ::= [ \t\n]*'''
# Espelha grammars/json.gbnf (JSON completo: objeto, array, string, número...).
GRAMMAR_JSON_FULL = r'''root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws

ws ::= | " " | "\n" [ \t]{0,20}'''
# Espelha grammars/list.gbnf byte a byte (lista JSON de strings).
GRAMMAR_JSON_ARRAY = r'''root   ::= array
array  ::= "[" ws (string (ws "," ws string)*)? ws "]"
string ::= "\"" char* "\""
char   ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
ws     ::= [ \t\n]*'''

# Templates e mensagens em pt/en: vision_tool/i18n.py (selecionados por
# --lang ou pelo locale do sistema). Gramática do --check varia junto:
# pt -> "Sim" | "Não", en -> "Yes" | "No".

# Caminhos comuns do binário além do PATH (último caso).
_COMMON_BINARY_PATHS = (
    "~/llama.cpp/build/bin/llama-mtmd-cli",
    "~/gitapps/llama.cpp/build/bin/llama-mtmd-cli",
    "/usr/local/bin/llama-mtmd-cli",
    "/usr/bin/llama-mtmd-cli",
)


def _find_binary(explicit: str | None) -> str:
    """Localiza o llama-mtmd-cli: --bin > env > PATH > caminhos comuns."""
    if explicit:
        return explicit
    for env_var in ("LLAMA_MTMD_CLI", "LLAMA_CPP_BIN"):
        value = os.environ.get(env_var)
        if value:
            return value
    in_path = shutil.which("llama-mtmd-cli")
    if in_path:
        return in_path
    for path in _COMMON_BINARY_PATHS:
        candidate = os.path.expanduser(path)
        if os.path.isfile(candidate):
            return candidate
    return "llama-mtmd-cli"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vision-tool",
        description=(
            "Consulta um modelo de visão (texto+imagem -> texto) com uma única "
            "execução do llama-mtmd-cli. O modelo é descarregado da memória ao "
            "final do processo."
        ),
        epilog=(
            "exemplos:\n"
            "  vision-tool tela.png \"descreva a interface\"\n"
            "  vision-tool --check-code tela.png \"o botão Salvar está visível\"\n"
            "  vision-tool --json tela.png \"liste os itens do menu\"\n"
            "  cat tela.png | vision-tool - \"o que mudou?\"\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    entrada = parser.add_argument_group("entrada")
    entrada.add_argument(
        "image",
        nargs="?",
        help="Imagem ou lista separada por vírgula; '-' lê do stdin (pipe); "
             "omita para chat interativo",
    )
    entrada.add_argument(
        "prompt",
        nargs="?",
        help="Descrição ou pergunta sobre a imagem; omita para chat interativo",
    )

    modos = parser.add_argument_group("modos de resposta (escolha no máximo um)")
    modos.add_argument(
        "--check",
        action="store_true",
        help="Sim/Não em texto (gramática restringe a resposta)",
    )
    modos.add_argument(
        "--check-code",
        action="store_true",
        help="Silencioso: veredito só no exit code (0=Sim, 1=Não)",
    )
    modos.add_argument(
        "--check-json",
        action="store_true",
        help='Veredito em JSON: {"ok": ..., "divergencias": [...]}',
    )
    modos.add_argument(
        "--type",
        choices=["json", "list"],
        metavar="{json,list}",
        help="Formato da resposta em pergunta aberta: json = JSON completo, "
             "list = lista JSON de strings",
    )
    parser.add_argument(
        "--validate",
        dest="check_code",
        action="store_true",
        help=argparse.SUPPRESS,  # apelido antigo de --check-code
    )

    modelo = parser.add_argument_group("modelo")
    modelo.add_argument(
        "-m", "--model",
        help="GGUF local alternativo (padrão: env VISION_MODEL ou o modelo padrão)",
    )
    modelo.add_argument(
        "--hf",
        dest="hf_repo",
        metavar="REPO",
        help=(
            "Repo GGUF alternativo no Hugging Face (padrão: "
            f"{DEFAULT_HF_REPO})"
        ),
    )
    modelo.add_argument(
        "--mmproj",
        help="Projetor multimodal .gguf (apenas com -m; padrão: env VISION_MMPROJ)",
    )

    infer = parser.add_argument_group("inferência")
    infer.add_argument(
        "--grammar",
        metavar="ARQUIVO",
        help="Gramática GBNF alternativa (os modos já têm gramática própria)",
    )
    infer.add_argument(
        "--ctx",
        type=int,
        default=DEFAULT_CTX,
        help="Tamanho do contexto em tokens (0 = padrão do modelo, 128k)",
    )
    infer.add_argument(
        "-n", "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Máximo de tokens a gerar",
    )
    infer.add_argument(
        "--ngl",
        type=int,
        default=DEFAULT_NGL,
        help="Camadas na GPU (padrão: 99 = tudo; use 0 para forçar CPU)",
    )
    infer.add_argument(
        "--timeout",
        type=float,
        default=None,
        metavar="SEGUNDOS",
        help="Mata o processo após N segundos (garante liberação de memória)",
    )

    ui = parser.add_argument_group("interface")
    ui.add_argument(
        "--lang",
        choices=["auto", "pt", "en"],
        default="auto",
        help="Idioma dos prompts e mensagens (auto = locale do sistema)",
    )

    dep = parser.add_argument_group("depuração")
    dep.add_argument(
        "--bin",
        dest="binary",
        metavar="CAMINHO",
        help="Caminho do llama-mtmd-cli (padrão: env LLAMA_MTMD_CLI ou PATH)",
    )
    dep.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Mostra o comando, o texto gerado e os logs do llama.cpp",
    )
    return parser


# Limites do decodificador do llama.cpp (stb_image): ~16,7 MP. Normalizamos
# tudo para PNG e reduzimos imagens acima disso antes de enviar ao modelo.
_MAX_PIXELS = 15_000_000


def _sniff_type(data: bytes, sniff: dict[str, str]) -> str:
    """Identifica o tipo real do conteúdo (magic numbers) para diagnóstico."""
    if data.startswith(b"\x89PNG"):
        return sniff["png"]
    if data.startswith(b"\xff\xd8\xff"):
        return sniff["jpeg"]
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return sniff["webp"]
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return sniff["gif"]
    if data[:2] == b"BM":
        return sniff["bmp"]
    if data[:4] == b"%PDF":
        return sniff["pdf"]
    if data[:2] == b"PK":
        return sniff["zip"]
    if data[4:12] in (b"ftypavif", b"ftypavis"):
        return sniff["avif"]
    if data[4:12] in (b"ftypheic", b"ftypheix", b"ftypmif1"):
        return sniff["heic"]
    if data.lstrip().startswith((b"<svg", b"<?xml", b"<html", b"<!DOCTYPE")):
        return sniff["svg"]
    try:
        sample = data[:200].decode("utf-8")
        if all(ch.isprintable() or ch in "\n\r\t" for ch in sample):
            return sniff["text"].format(preview=repr(sample[:60]))
    except UnicodeDecodeError:
        pass
    return sniff["unknown"]


def _normalize_image(data: bytes, t: dict[str, str]) -> str:
    """Decodifica bytes de imagem (qualquer formato do Pillow), converte para
    PNG, reduz se for grande demais e devolve o caminho do arquivo temporário."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(
            t["not_image"].format(tipo=_sniff_type(data, t["sniff"]))
        ) from exc

    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")

    width, height = img.size
    pixels = width * height
    if pixels > _MAX_PIXELS:
        factor = (_MAX_PIXELS / pixels) ** 0.5
        img = img.resize((max(1, int(width * factor)), max(1, int(height * factor))),
                         Image.LANCZOS)

    fd, path = tempfile.mkstemp(suffix=".png", prefix="vision-stdin-")
    with os.fdopen(fd, "wb") as f:
        img.save(f, "PNG")
    return path


# Browsers colocam a imagem copiada no clipboard como HTML com <img
# src="data:image/...;base64,...">. Extraímos isso automaticamente.
_DATA_IMAGE_RE = re.compile(rb"data:(image/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=]+)")


def _read_stdin_image(t: dict[str, str]) -> str:
    """Lê a imagem do stdin e normaliza (devolve o caminho do temp).

    Se o clipboard entregar HTML (wl-paste sem --type), procura a imagem
    embutida como data:image e a usa."""
    data = sys.stdin.buffer.read()
    if not data:
        raise ValueError(t["stdin_empty"])
    head = data[:512].lstrip()
    if head.startswith((b"<html", b"<!DOCTYPE", b"<meta", b"<img", b"<?xml")):
        match = _DATA_IMAGE_RE.search(data)
        if match:
            data = base64.b64decode(match.group(2))
        else:
            raise ValueError(t["html_no_image"])
    return _normalize_image(data, t)


def _die_with_parent():
    """Mata o filho (llama-mtmd-cli) se o Python morrer sem limpá-lo.

    Evita órfãos segurando VRAM quando o processo uvx é morto abruptamente
    (SIGKILL, abort de script etc.). Usa PR_SET_PDEATHSIG do Linux.
    """
    import ctypes
    try:
        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    except Exception:
        pass  # não crítico: timeout e encerramento normal já limpam


def _preexec():
    return _die_with_parent if sys.platform.startswith("linux") else None


def _strip_fences(text: str) -> str:
    """Remove code fences (```json ... ```) que o modelo às vezes adiciona."""
    out = text.strip()
    lines = out.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    lang, t = resolve_lang(None if args.lang == "auto" else args.lang)

    if args.image and not args.prompt:
        parser.error(t["image_requires_prompt"])

    model = args.model or os.environ.get("VISION_MODEL")
    mmproj = args.mmproj or os.environ.get("VISION_MMPROJ")
    binary = _find_binary(args.binary)

    if model:
        cmd = [binary, "-m", model]
        if mmproj:
            cmd += ["--mmproj", mmproj]
    else:
        repo = (
            args.hf_repo
            or os.environ.get("VISION_HF_REPO")
            or _LOCAL_HF_REPO
            or DEFAULT_HF_REPO
        )
        cmd = [binary, "-hf", repo]

    modes = {
        "--check": args.check,
        "--check-code": args.check_code,
        "--check-json": args.check_json,
        "--type": args.type,
    }
    active = [name for name, on in modes.items() if on]
    if len(active) > 1:
        parser.error(t["one_mode"].format(modos=", ".join(active)))

    check_mode = args.check or args.check_code or args.check_json
    check_json = args.check_json

    tmp_image = None
    image = args.image
    if image == "-":
        try:
            image = tmp_image = _read_stdin_image(t)
        except ValueError as exc:
            parser.error(str(exc))

    prompt = args.prompt
    if prompt and check_json:
        prompt = t["validate_json"].format(prompt=prompt)
    elif prompt and check_mode:
        prompt = t["validate"].format(prompt=prompt)
    elif prompt and args.type == "list":
        prompt = t["list"].format(prompt=prompt)
    elif prompt and args.type == "json":
        prompt = t["type_json"].format(prompt=prompt)

    if image:
        cmd += ["--image", image]

    # Gramática: explícita (arquivo) > automática dos modos.
    grammar_file = args.grammar
    if not grammar_file and check_json:
        cmd += ["--grammar", GRAMMAR_VALIDATE_JSON]
    elif not grammar_file and args.type == "list":
        cmd += ["--grammar", GRAMMAR_JSON_ARRAY]
    elif not grammar_file and args.type == "json":
        cmd += ["--grammar", GRAMMAR_JSON_FULL]
    elif not grammar_file and check_mode:
        cmd += ["--grammar", f'root ::= "{t["sim"]}" | "{t["nao"]}"']
    elif grammar_file:
        cmd += ["--grammar-file", grammar_file]

    if args.ctx > 0:
        cmd += ["-c", str(args.ctx)]

    # Silencia logs do llama-mtmd-cli por padrão (só erros); -v mostra tudo.
    if not args.verbose:
        cmd += ["-lv", "0"]

    cmd += ["-ngl", str(args.ngl), "-n", str(args.max_tokens)]

    if prompt:
        cmd += ["-p", prompt]

    if args.verbose:
        print("+", " ".join(cmd), file=sys.stderr)

    try:
        if args.check_code:
            # --check-code: veredito (Sim/Não, garantido pela gramática)
            # vira código de saída — 0 = Sim, 1 = Não (convenção test/grep;
            # >= 2 reservado para erros). Sem saída por padrão; -v imprime
            # tudo e falhas inesperadas mostram o diagnóstico.
            proc = subprocess.run(
                cmd, check=False, timeout=args.timeout,
                capture_output=True, text=True, preexec_fn=_preexec(),
            )
            verdict = proc.stdout.strip().strip('"').lower()
            if proc.stdout and (args.verbose or not verdict):
                print(proc.stdout, end="")
            if proc.stderr and (args.verbose or not verdict):
                print(proc.stderr, file=sys.stderr, end="")
            if verdict in ("sim", "yes"):
                return 0
            if verdict in ("não", "nao", "no"):
                return 1  # falso, como test/grep/diff (>=2 fica para erros)
            return proc.returncode
        if args.check:
            # --check: imprime o veredito em texto; exit code normal do processo.
            proc = subprocess.run(
                cmd, check=False, timeout=args.timeout,
                capture_output=True, text=True, preexec_fn=_preexec(),
            )
            if proc.stdout:
                print(proc.stdout, end="")
            if proc.stderr and (args.verbose or proc.returncode != 0):
                print(proc.stderr, file=sys.stderr, end="")
            return proc.returncode
        if args.type or check_json:
            proc = subprocess.run(
                cmd, check=False, timeout=args.timeout,
                capture_output=True, text=True, preexec_fn=_preexec(),
            )
            if proc.stdout:
                out = _strip_fences(proc.stdout)
                print(out)
            if proc.stderr and (args.verbose or proc.returncode != 0):
                print(proc.stderr, file=sys.stderr, end="")
            return proc.returncode
        return subprocess.run(
            cmd, check=False, timeout=args.timeout, preexec_fn=_preexec(),
        ).returncode
    except FileNotFoundError:
        print(
            t["binary_not_found"].format(bin=binary),
            file=sys.stderr,
        )
        return 127
    except subprocess.TimeoutExpired:
        print(
            t["timeout"].format(seg=args.timeout),
            file=sys.stderr,
        )
        return 124
    finally:
        if tmp_image:
            try:
                os.remove(tmp_image)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
