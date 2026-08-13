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
import io
import os
import shutil
import signal
import subprocess
import sys
import tempfile

from PIL import Image, UnidentifiedImageError

# Modelo padrão combinado: Gemma 3 4B com visão, baixado automaticamente.
DEFAULT_HF_REPO = "ggml-org/gemma-3-4b-it-GGUF"
DEFAULT_MAX_TOKENS = 512
DEFAULT_NGL = 99  # GPU (CUDA) como padrão; build CPU ignora com aviso
# Contexto: Gemma 3 4B suporta 128k, mas cada imagem = 256 tokens e a
# validação usa <1.5k no pior caso. 8k dá folga de ~6x e reduz a KV cache
# de ~2,5 GB (128k) para ~160 MB. Use --ctx 0 para voltar ao padrão do modelo.
DEFAULT_CTX = 8192

# Gramáticas GBNF (fonte canônica: grammars/*.gbnf neste repositório).
# Restringem a saída no nível dos tokens: o modelo escolhe entre as opções
# permitidas, mas não consegue gerar texto extra nem aprovar tudo.
GRAMMAR_VALIDATE = 'root ::= "Sim" | "Não"'
# Espelha grammars/validate-json.gbnf byte a byte (string raw, sem escapes).
GRAMMAR_VALIDATE_JSON = r'''root   ::= object
object ::= "{" ws "\"ok\"" ws ":" ws boolean ws "," ws "\"divergencias\"" ws ":" ws array ws "}"
array  ::= "[" ws (string (ws "," ws string)*)? ws "]"
string ::= "\"" char* "\""
char   ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
boolean ::= "true" | "false"
ws     ::= [ \t\n]*'''

# Templates dos modos --check/--check-code.
# Achado empírico: no formato de AFIRMAÇÃO o modelo aprova tudo (viés de
# concordância); no formato de PERGUNTA ele verifica de verdade. Por isso
# cada condição vira uma pergunta de sim/não.
VALIDATE_TEMPLATE = (
    "Analise a imagem com atenção. Para CADA pergunta abaixo, responda "
    "apenas 'sim' ou 'não' (sem repetir a pergunta), uma linha por resposta:\n"
    "É verdade que {prompt}?"
)
VALIDATE_JSON_TEMPLATE = (
    "Analise a imagem com atenção. Responda apenas com JSON à pergunta: "
    'é verdade que {prompt}? Regras: use "ok": true apenas se a condição '
    'for verdadeira na imagem; se for falsa, use "ok": false e liste o '
    'motivo em "divergencias".'
)

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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "image",
        nargs="?",
        help="Imagem PNG/JPG ou lista separada por vírgula; '-' lê do stdin (pipe); "
             "omitia para chat interativo",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Descrição/resposta esperada sobre a imagem; omita para chat interativo",
    )

    parser.add_argument(
        "-m", "--model",
        help="GGUF local alternativo (padrão: env VISION_MODEL ou o modelo padrão)",
    )
    parser.add_argument(
        "--hf",
        dest="hf_repo",
        metavar="REPO",
        help=(
            "Repo GGUF alternativo no Hugging Face (padrão: "
            f"{DEFAULT_HF_REPO})"
        ),
    )
    parser.add_argument(
        "--mmproj",
        help="Projetor multimodal .gguf (apenas com -m; padrão: env VISION_MMPROJ)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Checagem sim/não: imprime 'Sim' ou 'Não' (gramática restringe)",
    )
    parser.add_argument(
        "--check-code",
        action="store_true",
        help="Checagem silenciosa: veredito só no exit code (0=Sim, 1=Não)",
    )
    parser.add_argument(
        "--validate",
        dest="check_code",
        action="store_true",
        help=argparse.SUPPRESS,  # apelido antigo de --check-code
    )
    parser.add_argument(
        "--grammar",
        metavar="ARQUIVO",
        help="Gramática GBNF alternativa (--check já usa validate.gbnf por padrão)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help='Com --check: resposta em JSON {"ok": ..., "divergencias": [...]}',
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=DEFAULT_CTX,
        help="Tamanho do contexto em tokens (0 = padrão do modelo, 128k)",
    )
    parser.add_argument(
        "-n", "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Máximo de tokens a gerar",
    )
    parser.add_argument(
        "--ngl",
        type=int,
        default=DEFAULT_NGL,
        help="Camadas na GPU (padrão: 99 = tudo; use 0 para forçar CPU)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        metavar="SEGUNDOS",
        help="Mata o processo após N segundos (garante liberação de memória)",
    )
    parser.add_argument(
        "--bin",
        dest="binary",
        metavar="CAMINHO",
        help="Caminho do llama-mtmd-cli (padrão: env LLAMA_MTMD_CLI ou PATH)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Mostra o comando executado no stderr",
    )
    return parser


# Limites do decodificador do llama.cpp (stb_image): ~16,7 MP. Normalizamos
# tudo para PNG e reduzimos imagens acima disso antes de enviar ao modelo.
_MAX_PIXELS = 15_000_000


def _sniff_type(data: bytes) -> str:
    """Identifica o tipo real do conteúdo (magic numbers) para diagnóstico."""
    if data.startswith(b"\x89PNG"):
        return "PNG (decodificação falhou)"
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG (decodificação falhou)"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WebP (Pillow sem suporte a webp?)"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "GIF"
    if data[:2] == b"BM":
        return "BMP"
    if data[:4] == b"%PDF":
        return "PDF (não é imagem raster)"
    if data[:2] == b"PK":
        return "ZIP/Office"
    if data[4:12] in (b"ftypavif", b"ftypavis"):
        return "AVIF (não suportado)"
    if data[4:12] in (b"ftypheic", b"ftypheix", b"ftypmif1"):
        return "HEIC/HEIF (não suportado)"
    if data.lstrip().startswith((b"<svg", b"<?xml", b"<html", b"<!DOCTYPE")):
        return "SVG/XML/HTML (vetorial — o modelo só aceita imagem raster)"
    try:
        sample = data[:200].decode("utf-8")
        if all(ch.isprintable() or ch in "\n\r\t" for ch in sample):
            return f"texto (começa com: {sample[:60]!r})"
    except UnicodeDecodeError:
        pass
    return "formato desconhecido"


def _normalize_image(data: bytes) -> str:
    """Decodifica bytes de imagem (qualquer formato do Pillow), converte para
    PNG, reduz se for grande demais e devolve o caminho do arquivo temporário."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(
            "o conteúdo do stdin não é uma imagem decodificável — detectado: "
            f"{_sniff_type(data)}. Dica: tente 'wl-paste --type image/png' "
            "ou copie a imagem novamente."
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


def _read_stdin_image() -> str:
    """Lê a imagem do stdin e normaliza (devolve o caminho do temp)."""
    data = sys.stdin.buffer.read()
    if not data:
        raise ValueError(
            "stdin vazio: envie a imagem via pipe "
            '(ex.: cat tela.png | vision-tool - "...")'
        )
    return _normalize_image(data)


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

    if args.image and not args.prompt:
        parser.error("informe também o prompt/descrição esperada (ou omita a imagem para chat interativo)")

    model = args.model or os.environ.get("VISION_MODEL")
    mmproj = args.mmproj or os.environ.get("VISION_MMPROJ")
    binary = _find_binary(args.binary)

    if model:
        cmd = [binary, "-m", model]
        if mmproj:
            cmd += ["--mmproj", mmproj]
    else:
        repo = args.hf_repo or os.environ.get("VISION_HF_REPO") or DEFAULT_HF_REPO
        cmd = [binary, "-hf", repo]

    if args.check and args.check_code:
        parser.error("--check e --check-code são exclusivos")
    if args.json and not args.check:
        parser.error("--json exige --check")
    if args.json and args.check_code:
        parser.error("--json não combina com --check-code")

    check_mode = args.check or args.check_code

    tmp_image = None
    image = args.image
    if image == "-":
        try:
            image = tmp_image = _read_stdin_image()
        except ValueError as exc:
            parser.error(str(exc))

    prompt = args.prompt
    if prompt and check_mode:
        template = VALIDATE_JSON_TEMPLATE if args.json else VALIDATE_TEMPLATE
        prompt = template.format(prompt=prompt)

    if image:
        cmd += ["--image", image]

    # Gramática: explícita (arquivo) > automática do --check/--json.
    grammar_file = args.grammar
    if not grammar_file and check_mode:
        cmd += ["--grammar", GRAMMAR_VALIDATE_JSON if args.json else GRAMMAR_VALIDATE]
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
        if args.check_code and not args.json:
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
            if verdict == "sim":
                return 0
            if verdict in ("não", "nao"):
                return 1  # falso, como test/grep/diff (>=2 fica para erros)
            return proc.returncode
        if args.check and not args.json:
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
        if args.json:
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
            f"erro: '{binary}' não encontrado. Instale o llama.cpp "
            f"(https://github.com/ggml-org/llama.cpp) ou defina LLAMA_MTMD_CLI.",
            file=sys.stderr,
        )
        return 127
    except subprocess.TimeoutExpired:
        print(
            f"erro: tempo esgotado ({args.timeout}s) — processo finalizado e "
            "memória liberada.",
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
