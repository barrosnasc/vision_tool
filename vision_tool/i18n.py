"""Textos da interface (prompts do modelo e mensagens) em pt-BR e en."""

from __future__ import annotations

import os

STRINGS: dict[str, dict[str, str]] = {
    "pt": {
        "sim": "Sim",
        "nao": "Não",
        "validate": (
            "Analise a imagem com atenção. Para CADA pergunta abaixo, responda "
            "apenas 'sim' ou 'não' (sem repetir a pergunta), uma linha por "
            "resposta:\nÉ verdade que {prompt}?"
        ),
        "validate_json": (
            "Analise a imagem com atenção. Responda apenas com JSON à "
            "pergunta: é verdade que {prompt}? Regras: use \"ok\": true "
            "apenas se a condição for verdadeira na imagem; se for falsa, "
            "use \"ok\": false e liste o motivo em \"divergencias\"."
        ),
        "list": (
            "Analise a imagem com atenção. Responda apenas com JSON: uma "
            "lista de strings, uma por item pedido. {prompt}"
        ),
        "type_json": (
            "Analise a imagem com atenção. Responda apenas com JSON válido e "
            "CONCISO: no máximo 5 campos por objeto e 3 níveis de "
            "profundidade. Não repita valores nem gere conteúdo além do "
            "necessário. {prompt}"
        ),
        "bbox": (
            "Analise a imagem com atenção. Responda apenas com JSON: uma "
            "lista de objetos, cada um com \"label\" (o que é o elemento) e "
            "\"bbox\" (lista [x1, y1, x2, y2] com coordenadas normalizadas "
            "de 0 a 1000 — x1,y1 canto superior esquerdo, x2,y2 canto "
            "inferior direito). Regras: o bbox deve ser o MENOR retângulo "
            "que contém apenas o elemento pedido; nunca use [0,0,1000,1000] "
            "nem caixas que cubram a imagem inteira; se o elemento não "
            "existir, responda com a lista vazia []. Inclua apenas os "
            "elementos pedidos. {prompt}"
        ),
        "stdin_empty": (
            "stdin vazio: envie a imagem via pipe "
            '(ex.: cat tela.png | vision-tool - "...")'
        ),
        "not_image": (
            "o conteúdo do stdin não é uma imagem decodificável — detectado: "
            "{tipo}. Dica: tente 'wl-paste --type image/png' ou copie a "
            "imagem novamente."
        ),
        "html_no_image": (
            "o clipboard entregou HTML sem imagem embutida. Dica: use "
            "'wl-paste --type image/png' ou copie a imagem novamente."
        ),
        "one_mode": "escolha apenas um modo de resposta por vez ({modos})",
        "image_requires_prompt": (
            "informe também o prompt/descrição esperada (ou omita a imagem "
            "para chat interativo)"
        ),
        "binary_not_found": (
            "erro: '{bin}' não encontrado. Instale o llama.cpp "
            "(https://github.com/ggml-org/llama.cpp) ou defina LLAMA_MTMD_CLI."
        ),
        "timeout": (
            "erro: tempo esgotado ({seg}s) — processo finalizado e memória "
            "liberada."
        ),
        "sniff": {
            "png": "PNG (decodificação falhou)",
            "jpeg": "JPEG (decodificação falhou)",
            "webp": "WebP (Pillow sem suporte a webp?)",
            "gif": "GIF",
            "bmp": "BMP",
            "pdf": "PDF (não é imagem raster)",
            "zip": "ZIP/Office",
            "avif": "AVIF (não suportado)",
            "heic": "HEIC/HEIF (não suportado)",
            "svg": "SVG/XML/HTML (vetorial — o modelo só aceita imagem raster)",
            "text": "texto (começa com: {preview})",
            "unknown": "formato desconhecido",
        },
    },
    "en": {
        "sim": "Yes",
        "nao": "No",
        "validate": (
            "Look at the image carefully. For EACH question below, answer "
            "only 'yes' or 'no' (do not repeat the question), one line per "
            "answer:\nIs it true that {prompt}?"
        ),
        "validate_json": (
            "Look at the image carefully. Answer only with JSON to the "
            'question: is it true that {prompt}? Rules: use "ok": true only '
            'if the condition is true in the image; if false, use "ok": '
            'false and list the reason in "divergencias".'
        ),
        "list": (
            "Look at the image carefully. Answer only with JSON: a list of "
            "strings, one per requested item. {prompt}"
        ),
        "type_json": (
            "Look at the image carefully. Answer only with valid and CONCISE "
            "JSON: at most 5 fields per object and 3 levels of depth. Do not "
            "repeat values or generate more content than needed. {prompt}"
        ),
        "bbox": (
            "Look at the image carefully. Answer only with JSON: a list of "
            "objects, each with \"label\" (what the element is) and \"bbox\" "
            "(a [x1, y1, x2, y2] list with coordinates normalized from 0 to "
            "1000 — x1,y1 top-left corner, x2,y2 bottom-right corner). "
            "Rules: the bbox must be the SMALLEST rectangle containing only "
            "the requested element; never use [0,0,1000,1000] or boxes "
            "covering the whole image; if the element does not exist, "
            "answer with the empty list []. Include only the requested "
            "elements. {prompt}"
        ),
        "stdin_empty": (
            "empty stdin: pipe the image "
            '(e.g.: cat screen.png | vision-tool - "...")'
        ),
        "not_image": (
            "stdin content is not a decodable image — detected: {tipo}. "
            "Hint: try 'wl-paste --type image/png' or copy the image again."
        ),
        "html_no_image": (
            "clipboard delivered HTML with no embedded image. Hint: use "
            "'wl-paste --type image/png' or copy the image again."
        ),
        "one_mode": "choose only one response mode at a time ({modos})",
        "image_requires_prompt": (
            "also provide the prompt/expected description (or omit the image "
            "for interactive chat)"
        ),
        "binary_not_found": (
            "error: '{bin}' not found. Install llama.cpp "
            "(https://github.com/ggml-org/llama.cpp) or set LLAMA_MTMD_CLI."
        ),
        "timeout": (
            "error: timed out ({seg}s) — process killed and memory released."
        ),
        "sniff": {
            "png": "PNG (decode failed)",
            "jpeg": "JPEG (decode failed)",
            "webp": "WebP (Pillow without webp support?)",
            "gif": "GIF",
            "bmp": "BMP",
            "pdf": "PDF (not a raster image)",
            "zip": "ZIP/Office",
            "avif": "AVIF (unsupported)",
            "heic": "HEIC/HEIF (unsupported)",
            "svg": "SVG/XML/HTML (vector — the model only accepts raster images)",
            "text": "text (starts with: {preview})",
            "unknown": "unknown format",
        },
    },
}

LANGS = tuple(STRINGS)


def detect_lang() -> str:
    """Idioma padrão pelo locale do sistema (LANG/LC_ALL/LC_MESSAGES)."""
    for var in ("LC_ALL", "LC_MESSAGES", "LANG"):
        value = os.environ.get(var, "")
        if value.lower().startswith("pt"):
            return "pt"
        if value and not value.lower().startswith(("c", "c.", "posix")):
            return "en"
    return "en"


def get_strings(lang: str | None) -> dict[str, str]:
    """Devolve o dicionário de strings do idioma (auto detecta o sistema)."""
    if lang in STRINGS:
        return STRINGS[lang]
    return STRINGS[detect_lang()]


def resolve_lang(lang: str | None) -> tuple[str, dict[str, str]]:
    """Resolve o idioma ('auto' vira pt/en pelo locale) e devolve as strings."""
    key = lang if lang in STRINGS else detect_lang()
    return key, STRINGS[key]
