# grammars

Gramáticas [GBNF](https://github.com/ggml-org/llama.cpp/tree/master/grammars)
para restringir a saída do `llama-mtmd-cli` no vision-tool.

## Por quê

O modo `--validate` pede uma resposta `Sim`/`Não`, mas o modelo pode devolver
texto extra ("Sim, o botão está visível") ou aprovar tudo (viés de
concordância). A gramática **força a resposta a ser exatamente `Sim` ou
`Não`** no nível dos tokens — o modelo escolhe entre as duas opções, mas não
consegue gerar mais nada.

## Arquivos

| Arquivo | Restringe a | Usado em |
|---|---|---|
| [`validate.gbnf`](validate.gbnf) | `Sim` \| `Não` | `--check` / `--check-code` (automático) |
| [`validate-json.gbnf`](validate-json.gbnf) | objeto JSON `{"ok", "divergencias"}` | `--check --check-json` (automático) |
| [`list.gbnf`](list.gbnf) | lista JSON de strings | `--type list` (automático) |
| [`json.gbnf`](json.gbnf) | JSON completo (objeto, array, string, número) | `--type json` (automático) |

Os arquivos são a fonte canônica; o CLI embute cópias idênticas (strings raw
em `vision_tool/cli.py`) para funcionar quando instalado via `uv tool install`.

## Uso manual (sem o vision-tool)

```bash
llama-mtmd-cli \
  -hf ggml-org/gemma-3-4b-it-GGUF \
  --image tela.png \
  --grammar-file grammars/validate.gbnf \
  -p "É verdade que o botão Salvar está visível?"
# → Sim
```

## Resultados dos testes (Gemma 3 4B)

| Caso | Sem gramática | Com gramática |
|---|---|---|
| Condição verdadeira | "Sim" ✅ | "Sim" ✅ (exato) |
| Condição falsa (`main.rs` inexistente) | "Não" ✅ | "Não" ✅ (exato) |
| Texto extra na resposta | frequente | **impossível** |
| JSON válido (`--type`) | às vezes code fence | **sempre** (formatado) |
| Campo `ok` do JSON em condição falsa | `ok: true` ❌ | `ok: true` ❌ (limite do modelo) |

**Limite conhecido:** a gramática trava o *formato*, não o *raciocínio*. No
JSON, o campo `ok` continua com viés de aprovação no modelo de 4B — para
validação confiável use o modo texto (`--check` sem `--check-json`); o JSON fica
confiável em modelos maiores (ex.: Qwen2.5-VL 7B).

## Referências

- [GBNF guide — llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md)
- [JSON grammars — llama.cpp](https://github.com/ggml-org/llama.cpp/tree/master/grammars#json)
