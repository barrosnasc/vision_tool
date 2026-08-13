# grammars

Gramáticas [GBNF](https://github.com/ggml-org/llama.cpp/tree/master/grammars)
para restringir a saída do `llama-mtmd-cli` no vision-tool.

## Por quê

O modo `--validate` pede uma resposta `Sim`/`Não`, mas o modelo pode devolver
texto extra ("Sim, o botão está visível") ou aprovar tudo (viés de
concordância). Uma gramática **força a resposta a ser exatamente `Sim` ou
`Não`** no nível dos tokens — o modelo escolhe entre as duas opções, mas não
consegue gerar mais nada.

## Exemplo: `validate.gbnf`

```gbnf
root ::= "Sim" | "Não"
```

Uso manual (sem o vision-tool):

```bash
llama-mtmd-cli \
  -hf ggml-org/gemma-3-4b-it-GGUF \
  --image tela.png \
  --grammar-file grammars/validate.gbnf \
  -p "Analise a imagem. É verdade que o botão Salvar está visível?"
# → Sim
```

## Gramáticas previstas

| Arquivo | Restringe a |
|---|---|
| `validate.gbnf` | `Sim` ou `Não` (modo `--validate`) |
| `validate-json.gbnf` | objeto JSON `{"ok": ..., "divergencias": [...]}` (modo `--json`) |

## Próximos passos

1. Criar os arquivos `.gbnf` desta pasta
2. Testar manualmente com `--grammar-file` e medir o ganho de acerto
3. Integrar no vision-tool: flag `--grammar <arquivo>` (e, se o ganho for
   confirmado, gramática automática no `--validate`)
4. Atualizar `AGENT_COMMAND.md` e `README.md` com o novo comportamento

## Referências

- [GBNF guide — llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md)
- [JSON grammars — llama.cpp](https://github.com/ggml-org/llama.cpp/tree/master/grammars#json)
