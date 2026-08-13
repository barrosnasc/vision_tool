# Ferramenta: vision-tool (validação visual de interface)

Consulta um modelo de visão (Gemma 3 4B) sobre um screenshot. Uma chamada =
uma sessão; o modelo é descarregado da memória ao final do processo.
GPU e modelo padrão já configurados — o agente não especifica nada disso.

## Comando

```
uvx vision-tool [--validate] [--timeout 300] "<imagem.png>" "<condição>"
```

## Parâmetros

- `<imagem.png>` (obrigatório): caminho local da imagem. Para comparar duas
  telas, use `"antes.png,depois.png"`.
- `<condição>` (obrigatório): pergunta ou descrição sobre a imagem.

## Modos

1. **Validação (recomendado)** — use `--validate` e frases na forma POSITIVA.
   Resposta é `Sim` ou `Não`:

   ```
   uvx vision-tool --validate "tela.png" "o botão Salvar está visível e o título é Configurações"
   ```

   Regras:
   - Uma condição falsa ⇒ resposta `Não`
   - Evite negações ("não está", "oculto") — confundem o modelo

2. **Pergunta aberta** — sem `--validate`. Resposta em texto livre
   (descrição da tela, extrair textos, listar erros visíveis):

   ```
   uvx vision-tool "tela.png" "Liste os textos visíveis na barra superior"
   ```

## Comportamento

- Primeira execução pode demorar (download do modelo); depois, segundos.
- O processo encerra e libera a memória ao final (use `--timeout 300` como rede
  de segurança em execuções longas).
- Códigos de saída: `0` sucesso, `127` binário não encontrado, `124` timeout.

## Exemplos de uso (fluxo de alteração de interface)

```
# validar mudança específica
uvx vision-tool --validate "apos.png" "o menu tem os itens File, Edit, Selection, View e Run"

# conferir regressão antes/depois
uvx vision-tool --validate "antes.png,depois.png" "as duas telas mostram o mesmo layout"

# extrair estado da tela para decidir o próximo passo
uvx vision-tool "tela.png" "Há algum erro ou aviso visível? Liste-os"
```
