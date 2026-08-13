# Ferramenta: vision-tool (validação visual de interface)

Consulta um modelo de visão (Gemma 3 4B) sobre um screenshot. Uma chamada =
uma sessão; o modelo é descarregado da memória ao final do processo.
GPU e modelo padrão já configurados — o agente não especifica nada disso.

## Comando

```
uvx vision-tool [--check | --check-code] [--timeout 300] "<imagem.png>" "<condição>"
```

## Parâmetros

- `<imagem.png>` (obrigatório): caminho local da imagem. Para comparar duas
  telas, use `"antes.png,depois.png"`.
- `<condição>` (obrigatório): pergunta ou descrição sobre a imagem.

## Modos

1. **Checagem silenciosa (recomendado)** — `--check-code`: sem saída de
   texto; o veredito vai no código de saída (0 = Sim, 1 = Não). Use frases
   na forma POSITIVA:

   ```
   uvx vision-tool --check-code "tela.png" "o botão Salvar está visível e o título é Configurações"
   # → exit 0 (Sim)
   ```

   Regras:
   - Uma condição falsa ⇒ `Não` (exit 1)
   - Evite negações ("não está", "oculto") — confundem o modelo
   - A resposta é restringida por gramática: o modelo só pode responder
     `Sim` ou `Não`; `-v` mostra o texto e os logs para depuração

2. **Checagem com texto** — `--check`: imprime `Sim` ou `Não` (exit code
   normal do processo):

   ```
   uvx vision-tool --check "tela.png" "o botão Salvar está visível"
   # → Sim
   ```

3. **JSON estruturado** — `--check --json`: formato sempre válido
   (o campo `ok` tem viés de aprovação no modelo padrão de 4B):

   ```
   uvx vision-tool --check --json "tela.png" "o botão está visível"
   # → {"ok": true, "divergencias": []}
   ```

4. **Pergunta aberta** — sem flags de checagem. Resposta em texto livre
   (descrição da tela, extrair textos, listar erros visíveis):

   ```
   uvx vision-tool "tela.png" "Liste os textos visíveis na barra superior"
   ```

## Comportamento

- Primeira execução pode demorar (download do modelo); depois, segundos.
- O processo encerra e libera a memória ao final (use `--timeout 300` como rede
  de segurança em execuções longas).
- Códigos de saída: com `--check-code`, `0`=Sim e `1`=Não (convenção
  test/grep/diff); demais: `127` binário não encontrado, `124` timeout.

## Exemplos de uso (fluxo de alteração de interface)

```
# validar mudança específica (silencioso, veredito no exit code)
uvx vision-tool --check-code "apos.png" "o menu tem os itens File, Edit, Selection, View e Run"

# conferir regressão antes/depois
uvx vision-tool --check-code "antes.png,depois.png" "as duas telas mostram o mesmo layout"

# extrair estado da tela para decidir o próximo passo
uvx vision-tool "tela.png" "Há algum erro ou aviso visível? Liste-os"
```
