# vision-tool — modelo de visão (texto+imagem → texto) via uvx

Wrapper Python em volta do **`llama-mtmd-cli`** (llama.cpp) para consultar um
modelo de visão por linha de comando, com uma sessão por execução e liberação
automática de memória.

**Modelo padrão combinado:** Gemma 3 4B com visão (`ggml-org/gemma-3-4b-it-GGUF`),
baixado automaticamente — não precisa especificar modelo na chamada.

## Requisitos

- [uv](https://docs.astral.sh/uv/) (fornece o `uvx`)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) compilado com o binário
  `llama-mtmd-cli`:
  ```bash
  git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
  cmake -B build && cmake --build build --target llama-mtmd-cli
  # binário em ./build/bin/llama-mtmd-cli (detectado automaticamente em ~/llama.cpp)
  ```

## Instalação (uma vez)

```bash
uv tool install --from /home/barros/projetos/llm/vision_tool vision-tool
```

Após isso, `uvx vision-tool` já usa o comando instalado.

## Uso padrão

Envie a imagem e a resposta esperada — o modelo valida e responde:

```bash
uvx vision-tool ui.png "O título deve ser 'Configurações' e o botão Salvar deve estar visível"
```

Comparação antes/depois (validação visual):

```bash
uvx vision-tool antes.png,depois.png "Liste o que mudou entre as duas interfaces"
```

Chat interativo (sem argumentos):

```bash
uvx vision-tool
```

## Modo validação (`--validate`)

Transforma as condições em perguntas de sim/não — o formato que, nos testes,
fez o modelo verificar de verdade em vez de aprovar tudo:

```bash
uvx vision-tool --ngl 99 --validate ui.png "o botão Salvar está visível e o título é Configurações"
# → Sim

uvx vision-tool --ngl 99 --validate ui.png "o arquivo aberto na aba é main.rs"
# → Não
```

Regras de ouro (descobertas empiricamente com a Gemma 3 4B):

1. **Fraseie as condições na forma positiva** — "o botão está visível" funciona;
   "o botão não está visível / está oculto" confunde o modelo
2. Uma condição falsa faz a resposta ser "Não" (e vale para listas de condições)
3. `--json` (`{"ok": ..., "divergencias": [...]}`) existe, mas no modelo de 4B
   tem viés de aprovar tudo — prefira o modo texto; o JSON fica confiável em
   modelos maiores (ex.: Qwen2.5-VL 7B)

## Sobrescrever o modelo (opcional)

GGUF local + projetor:

```bash
uvx vision-tool -m modelos/qwen2.5-vl-7b.gguf \
  --mmproj modelos/mmproj-qwen2.5-vl-7b.gguf \
  ui.png "Descreva a interface"
```

Outro repo do Hugging Face:

```bash
uvx vision-tool --hf ggml-org/gemma-3-12b-it-GGUF ui.png "Descreva a interface"
```

Sem instalar (uma execução só):

```bash
uvx --from /home/barros/projetos/llm/vision_tool vision-tool ui.png "prompt"
```

## Opções

| Opção | Descrição |
|---|---|
| `image` | Imagem PNG/JPG ou lista separada por vírgula (posicional) |
| `prompt` | Descrição/resposta esperada (posicional) |
| `-m, --model` | GGUF local alternativo (env `VISION_MODEL`) |
| `--hf REPO` | Repo GGUF alternativo (padrão: `ggml-org/gemma-3-4b-it-GGUF`) |
| `--mmproj` | Projetor multimodal (env `VISION_MMPROJ`) |
| `--ctx` | Contexto em tokens (padrão: 8192; 0 = padrão do modelo, 128k) |
| `-n, --max-tokens` | Máximo de tokens (padrão: 512) |
| `--ngl` | Camadas na GPU (padrão: 99; use 0 para forçar CPU) |
| `--timeout SEG` | Mata o processo após N segundos (libera memória) |
| `--bin` | Caminho do `llama-mtmd-cli` (env `LLAMA_MTMD_CLI`) |
| `--validate` | Modo validação: responde Sim/Não para as condições |
| `--json` | Com `--validate`: resposta JSON (modelos maiores) |
| `-v, --verbose` | Mostra o comando executado |

## Variáveis de ambiente

```bash
export LLAMA_MTMD_CLI=/caminho/para/llama-mtmd-cli   # binário
export VISION_HF_REPO=ggml-org/gemma-3-12b-it-GGUF   # modelo padrão alternativo
export VISION_MODEL=/caminho/para/modelo.gguf        # troca para modo local
export VISION_MMPROJ=/caminho/para/mmproj.gguf
```

## Contexto e VRAM

O modelo (Gemma 3 4B) suporta 128k tokens, mas a validação visual usa <1,5k
no pior caso (1 imagem = 256 tokens). O contexto padrão é **8192** (6× folga),
reduzindo a KV cache de ~2,5 GB para ~160 MB — VRAM por chamada: ~4,4 GB em
vez de ~6,9 GB. Para usar o contexto completo do modelo: `--ctx 0`.

## Semântica de sessão e memória

- **Uma sessão por vez**: cada invocação é um processo independente; o modelo
  atende a chamada e o processo encerra.
- **Descarregamento automático**: ao final do processo, a RAM/VRAM usada pelo
  modelo é liberada pelo SO — nada fica residente.
- **Garantia extra**: use `--timeout 300` para matar execuções demoradas.
- Se preferir um servidor HTTP residente com descarga por inatividade, use o
  `llama-server` diretamente:
  ```bash
  llama-server -m modelo.gguf --mmproj mmproj.gguf -np 1 --sleep-idle-seconds 300
  ```

## Modelos alternativos

| Modelo | Como obter |
|---|---|
| Gemma 3 4B/12B (visão) | `--hf ggml-org/gemma-3-4b-it-GGUF` (padrão) |
| Qwen2.5-VL 3B/7B | GGUF + mmproj (ex.: repositórios `ggml-org/Qwen2.5-VL-*-GGUF` no HF) |
| LLaVA 1.6 7B | GGUF + mmproj |
| MiniCPM-V 2.6 | GGUF + mmproj |
