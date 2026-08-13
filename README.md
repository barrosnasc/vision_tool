# vision-tool

Consulta um modelo de visão (texto+imagem → texto) por linha de comando, com
**uma sessão por execução** e **liberação automática de memória** ao final de
cada chamada.

Wrapper Python em volta do [`llama-mtmd-cli`](https://github.com/ggml-org/llama.cpp)
com o modelo **Gemma 3 4B** (visão) configurado por padrão — o chamador não
precisa especificar modelo, GPU nem flags de contexto.

```bash
uvx vision-tool ui.png "O título deve ser 'Configurações' e o botão Salvar deve estar visível"
```

## Para que serve

- **Validação visual de interface**: conferir se uma alteração foi aplicada
  (botão, título, menu, layout) em um screenshot
- **Regressão visual**: comparar `antes.png,depois.png`
- **Extração de estado da tela**: ler textos, erros e avisos para um agente
  decidir o próximo passo
- **Perguntas livres** sobre qualquer imagem

## Como funciona

```
uvx vision-tool <imagem> <prompt>
        │
        ▼
  llama-mtmd-cli  ── carrega Gemma 3 4B (GGUF) na GPU
        │
        ▼
  resposta no stdout  ──  processo encerra, VRAM liberada
```

- Modelo padrão: `ggml-org/gemma-3-4b-it-GGUF` (baixado e cacheado automaticamente)
- GPU por padrão (`--ngl 99`); `--ngl 0` força CPU
- Contexto padrão: 8192 tokens (~4,4 GB de VRAM por chamada; o modelo suporta 128k)
- `PR_SET_PDEATHSIG`: se o processo Python morrer, o kernel mata o modelo junto
  — sem processos órfãos segurando VRAM

## Requisitos

- [uv](https://docs.astral.sh/uv/) (fornece o `uvx`)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) compilado com o binário
  `llama-mtmd-cli`:

  ```bash
  git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native  # ou sem -DGGML_CUDA para CPU
  cmake --build build --target llama-mtmd-cli
  ```

  O binário é procurado em: `--bin` > env `LLAMA_MTMD_CLI` > `PATH` >
  `~/llama.cpp/build/bin` > `/usr/bin`.

## Instalação

Uma vez só — instala o comando `vision-tool`:

```bash
uv tool install --from git+https://github.com/barrosnasc/vision_tool vision-tool
```

Depois disso, `uvx vision-tool` usa o comando instalado. Alternativa sem
instalar (uma execução só):

```bash
uvx --from git+https://github.com/barrosnasc/vision_tool vision-tool ui.png "prompt"
```

## Uso

Validação simples:

```bash
uvx vision-tool --check ui.png "o botão Salvar está visível e o título é Configurações"
# → Sim
```

Comparação antes/depois:

```bash
uvx vision-tool antes.png,depois.png "Liste o que mudou entre as duas interfaces"
```

Pergunta aberta:

```bash
uvx vision-tool ui.png "Há algum erro ou aviso visível? Liste-os"
```

Chat interativo (sem argumentos):

```bash
uvx vision-tool
```

## Modo checagem (`--check` e `--check-code`)

Transforma as condições em perguntas de sim/não — o formato que, nos testes,
fez o modelo verificar de verdade em vez de aprovar tudo. A gramática GBNF
restringe a resposta a exatamente `Sim` ou `Não`.

Com texto (`--check`):

```bash
uvx vision-tool --check ui.png "o arquivo aberto na aba é geometry.odin"
# → Sim

uvx vision-tool --check ui.png "o arquivo aberto na aba é main.rs"
# → Não
```

Silencioso, com veredito no código de saída (`--check-code`):

```bash
uvx vision-tool --check-code ui.png "o botão Salvar está visível"
echo $?   # 0 = Sim, 3 = Não
```

Regras de ouro (descobertas empiricamente com a Gemma 3 4B):

1. **Fraseie as condições na forma positiva** — "o botão está visível"
   funciona; "o botão não está visível / está oculto" confunde o modelo
2. Uma condição falsa faz a resposta ser "Não" (vale para listas de condições)
3. `--json` (`{"ok": ..., "divergencias": [...]}`) existe, mas no modelo de 4B
   tem viés de aprovar tudo — prefira os modos texto/código; o JSON fica
   confiável em modelos maiores (ex.: Qwen2.5-VL 7B)
4. `-v` mostra o texto gerado e os logs do llama.cpp (útil para depurar)

## Opções

| Opção | Descrição |
|---|---|
| `image` | Imagem PNG/JPG ou lista separada por vírgula (posicional) |
| `prompt` | Descrição ou pergunta sobre a imagem (posicional) |
| `--check` | Checagem sim/não: imprime `Sim` ou `Não` |
| `--check-code` | Checagem silenciosa: veredito no exit code (0=Sim, 3=Não) |
| `--json` | Com `--check`: resposta JSON (modelos maiores) |
| `-m, --model` | GGUF local alternativo (env `VISION_MODEL`) |
| `--hf REPO` | Repo GGUF alternativo (padrão: `ggml-org/gemma-3-4b-it-GGUF`) |
| `--mmproj` | Projetor multimodal, apenas com `-m` (env `VISION_MMPROJ`) |
| `--ctx` | Contexto em tokens (padrão: 8192; 0 = padrão do modelo, 128k) |
| `-n, --max-tokens` | Máximo de tokens a gerar (padrão: 512) |
| `--ngl` | Camadas na GPU (padrão: 99; 0 = CPU) |
| `--timeout SEG` | Mata o processo após N segundos (libera memória) |
| `--bin` | Caminho do `llama-mtmd-cli` (env `LLAMA_MTMD_CLI`) |
| `-v, --verbose` | Mostra o comando executado |

## Variáveis de ambiente

```bash
export LLAMA_MTMD_CLI=/caminho/para/llama-mtmd-cli   # binário
export VISION_HF_REPO=ggml-org/gemma-3-12b-it-GGUF   # modelo padrão alternativo
export VISION_MODEL=/caminho/para/modelo.gguf        # troca para modo local
export VISION_MMPROJ=/caminho/para/mmproj.gguf
```

## Contexto e VRAM

O Gemma 3 4B suporta 128k tokens, mas a validação visual usa <1,5k no pior
caso (1 imagem = 256 tokens). O contexto padrão é **8192** (6× de folga),
reduzindo a KV cache de ~2,5 GB para ~160 MB — **~4,4 GB de VRAM por chamada**
em vez de ~6,9 GB. Para usar o contexto completo do modelo: `--ctx 0`.

## Semântica de sessão e memória

- **Uma sessão por vez**: cada invocação é um processo independente; o modelo
  atende a chamada e o processo encerra
- **Descarregamento automático**: a VRAM volta ao baseline ao final do processo
- **Rede de segurança**: `--timeout 300` mata execuções demoradas
- Servidor HTTP residente com descarga por inatividade (alternativa ao modo
  processo):
  ```bash
  llama-server -m modelo.gguf --mmproj mmproj.gguf -np 1 --sleep-idle-seconds 300
  ```

## Modelos alternativos

| Modelo | Como obter |
|---|---|
| Gemma 3 4B/12B (visão) | `--hf ggml-org/gemma-3-4b-it-GGUF` (padrão) |
| Qwen2.5-VL 3B/7B | GGUF + mmproj (repositórios `ggml-org/Qwen2.5-VL-*-GGUF` no HF) |
| LLaVA 1.6 7B | GGUF + mmproj |
| MiniCPM-V 2.6 | GGUF + mmproj |

## Licença

[MIT](LICENSE) © 2026 João Pedro Barros
