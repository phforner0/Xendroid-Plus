# Halo: Reach — otimização de 2026-11-04

POCO F7 (Adreno 825, Turnip 27.0.0-devel V38, HyperOS), pacote de teste
`xendroid.compose.fork.ui`. Cena: primeira missão, parado junto ao Warthog com o
aviso "Hold X to Enter/Exit Vehicles", cerca de 2775 draws por frame, estável
(±0,5 ms entre braços de controle). Para chegar lá, A nos menus; depois, sem
tocar no aparelho.

## Resumo

**Mesmo build, sem os timestamps de GPU, quadros contados contra o relógio:**

| Configuração | fps nos primeiros 60 s | fps nos 38–41 s seguintes |
|---|---:|---:|
| Tudo desligado no config do jogo | 16,90 | 17,31 |
| **Com as quirks do jogo** | **30,03** | **28,60** |

Ou seja, **de 17 para 30 fps**, o limite do próprio jogo nessa cena, sem perda de
imagem medida nas regiões comparadas (céu, vale distante, rocha, HUD) e com um teste
jogando a pé na primeira missão, sem defeito visto. As duas rodadas foram no
carregador com a bateria acima de 85%, uma em seguida da outra (no fim, bateria a 38,8
e 38,5 °C). A captura de tela da medição cai no começo da segunda janela.

Com o aparelho mais quente, a mesma comparação deu 15,5 e 28 fps (build anterior,
bateria de 19% a 40% no carregador e SoC a 95 °C).

Os rostos dos personagens nas cinemáticas não apareciam com nenhuma das configurações
acima. Veja [Rostos](#rostos).

## Por etapa

Acumulado, com `log_gpu_frame_time_breakdown` (456 timestamps por frame, que custam
um pouco; por isso o fps menor que o do resumo):

| Etapa | fps | GPU ms/frame |
|---|---:|---:|
| Base (0.2.2 + scheduler) | 14,9 | 64,4 |
| + barreiras de memexport por faixa (`vulkan_memexport_barriers_by_range`) | 18,1 | 53,0 |
| + saltos de shader como if/else (`spirv_structured_forward_jumps`) | 18,9 | 50,5 |
| + sem arredondamento para zero em software (`alu_round_toward_zero = false`) | 22,2 | 43,1 |
| + VS relaxado, 4x com textura na superfície 1x com shading 2x2, LOD implícito | 26,7 | 35,9 |

Isolados, contra 52,6–53 ms: 4x com textura na superfície 1x **−2,9 ms** (tiles
transferidos de 24,9 mil para 13,1 mil por frame), `spirv_vs_relaxed_math = 1`
**−1,4 ms**.

## Causas

- **Barreiras.** Cada draw com memexport ganhava uma barreira, que encerrava o render
  pass e invalidava os caches, tocasse o draw seguinte ou não naquela memória. O Reach
  tem centenas desses draws por frame: 227 render passes por frame, 103 por faixa. O
  Edge tem a mesma barreira por draw (`TODO(Triang3l)`), logo **o Edge não cobre este
  caso**, e a diferença se justifica pela medição acima.
- **VS `838B50F94967ACAD`.** Atualiza as partículas no próprio buffer, 115 draws de 16
  pontos. Tem 28 labels, 43 saltos e um loop de 14 iterações. Cada salto virava uma
  volta do loop principal mais um `OpSwitch` sobre todos os labels: 11,0 ms para esses
  draws, 8,6 ms com if/else. O Edge traduz os saltos do mesmo jeito.
- **RTZ em software.** Todo resultado de ALU que pode chegar a eA/eM, índice de vfetch,
  a0 ou floor/trunc/frc passa por Dekker/2Sum (o mesmo desenho de Edge `b0f7c0c2c` e
  `043a7cb40`). Esse VS compilava para 39184 instruções; sem isso, esses draws caem de
  8,6 para 4,9 ms. Não há caminho em hardware: `shaderRoundingModeRTZFloat32` é 0 no
  Turnip e no driver da Qualcomm deste aparelho.
- **4x de meia resolução.** Cerca de 11 draws de partículas suaves (lendo a posição e a
  profundidade da cena) faziam a profundidade ir e voltar entre 1x e 4x. Desenhados na
  superfície 1x, com um shading por 2x2 pixels, as transferências caem à metade.

## Rostos

Na cinemática de abertura da primeira missão, Kat e Carter apareciam só com cabelo,
olhos e dentes, com a parede visível através da cabeça, também com todas as opções deste
jogo desligadas. O defeito já existia antes delas.

Os rostos são `kTriangleStrip` indexados com tesselação discreta, sobre vértices que outro
shader exporta para a memória (VS `15F95702DE4643BD`, depois 512 pontos vazios de espera, e
o VS `B66CEB98EF994623` desenha). O domain shader lê em `r1` os índices dos três cantos de
cada patch, mas o pipeline Vulkan usava 1 control point: cada índice virava um patch, e os
outros dois índices ficavam indefinidos.

O Edge corrigiu os mesmos rostos em `9c30b16e1`, onde os strips indexados não tinham
conversão para lista; o fork já tinha essa conversão. O backend Vulkan do Edge ainda tem os
mesmos control points, e o Direct3D 12 dele escolhe pelo tipo de vertex shader do host, como
agora aqui. **Essa correção vale para todos os jogos**, não é uma quirk: testada só no
Reach.

## Rejeitado

- `spirv_ps_relaxed_math = 1`: −1 ms, mas escurece o céu e clareia o vale.
- Clocks máximos forçados: 13,4 fps e 71,3 ms (aquece e cai).
- Pular os draws vazios do VS `B6C9863F710683EC` (318 por frame, a espera do jogo pelas
  exportações): sem ganho de GPU.
- `transfer_cleared_sources_as_clears` e `vulkan_resolve_clear_in_guest_pass`: +0,3 e
  −0,2 ms, dentro do ruído.

## Como comparar imagens deste jogo

A nitidez das capturas alterna entre dois estados **com a mesma configuração** (energia
de borda de cerca de 3,0 e 3,8 na região das árvores, nos dois casos dentro de uma mesma
rodada). Uma captura mais serrilhada que outra, sozinha, não indica efeito de uma opção.

## Não testado

- Jogo longo sem o RTZ, além do começo da primeira missão a pé: veículos, combate com
  muitas partículas, outras missões. Se aparecer geometria explodindo ou z-fighting, é a
  primeira quirk a suspeitar.
- Cenas mais pesadas que esta. Nesta o jogo já chega ao seu limite de 30 fps, então daqui
  em diante um ganho só aparece em GPU ms ou em outra cena.
- A correção dos rostos em outros jogos com tesselação indexada (`kTriangleDomainCPIndexed`
  e `kQuadDomainCPIndexed`), que também passam a usar 3 ou 4 control points.
- O resto da campanha com os rostos: só a cinemática de abertura foi conferida.
- A primeira abertura depois da atualização: os pipelines do jogo são recriados, o que
  numa rodada com configuração nova levou até o quadro 2988. Nesse tempo, draws de
  pipelines ainda não prontos são descartados e objetos podem aparecer errados.
- LOD implícito isolado neste jogo; `render_target_7e3_as_r11g11b10`;
  `occlusion_query=fake` (116 queries por frame).
- Outros aparelhos e o driver da Qualcomm: sem shading rate por pipeline, o caminho 4x
  grosso fica desligado sozinho.
