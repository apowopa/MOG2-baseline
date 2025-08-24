
# MOG2-baseline project

## baseline_counter.py

args:

```python
  -h, --help            show this help message and exit
  --source SOURCE       Ruta a video.mp4 o índice de cámara (e.g., 0)
  --width WIDTH
  --height HEIGHT
  --fps FPS
  --no-mjpg             No intentar MJPG
  --no-v4l2             No usar CAP_V4L2
  --history HISTORY
  --varThreshold VARTHRESHOLD
  --detectShadows       Activar sombras de MOG2
  --open-k OPEN_K       Kernel odd para opening
  --close-k CLOSE_K     Kernel odd para closing
  --open-it OPEN_IT
  --close-it CLOSE_IT
  --min-area MIN_AREA
  --line-y LINE_Y       Línea horizontal en Y; si no, defínela con 2
                        clics (tecla 'l')
  --assoc-radius ASSOC_RADIUS
                        R en píxeles (30–60 típico)
  --ttl-frames TTL_FRAMES
                        TTL (6–10 típico)
  --age-min AGE_MIN     Edad mínima para contar (histéresis)
  --print-times         Imprime tiempos por etapa en ms
 ```

## check camera script usage

```python
  python check_camera.py                # intenta /dev/video0
  python check_camera.py --source 0     # también válido (índice)
  python check_camera.py --source /dev/video1
  python check_camera.py --source video.mp4

```
