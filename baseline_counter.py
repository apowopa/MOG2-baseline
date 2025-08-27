import argparse
import time
from collections import deque
import cv2
import numpy as np
import sys
from math import hypot

def try_open_capture(device, width, height, fps, prefer_mjpg=True, use_v4l2=True):
    """Intenta abrir cámara con MJPG; si falla, usa formato por defecto."""
    cap_flags = cv2.CAP_V4L2 if use_v4l2 else 0
    cap = cv2.VideoCapture(device, cap_flags) if isinstance(device, int) or device.startswith('/dev/') else cv2.VideoCapture(device)
    if not cap.isOpened():
        return None, None

    # Set resolución y FPS deseados
    if width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:    cap.set(cv2.CAP_PROP_FPS, fps)

    fourcc_eff = None
    if prefer_mjpg:
        # Intentar MJPG
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
        time.sleep(0.05)  # pequeño respiro
        fourcc_eff = int(cap.get(cv2.CAP_PROP_FOURCC))
        if fourcc_eff == 0:
            # Algunos backends reportan 0: intentar lectura real para confirmar
            ret, _ = cap.read()
            if not ret:
                cap.release()
                return None, None

    # Lectura de propiedades efectivas
    eff_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    eff_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    eff_fps = cap.get(cv2.CAP_PROP_FPS)
    eff_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    return cap, (eff_w, eff_h, eff_fps, eff_fourcc)

def fourcc_to_str(code_int):
    if code_int is None or code_int == 0:
        return "UNKNOWN"
    return ''.join([chr((int(code_int) >> (8 * i)) & 0xFF) for i in range(4)])

# Dibujo y línea de conteo
def put_hud(img, lines, x, y):
    for line in lines:
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        y += 18

def side_of_line(p, a, b):
    # Signo del producto cruzado 2D (orientación)
    return np.sign((b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]))


# Tracker simple por centroides

class Track:
    def __init__(self, tid, cxy):
        self.id = tid
        self.centroid = cxy
        self.age = 0            # frames totales vivos
        self.ttl = 0            # frames desde última vez visto (para expiración)
        self.prev_side = None
        self.last_side = None
        self.counted_in = False
        self.counted_out = False

def associate_tracks(tracks, detections, assoc_radius, ttl_frames):
    """
    Asocia detecciones (centroides) a tracks por vecino más cercano con radio.
    Retorna mapping: det_idx -> track, y actualiza tracks in-place.
    """
    assigned = {}
    used_tracks = set()

    for i, c in enumerate(detections):
        # Encuentra track más cercano no usado
        best_tid, best_d = None, 1e9
        for tid, tr in tracks.items():
            if tid in used_tracks:
                continue
            d = hypot(tr.centroid[0]-c[0], tr.centroid[1]-c[1])
            if d < best_d:
                best_d, best_tid = d, tid
        if best_tid is not None and best_d <= assoc_radius:
            # Actualiza track
            tr = tracks[best_tid]
            tr.centroid = c
            tr.ttl = 0
            tr.age += 1
            assigned[i] = tr
            used_tracks.add(best_tid)

    # Crea nuevos tracks para no asignados
    next_id = (max(tracks.keys())+1) if tracks else 1
    for i, c in enumerate(detections):
        if i in assigned: 
            continue
        tr = Track(next_id, c)
        tr.age = 1
        tr.ttl = 0
        tracks[next_id] = tr
        assigned[i] = tr
        next_id += 1

    # Incrementa TTL de no vistos
    to_delete = []
    for tid, tr in tracks.items():
        if tid not in used_tracks and tr.centroid not in detections:
            tr.ttl += 1
        # marcar expirados
        if tr.ttl > ttl_frames:
            to_delete.append(tid)
    for tid in to_delete:
        tracks.pop(tid, None)

    return assigned

# Funcion MAE 
def reporte_mae(conteo_auto, conteo_manual, tipo: str):
    mae = abs(conteo_auto - conteo_manual)
    pct = (mae / max(1, conteo_manual)) * 100.0
    print(f"==== Validación de Conteo {tipo}====")
    print(f"Conteo automático: {conteo_auto}")
    print(f"Conteo manual:     {conteo_manual}")
    print(f"MAE:               {mae:.2f}")
    print(f"% Error:           {pct:.2f}%")
    return mae, pct

# Main (Pasos 0–4 y visualización)
def main():
    ap = argparse.ArgumentParser(description="Conteo por cruce con MOG2 + asociación simple")
    # Parametros de video de entrada 
    ap.add_argument("--source", type=str, default="/dev/video0", help="Ruta a video.mp4 o índice de cámara (e.g., 0)")
    ap.add_argument("--width",  type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fps",    type=int, default=30)
    ap.add_argument("--no-mjpg", action="store_true", help="No intentar MJPG")
    ap.add_argument("--no-v4l2", action="store_true", help="No usar CAP_V4L2")

    # MOG2 configuraciones
    ap.add_argument("--history", type=int, default=500)
    ap.add_argument("--varThreshold", type=float, default=32.0)
    ap.add_argument("--detectShadows", action="store_true", help="Activar sombras de MOG2")

    # Morfología para los centroides
    ap.add_argument("--open-k", type=int, default=3, help="Kernel odd para opening")
    ap.add_argument("--close-k", type=int, default=5, help="Kernel odd para closing")
    ap.add_argument("--open-it", type=int, default=1)
    ap.add_argument("--close-it", type=int, default=1)

    # Contornos
    ap.add_argument("--min-area", type=int, default=300)

    # Línea
    ap.add_argument("--line-y", type=int, default=None, help="Línea horizontal en Y; si no, defínela con 2 clics (tecla 'l')")
    
    # Configuracion de Asociacion
    ap.add_argument("--assoc-radius", type=float, default=45.0, help="R en píxeles (30–60 típico)")
    ap.add_argument("--ttl-frames", type=int, default=8, help="TTL (6–10 típico)")
    ap.add_argument("--age-min", type=int, default=3, help="Edad mínima para contar (histéresis)")

    # Tiempos
    ap.add_argument("--print-times", action="store_true", help="Imprime tiempos por etapa en ms")

    args = ap.parse_args()

    # Abrir fuente (índice de cámara o ruta de archivo)
    source_arg = args.source
    try:
        # permitir índice numérico
        if source_arg.isdigit():
            source_input = int(source_arg)
        else:
            source_input = source_arg
    except Exception:
        source_input = source_arg

    cap, eff = try_open_capture(
        source_input, args.width, args.height, args.fps,
        prefer_mjpg=(not args.no_mjpg),
        use_v4l2=(not args.no_v4l2)
    )

    # Si falla abrir el capture
    if cap is None or not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la fuente especificada: '{args.source}'")
        print(f"abriendo camera en /dev/video0 por defecto")
        cap, eff = try_open_capture("/dev/video0", args.width, args.height, args.fps,
        prefer_mjpg=(not args.no_mjpg),
        use_v4l2=(not args.no_v4l2)
        )

    if cap is None or not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la fuente especificada: '{args.source}'")
        sys.exit(1)

    eff_w, eff_h, eff_fps, eff_fourcc = eff
    print("=== Configuración efectiva de captura ===")
    print(f"Resolución: {eff_w}x{eff_h}")
    print(f"FPS (reportado): {eff_fps:.2f}")
    print(f"FOURCC: {fourcc_to_str(eff_fourcc)}")
    print(f"Fuente: '{args.source}'")

    # Invocacion del mog
    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=args.history,
        varThreshold=args.varThreshold,
        detectShadows=args.detectShadows
    )

    # Kernels morfológicos
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1,args.open_k)|1, max(1,args.open_k)|1))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1,args.close_k)|1, max(1,args.close_k)|1))

    # Línea de conteo
    line_defined = False
    line_pts = []  # [(x1,y1), (x2,y2)]
    horizontal_y = args.line_y
    baseline_mode = 'horizontal' if horizontal_y is not None else 'manual'

    # Conteos
    count_in, count_out, total = 0, 0, 0
    count_in_manual, count_out_manual = 0, 0

    # Tracking
    tracks = {}
    show_mask = False
    show_blobs = True
    drawing_line_mode = False

    # FPS
    fps_deque = deque(maxlen=30)
    t_prev = time.perf_counter()

    win = "Conteo"
    cv2.namedWindow(win)

    def on_mouse(event, x, y, flags, param):
        nonlocal line_pts, line_defined, baseline_mode, horizontal_y
        if drawing_line_mode and event == cv2.EVENT_LBUTTONDOWN:
            line_pts.append((x,y))
            if len(line_pts) == 2:
                line_defined = True
                baseline_mode = 'manual'

    cv2.setMouseCallback(win, on_mouse)

    while True:
        t0_all = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        t1_pre = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Prepro ligera (opcional): blur para reducir ruido
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        t2_mog_start = time.perf_counter()
        fg = mog2.apply(blur)
        # Binarizar si detectShadows activo (255=fg, 127=sombra)
        _, fg_bin = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        # Opening + Closing
        fg_clean = cv2.morphologyEx(fg_bin, cv2.MORPH_OPEN,  k_open,  iterations=args.open_it)
        fg_clean = cv2.morphologyEx(fg_clean, cv2.MORPH_CLOSE, k_close, iterations=args.close_it)
        t3_cnt_start = time.perf_counter()

        # Contornos
        contours, _ = cv2.findContours(fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < args.min_area:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0: 
                continue
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            detections.append((cx, cy))

        # Línea: horizontal o por 2 puntos
        a, b = None, None
        if baseline_mode == 'horizontal' and horizontal_y is not None:
            a = (0, horizontal_y)
            b = (frame.shape[1]-1, horizontal_y)
            cv2.line(frame, a, b, (0,255,255), 2)
        elif baseline_mode == 'manual' and line_defined and len(line_pts) >= 2:
            a, b = line_pts[0], line_pts[1]
            cv2.line(frame, a, b, (0,255,255), 2)

        # Asociación + histéresis (Paso-2)
        assigned = associate_tracks(tracks, detections, args.assoc_radius, args.ttl_frames)

        # Actualizar sides y contar cruces
        if a is not None and b is not None:
            for tr in tracks.values():
                c = tr.centroid
                s = side_of_line(c, a, b)
                tr.prev_side = tr.last_side
                tr.last_side = s if s != 0 else tr.last_side  # evita 0 si cae justo en la línea

                # Histeresis: contar si la pista tiene edad suficiente
                if tr.age >= args.age_min and tr.prev_side is not None and tr.last_side is not None:
                    if tr.prev_side < 0 and tr.last_side > 0 and not tr.counted_in:
                        count_in += 1
                        total += 1
                        tr.counted_in = True
                    elif tr.prev_side > 0 and tr.last_side < 0 and not tr.counted_out:
                        count_out += 1
                        total += 1
                        tr.counted_out = True

        # Dibujo
        if show_blobs:
            for c in contours:
                if cv2.contourArea(c) >= args.min_area:
                    x,y,w,h = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            for tr in tracks.values():
                cx, cy = int(tr.centroid[0]), int(tr.centroid[1])
                cv2.circle(frame, (cx,cy), 4, (0,0,255), -1)
                cv2.putText(frame, f"ID{tr.id}", (cx+6, cy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(frame, f"ID{tr.id}", (cx+6, cy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

        # HUD conteos
        hud = [
            f"Auto IN: {count_in}  OUT: {count_out}",
            f"Manual IN: {count_in_manual}  OUT: {count_out_manual}",
        ]

        if count_in_manual > 0:
            mae_in = abs(count_in - count_in_manual)
            hud.append(f"MAE IN: {mae_in}")
        if count_out_manual > 0:
            mae_out = abs(count_out - count_out_manual)
            hud.append(f"MAE OUT: {mae_out}")

        # FPS
        t_now = time.perf_counter()
        fps = 1.0 / max(1e-6, (t_now - t_prev))
        fps_deque.append(fps)
        fps_smooth = sum(fps_deque) / len(fps_deque)
        t_prev = t_now

        hud.append(f"FPS: {fps_smooth:.1f}  FOURCC: {fourcc_to_str(eff_fourcc)}")
        put_hud(frame, hud, 10, 20)

        # Mostrar máscara opcional
        if show_mask:
            mask_bgr = cv2.cvtColor(fg_clean, cv2.COLOR_GRAY2BGR)
            disp = np.hstack((frame, mask_bgr))
        else:
            disp = frame

        # Paso-3: tiempos
        if args.print_times:
            t_end_all = time.perf_counter()
            pre_ms  = (t2_mog_start - t1_pre) * 1000.0
            mog_ms  = (t3_cnt_start - t2_mog_start) * 1000.0
            cnt_ms  = (t_end_all - t3_cnt_start) * 1000.0
            tot_ms  = (t_end_all - t0_all) * 1000.0
            print(f"pre:{pre_ms:6.2f}  mog+morph:{mog_ms:7.2f}  cont:{cnt_ms:6.2f}  total:{tot_ms:7.2f}  fps:{fps_smooth:5.1f}")

        cv2.imshow(win, disp)

        # Teclas
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC o q
            break
        elif key == ord('m'):
            show_mask = not show_mask
        elif key == ord('b'):
            show_blobs = not show_blobs
        # Activar modo dibujo de línea    
        elif key == ord('l'):
            line_pts = []
            line_defined = False
            baseline_mode = 'manual'
            drawing_line_mode = True
            print("[INFO] Haz 2 clics para definir la línea.")
        # Volver a modo horizontal si se definió por parámetro
        elif key == ord('h'):
            if args.line_y is not None:
                baseline_mode = 'horizontal'
                line_defined = False
                print(f"[INFO] Línea horizontal activada en Y={args.line_y}")
        elif key == ord('i'):
            count_in_manual += 1
            print(f"Incrementando conteo manual IN: {count_in_manual}")
        elif key == ord('o'):
            count_out_manual += 1
            print(f"Incrementando conteo manual OUT: {count_out_manual}")
            

        # Finalizar dibujo al tener 2 puntos
        if drawing_line_mode and line_defined:
            drawing_line_mode = False
            print(f"[INFO] Línea definida: {line_pts[0]} -> {line_pts[1]}")

    cap.release()
    cv2.destroyAllWindows()

    # Validación MAE al terminar el video 
    if not args.source.isdigit() and not args.source.startswith('/dev/video'):
        print(f"Conteo AUTOMÁTICO - IN: {count_in}, OUT: {count_out}")
    
    try:
        if count_in_manual == 0 and count_out_manual == 0:
            manual_in = int(input("Conteo manual IN: "))
            manual_out = int(input("Conteo manual OUT: "))
        else:
            manual_in = count_in_manual
            manual_out = count_out_manual
        
        # Calcular y mostrar MAE
        print("\n--- REPORTE MAE ---")
        reporte_mae(count_in, manual_in, "IN")  # MAE para entradas 
        reporte_mae(count_out, manual_out, "OUT")  # MAE para salidas
    except ValueError:
        print("Error: Ingrese valores numéricos válidos para el conteo manual.")

if __name__ == "__main__":
    main()
