#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Paso-0 — Chequeo de cámara USB + FPS (OpenCV, V4L2)
# Requisitos: Python 3, OpenCV (cv2)
#
# Uso:
#   python usb_cam_check.py                # intenta /dev/video0
#   python usb_cam_check.py --source 0     # también válido (índice)
#   python usb_cam_check.py --source /dev/video1
#   python usb_cam_check.py --source video.mp4

import cv2
import time
import argparse
import os
import sys

PREF_WIDTH = 640
PREF_HEIGHT = 360
PREF_FPS = 30
PREF_FOURCC = 'MJPG'  # intentar MJPEG para mejorar FPS

def is_video_device(src: str) -> bool:
    # Considerar /dev/videoX (o un entero)
    if isinstance(src, int):
        return True
    if isinstance(src, str) and src.startswith("/dev/video"):
        return True
    # índices pasados como string, ej. "0"
    try:
        int(src)
        return True
    except Exception:
        return False

def parse_source(src_arg: str):
    # Devuelve (source, is_device)
    # Si es un entero como texto, convertir a int para VideoCapture
    if src_arg is None:
        return "/dev/video0", True
    if src_arg.startswith("/dev/video"):
        return src_arg, True
    try:
        idx = int(src_arg)
        return idx, True
    except Exception:
        return src_arg, False

def fourcc_to_str(v: float) -> str:
    # cv2.CAP_PROP_FOURCC devuelve un float con el entero
    try:
        v = int(v)
        return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])
    except Exception:
        return "----"

def backend_name(cap) -> str:
    try:
        return cap.getBackendName()
    except Exception:
        return "unknown"

def open_capture(source, prefer_v4l2=True):
    # Intenta abrir con V4L2 primero (si aplica)
    cap = None
    if prefer_v4l2 and (isinstance(source, int) or (isinstance(source, str) and source.startswith("/dev/video"))):
        cap = cv2.VideoCapture(source, apiPreference=cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            cap = None
    if cap is None:
        cap = cv2.VideoCapture(source)
    return cap

def try_set_camera_props(cap):
    # Establecer FOURCC primero (muchas webcams lo requieren antes de tamaño/FPS)
    ok_fourcc = cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*PREF_FOURCC))

    # Resolución
    ok_w = cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREF_WIDTH)
    ok_h = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREF_HEIGHT)

    # FPS deseado
    ok_fps = cap.set(cv2.CAP_PROP_FPS, PREF_FPS)

    # Leer valores efectivos
    eff_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    eff_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    eff_fps = cap.get(cv2.CAP_PROP_FPS)
    eff_fourcc = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))

    print("[INFO] Solicitudes:")
    print(f"       FOURCC={PREF_FOURCC} (set ok={ok_fourcc})  Resolución={PREF_WIDTH}x{PREF_HEIGHT} (ok_w={ok_w}, ok_h={ok_h})  FPS={PREF_FPS} (ok={ok_fps})")
    print("[INFO] Efectivos reportados por cámara/driver:")
    print(f"       FOURCC={eff_fourcc}  Resolución={eff_w}x{eff_h}  FPS={eff_fps:.2f}")

    return eff_w, eff_h, eff_fps, eff_fourcc

def print_source_info(src, cap, is_device):
    print("====================================================")
    print(f"[INFO] Backend: {backend_name(cap)}")
    if is_device:
        dev = src if isinstance(src, str) else f"index {src}"
        print(f"[INFO] Dispositivo: {dev}")
    else:
        print(f"[INFO] Archivo: {src}")
    print("====================================================")

def main():
    ap = argparse.ArgumentParser(description="Chequeo rápido de cámara USB + FPS (OpenCV, V4L2)")
    ap.add_argument("--source", type=str, default="/dev/video0",
                    help="Ruta de dispositivo (/dev/videoX), índice (0), o archivo de video (video.mp4).")
    args = ap.parse_args()

    source, is_device = parse_source(args.source)

    cap = open_capture(source, prefer_v4l2=True)
    if not cap or not cap.isOpened():
        # Si intentábamos la cámara por defecto y falló, permitir fallback rápido a ejemplo de archivo si el usuario lo dio
        if (isinstance(source, str) and source.startswith("/dev/video")) or (isinstance(source, int) and source == 0):
            print(f"[WARN] No se pudo abrir '{source}'. Si quieres probar con un archivo: --source video.mp4")
        else:
            print(f"[ERROR] No se pudo abrir la fuente: {source}")
        sys.exit(1)

    print_source_info(source, cap, is_device)

    # Si es cámara, intentar fijar propiedades; si es archivo, solo reportar lo que hay
    if is_device:
        eff_w, eff_h, eff_fps, eff_fourcc = try_set_camera_props(cap)
    else:
        eff_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        eff_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        eff_fps = cap.get(cv2.CAP_PROP_FPS)
        eff_fourcc = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))
        print("[INFO] Fuente de archivo. Valores reportados:")
        print(f"       FOURCC={eff_fourcc}  Resolución={eff_w}x{eff_h}  FPS (reportado)={eff_fps:.2f}")

    # Warm-up: leer un frame para verificar
    ok, frame = cap.read()
    if not ok:
        print("[ERROR] No se pudo leer frame inicial.")
        cap.release()
        sys.exit(1)

    win_name = "USB Cam Check — ESC para salir"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    # Medición de FPS (promedio exponencial)
    ema_fps = 0.0
    alpha = 0.1  # suavizado
    t_prev = time.time()
    frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Fin de stream o fallo de captura.")
            break

        t_now = time.time()
        dt = t_now - t_prev
        t_prev = t_now
        if dt > 0:
            inst_fps = 1.0 / dt
            ema_fps = inst_fps if ema_fps == 0 else (alpha * inst_fps + (1 - alpha) * ema_fps)
        frames += 1

        # Overlay con info
        overlay_lines = [
            f"FPS(meas): {ema_fps:5.1f}",
            f"FPS(cam):  {eff_fps:5.1f}" if eff_fps and eff_fps > 0 else "FPS(cam):  n/a",
            f"FOURCC: {eff_fourcc}",
            f"Res: {frame.shape[1]}x{frame.shape[0]}",
        ]
        y = 24
        for line in overlay_lines:
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            y += 24

        cv2.imshow(win_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("[INFO] Salida por ESC.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

