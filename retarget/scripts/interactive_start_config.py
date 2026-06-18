"""
Interactive start_config editor.
Sliders window controls all arm joints live in the MuJoCo viewer.
Press "Print & Quit" to print the config to the terminal (blue = left, orange = right).

Usage
-----
    python3 scripts/interactive_start_config.py
    python3 scripts/interactive_start_config.py --robot franka
    python3 scripts/interactive_start_config.py --robot xlerobot
"""

import argparse, os, sys, numpy as np, mujoco, mujoco.viewer
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from sim.robots import ROBOT_CONFIGS, ENV_CONFIGS

# ── ANSI colours ──────────────────────────────────────────────────────────────
_BLUE   = "\033[94m"
_ORANGE = "\033[38;5;214m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_RESET  = "\033[0m"

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Interactive start_config editor")
parser.add_argument("--robot", default="g1", choices=list(ROBOT_CONFIGS.keys()))
args = parser.parse_args()

robot_cfg = ROBOT_CONFIGS[args.robot]
env_cfg   = ENV_CONFIGS[args.robot]

# ── environment ───────────────────────────────────────────────────────────────
env = robot_cfg["env_cls"](mjcf_path=robot_cfg["scene_path"],
                            start_config=robot_cfg["start_config"])
env.reset()

joint_names_l = env_cfg.joint_groups["left"]
joint_names_r = env_cfg.joint_groups["right"]
n_dof = len(joint_names_l)

# Read joint limits directly from the loaded scene model.
def _jnt_range(model, name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError(f"Joint '{name}' not found in {robot_cfg['scene_path']}")
    return (float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1]))

limits_l = [_jnt_range(env.model, n) for n in joint_names_l]
limits_r = [_jnt_range(env.model, n) for n in joint_names_r]

# Strip side prefix/suffix for readable slider labels.
def _display_name(full_name):
    name = full_name.split("/")[-1]
    for tok in ("left_", "right_", "_L", "_R", "_joint", "joint"):
        name = name.replace(tok, "")
    return name or full_name

display_names = [_display_name(n) for n in joint_names_l]

# Heuristic mirror flip: roll/yaw/rotation joints negate for bilateral symmetry.
flip = [-1 if any(k in n.lower() for k in ("roll", "yaw", "rotation")) else 1
        for n in joint_names_l]

q_left  = robot_cfg["start_config"]["left"].copy()
q_right = robot_cfg["start_config"]["right"].copy()

# ── Tkinter sliders ───────────────────────────────────────────────────────────
root = tk.Tk()
root.title(f"{args.robot}  start_config  editor")
root.resizable(False, False)

sliders_l, sliders_r = [], []
val_labels_l, val_labels_r = [], []

def on_change(*_):
    for i in range(n_dof):
        q_left[i]  = sliders_l[i].get()
        q_right[i] = sliders_r[i].get()
    for i in range(n_dof):
        val_labels_l[i].config(text=f"{sliders_l[i].get():.3f}")
        val_labels_r[i].config(text=f"{sliders_r[i].get():.3f}")

def mirror_to_right():
    for i in range(n_dof):
        sliders_r[i].set(sliders_l[i].get() * flip[i])
    on_change()

def reset_to_zero():
    for i in range(n_dof):
        sliders_l[i].set(0.0)
        sliders_r[i].set(0.0)
    on_change()

def print_and_quit():
    ql = np.array([sliders_l[i].get() for i in range(n_dof)], dtype=np.float32)
    qr = np.array([sliders_r[i].get() for i in range(n_dof)], dtype=np.float32)
    ql_str = [round(float(v), 4) for v in ql]
    qr_str = [round(float(v), 4) for v in qr]
    print(f"\n{_BOLD}─── {args.robot}  start_config {'─' * 38}{_RESET}")
    print( '    "start_config": {')
    print(f'        "left":  {_BLUE}np.array({ql_str}, dtype=np.float32){_RESET},')
    print(f'        "right": {_ORANGE}np.array({qr_str}, dtype=np.float32){_RESET},')
    print( '    },')
    print(f"{_DIM}{'─' * 62}{_RESET}\n")
    os._exit(0)

# ── layout ────────────────────────────────────────────────────────────────────
header = tk.Frame(root, bg="#2b2b2b")
header.pack(fill="x")
tk.Label(header, text=f"{args.robot.upper()}  start_config  editor",
         bg="#2b2b2b", fg="white", font=("Helvetica", 13, "bold"), pady=6).pack()

body = tk.Frame(root, padx=12, pady=8)
body.pack()
tk.Label(body, text="Joint", width=16, anchor="w", font=("Helvetica",10,"bold")).grid(row=0, column=0)
tk.Label(body, text="Left",  width=28, anchor="c", font=("Helvetica",10,"bold")).grid(row=0, column=1, columnspan=2)
tk.Label(body, text="Right", width=28, anchor="c", font=("Helvetica",10,"bold")).grid(row=0, column=3, columnspan=2)

for i, (name, ll, rl) in enumerate(zip(display_names, limits_l, limits_r)):
    row = i + 1
    tk.Label(body, text=name, width=16, anchor="w").grid(row=row, column=0, sticky="w")

    sl = tk.Scale(body, from_=ll[0], to=ll[1], resolution=0.001, orient="horizontal",
                  length=220, command=on_change, showvalue=False)
    sl.set(float(q_left[i]))
    sl.grid(row=row, column=1)
    sliders_l.append(sl)

    vl = tk.Label(body, text=f"{q_left[i]:.3f}", width=7, font=("Courier", 9))
    vl.grid(row=row, column=2, padx=4)
    val_labels_l.append(vl)

    sr = tk.Scale(body, from_=rl[0], to=rl[1], resolution=0.001, orient="horizontal",
                  length=220, command=on_change, showvalue=False)
    sr.set(float(q_right[i]))
    sr.grid(row=row, column=3)
    sliders_r.append(sr)

    vr = tk.Label(body, text=f"{q_right[i]:.3f}", width=7, font=("Courier", 9))
    vr.grid(row=row, column=4, padx=4)
    val_labels_r.append(vr)

btn_frame = tk.Frame(root, pady=8)
btn_frame.pack()
tk.Button(btn_frame, text="Mirror L→R",   width=14, command=mirror_to_right).pack(side="left", padx=6)
tk.Button(btn_frame, text="Reset to Zero",width=14, command=reset_to_zero).pack(side="left", padx=6)
tk.Button(btn_frame, text="Print & Quit", width=14, bg="#2b5fa8", fg="white",
          font=("Helvetica", 10, "bold"), command=print_and_quit).pack(side="left", padx=6)

# ── main loop: viewer + tkinter on the same (main) thread ─────────────────────
# launch_passive opens a GLFW window and runs its own render loop internally.
# We drive sync() from tkinter's after() so GLFW stays on the main thread,
# avoiding the segfault that occurs when launch_passive is called from a thread.
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    viewer.cam.azimuth   = robot_cfg["cam_azimuth"]
    viewer.cam.elevation = robot_cfg["cam_elevation"]
    viewer.cam.distance  = robot_cfg["cam_distance"]
    viewer.cam.lookat[:] = robot_cfg["cam_lookat"]

    def _sync():
        if not viewer.is_running():
            os._exit(0)
            return
        env.set_arm_joints("left",  q_left)
        env.set_arm_joints("right", q_right)
        mujoco.mj_forward(env.model, env.data)
        viewer.sync()
        root.after(33, _sync)

    root.after(0, _sync)
    root.mainloop()
