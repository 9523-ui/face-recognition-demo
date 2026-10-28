"""
人脸识别 · 录入管理界面
=====================
图形界面版，不用记命令行参数：看着画面直接点按钮加人脸。

启动：
    python gui.py           或双击 run_gui.bat

界面说明：
    左侧 = 摄像头实时画面（绿框=已录入，橙框=陌生人）
    右侧 = 输入姓名 → 点「采集并添加」→ 自动连采 8 帧 → 入库
           列表里选中某人 → 点「删除选中」即可移除
"""
import os
import sys
import time

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

from config import get_logger
from face_demo import (FaceDetector, FaceEncoder, FaceLibrary, Tracker,
                       annotate, count_persons, update_liveness, face_quality,
                       MATCH_THRESH, MIN_QUALITY, RECOG_EVERY)

log = get_logger()

# ---------------------------------------------------------------- 参数
CAM_INDEX = 0
CAM_W, CAM_H = 640, 480
COLLECT_N = 8          # 采集多少帧求平均
COLLECT_EVERY = 8      # 每隔几帧采一次（8 帧 × 8 ≈ 2 秒）
COLLECT_TIMEOUT = 90   # 连续多少帧没脸就放弃（约 3 秒）
FPS_SMOOTH = 0.85

# 布局（固定尺寸，避免不同系统下被挤压变形）
WIN_W, WIN_H = 984, 578
LEFT_X, LEFT_Y, LEFT_W, LEFT_H = 14, 12, 642, 520
RIGHT_X, RIGHT_W = 670, 300

BG = "#f4f5f7"
CARD = "#ffffff"
LINE = "#dcdfe4"
INK = "#1f2328"
SUB = "#6b7280"
ACCENT = "#1f3864"
OK = "#0f7b3f"
WARN = "#b45309"


class FaceApp:
    def __init__(self, root):
        self.root = root
        root.title("人脸识别 · 录入管理")
        root.configure(bg=BG)
        root.resizable(False, False)

        # ---- 模型与库 ----
        self.det = FaceDetector()
        self.enc = FaceEncoder()
        self.lib = FaceLibrary()
        self.tracker = Tracker()
        log.info("人脸库已加载：%s", self.lib.stats())

        # ---- 摄像头 ----
        self.cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cam_ok = self.cap.isOpened()

        # ---- 运行状态 ----
        self.frame_no = 0
        self.fps = 0.0
        self._t_prev = time.perf_counter()
        self.mode = "live"          # live | collect
        self.samples = []
        self._sample_q = []
        self._since_sample = 0
        self._miss = 0
        self._pending_name = ""
        self._banner = ""           # 画面底部提示
        self._warn = ""             # 画面顶部红色告警（多人时用）
        self._multi_on = False      # 是否因为画面里出现多个人而暂停采集
        self._list_names = []
        self._audit_ts = {}         # 审计限流：每个身份上次记日志的时间
        self._photo = None
        self._stress_limit = 0

        self._build_ui()
        self.refresh_list()
        if not self.cam_ok:
            self.set_status("摄像头打不开，请检查是否被其他程序占用", WARN)
        else:
            self.set_status("就绪：右侧输入姓名即可添加人脸", SUB)
        self.tick()

    # ------------------------------------------------------------ 界面
    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TButton", font=("Microsoft YaHei UI", 10))
        style.configure("TLabel", background=BG, foreground=INK)

        self.root.geometry(f"{WIN_W}x{WIN_H}")

        # ================= 左：摄像头画面 =================
        left = tk.Frame(self.root, bg=CARD, highlightbackground=LINE,
                        highlightthickness=1)
        left.place(x=LEFT_X, y=LEFT_Y, width=LEFT_W, height=LEFT_H)

        self.canvas = tk.Canvas(left, width=CAM_W, height=CAM_H,
                                bg="#111418", highlightthickness=0)
        self.canvas.pack()
        # 只创建一次 PhotoImage，之后每帧 paste 更新内容。
        # 每帧新建 ImageTk.PhotoImage 会让 Tk 侧图像对象频繁创建/销毁，
        # 长时间运行容易累积泄漏，实测跑一会儿会段错误退出。
        self._photo = ImageTk.PhotoImage(Image.new("RGB", (CAM_W, CAM_H), (17, 20, 24)))
        self.canvas_img = self.canvas.create_image(CAM_W // 2, CAM_H // 2,
                                                   image=self._photo)

        tk.Label(left, text="绿=已通过活体   黄=待眨眼   橙=陌生人   灰=画面不清   Esc 退出",
                 bg=CARD, fg=SUB, font=("Microsoft YaHei UI", 9)).pack(pady=(8, 0))

        # ================= 右：控制面板 =================
        right = tk.Frame(self.root, bg=BG)
        right.place(x=RIGHT_X, y=LEFT_Y, width=RIGHT_W, height=LEFT_H)

        # ---------- 卡片一：添加人脸 ----------
        card1 = tk.Frame(right, bg=CARD, highlightbackground=LINE,
                         highlightthickness=1)
        card1.pack(fill="x")

        inner1 = tk.Frame(card1, bg=CARD, padx=14, pady=12)
        inner1.pack(fill="x")

        tk.Label(inner1, text="添加人脸", bg=CARD, fg=ACCENT,
                 font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w")
        tk.Label(inner1, text="姓名", bg=CARD, fg=SUB,
                 font=("Microsoft YaHei UI", 9)).pack(anchor="w", pady=(8, 3))

        self.name_var = tk.StringVar()
        self.entry = tk.Entry(inner1, textvariable=self.name_var,
                              font=("Microsoft YaHei UI", 11), relief="solid",
                              bd=1, highlightthickness=0, width=22)
        self.entry.pack(fill="x", ipady=4)
        self.entry.bind("<Return>", lambda e: self.on_add_click())

        self.btn_add = tk.Button(inner1, text="采集并添加", command=self.on_add_click,
                                 bg=ACCENT, fg="white", relief="flat",
                                 activebackground="#2c4a7c", activeforeground="white",
                                 font=("Microsoft YaHei UI", 10, "bold"),
                                 cursor="hand2", pady=6)
        self.btn_add.pack(fill="x", pady=(10, 0))

        self.live_var = tk.BooleanVar(value=True)
        self.chk_live = tk.Checkbutton(
            inner1, text="录入前校验眨眼（防止用照片冒充）",
            variable=self.live_var, bg=CARD, fg=SUB, selectcolor=CARD,
            activebackground=CARD, activeforeground=INK, bd=0,
            highlightthickness=0, anchor="w", cursor="hand2",
            font=("Microsoft YaHei UI", 9), command=self.on_live_toggle)
        self.chk_live.pack(fill="x", pady=(8, 0))

        self.hint = tk.Label(inner1, text="正对镜头点按钮，自动采集约 2 秒",
                             bg=CARD, fg=SUB, font=("Microsoft YaHei UI", 8),
                             wraplength=250, justify="left")
        self.hint.pack(anchor="w", pady=(6, 0))

        # ---------- 卡片二：已录入列表 ----------
        card2 = tk.Frame(right, bg=CARD, highlightbackground=LINE,
                         highlightthickness=1)
        card2.pack(fill="both", expand=True, pady=(10, 0))

        inner2 = tk.Frame(card2, bg=CARD, padx=14, pady=12)
        inner2.pack(fill="both", expand=True)

        head = tk.Frame(inner2, bg=CARD)
        head.pack(fill="x")
        tk.Label(head, text="已录入", bg=CARD, fg=ACCENT,
                 font=("Microsoft YaHei UI", 12, "bold")).pack(side="left")
        self.count_label = tk.Label(head, text="", bg=CARD, fg=SUB,
                                    font=("Microsoft YaHei UI", 9))
        self.count_label.pack(side="right")

        box = tk.Frame(inner2, bg=CARD)
        box.pack(fill="both", expand=True, pady=(8, 0))
        scroll = tk.Scrollbar(box)
        scroll.pack(side="right", fill="y")
        self.listbox = tk.Listbox(box, font=("Microsoft YaHei UI", 10),
                                  relief="solid", bd=1, highlightthickness=0,
                                  activestyle="none", selectbackground="#dbe5f5",
                                  selectforeground=INK, yscrollcommand=scroll.set,
                                  height=7)
        self.listbox.pack(side="left", fill="both", expand=True)
        scroll.config(command=self.listbox.yview)

        btns = tk.Frame(inner2, bg=CARD)
        btns.pack(fill="x", pady=(10, 0))
        self.btn_del = tk.Button(btns, text="删除选中", command=self.delete_selected,
                                 bg=CARD, fg="#b91c1c", relief="solid", bd=1,
                                 activebackground="#fee2e2",
                                 font=("Microsoft YaHei UI", 9), cursor="hand2", pady=5)
        self.btn_del.pack(side="left", fill="x", expand=True)
        self.btn_clear = tk.Button(btns, text="清空全部", command=self.clear_all,
                                   bg=CARD, fg=SUB, relief="solid", bd=1,
                                   activebackground="#f3f4f6",
                                   font=("Microsoft YaHei UI", 9), cursor="hand2", pady=5)
        self.btn_clear.pack(side="left", fill="x", expand=True, padx=(8, 0))

        # ================= 底部状态栏 =================
        self.status = tk.Label(self.root, text="就绪", bg="#e6e8ec", fg=SUB,
                               anchor="w", padx=16,
                               font=("Microsoft YaHei UI", 9))
        self.status.place(x=0, y=WIN_H - 34, width=WIN_W, height=34)

        self.root.bind("<Escape>", lambda e: self.on_close())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------ 状态栏
    def set_status(self, text, color=SUB):
        # 每帧都 set 会让 Tk 反复重排文字；内容和颜色没变就跳过
        if self.status.cget("text") == text and self.status.cget("fg") == color:
            return
        self.status.config(text=text, fg=color)

    def set_hint(self, text, color=SUB):
        if self.hint.cget("text") == text and self.hint.cget("fg") == color:
            return
        self.hint.config(text=text, fg=color)

    def on_live_toggle(self):
        if self.live_var.get():
            self.set_status("已开启活体校验：识别时要眨一次眼才算通过", ACCENT)
            self.set_hint("识别时先眨一下眼才显示姓名。\n录入也会先校验眨眼，照片冒充不过去。", SUB)
        else:
            self.set_status("已关闭活体校验：只做身份识别，不防照片", WARN)
            self.set_hint("活体校验已关闭，拿照片对着镜头也能认出。\n演示身份识别时可以关，真实验证时建议打开。", WARN)

    # ------------------------------------------------------------ 主循环
    def tick(self):
        if not self.cam_ok:
            self.root.after(400, self.tick)
            return

        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.set_status("读取画面失败，正在重试…", WARN)
            self.root.after(200, self.tick)
            return

        self.frame_no += 1
        faces = self.det(frame)
        if faces is None:
            faces = np.zeros((0, 15), np.float32)

        tracks = self.tracker.update([tuple(f[:4]) for f in faces])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 活体：每帧都推进（只做两次小 ROI 比较，开销可忽略）。
        # 开关只影响「是否据此判定」，这样勾选框来回切不用重建状态。
        update_liveness(tracks, gray, faces)

        n_person, _ = count_persons(faces)

        # 识别：实时模式每 N 帧一次；采集模式每帧都算
        if self.frame_no % RECOG_EVERY == 0 or self.mode == "collect":
            for t in tracks:
                if t.idx is None:
                    continue
                # 先看这一帧值不值得采信：太小、太糊、太暗、侧脸，就别拿去投票，
                # 否则垃圾帧会把滑动窗口里的正确结论冲掉
                q, qd = face_quality(gray, faces[t.idx])
                t.quality, t.qdetail = q, qd
                if q < MIN_QUALITY:
                    continue
                feat = self.enc.encode(frame, faces[t.idx])
                name, score = self.lib.match(t.push_feat(feat))   # 用几帧平均后的特征比对
                t.push_vote(name, score)

            # 审计：认出来的人记一条。同一个人 30 秒内只记一次，
            # 否则 30fps 下日志会被刷成几十万行，反而没法查。
            self._audit_recognitions(tracks)

        if self.mode == "collect":
            self._collect_step(frame, gray, faces, tracks, n_person)

        now = time.perf_counter()
        dt = now - self._t_prev
        self._t_prev = now
        if dt > 0:
            inst = 1.0 / dt
            self.fps = inst if self.fps == 0 else FPS_SMOOTH * self.fps + (1 - FPS_SMOOTH) * inst

        extra = self._banner or None
        if self.mode != "collect" and len(self.lib.names) == 0:
            extra = "人脸库为空：右侧输入姓名后点「采集并添加」"

        vis = annotate(frame, faces, tracks, fps=self.fps, extra=extra,
                       require_liveness=self.live_var.get(), warn=self._warn)
        self._show(vis)

        if self._stress_limit and self.frame_no >= self._stress_limit:
            print(f"STRESS OK 跑满 {self.frame_no} 帧未崩溃 | fps {self.fps:.1f} | "
                  f"库内 {len(self.lib.names)} 条", flush=True)
            self.on_close()
            return

        self.root.after(8, self.tick)

    def _show(self, vis):
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        if img.size != (CAM_W, CAM_H):
            img = img.resize((CAM_W, CAM_H))
        self._photo.paste(img)          # 复用同一个 Tk 图像对象，不再每帧新建

    # ------------------------------------------------------------ 采集
    def on_add_click(self):
        """按钮一个身兼两职：空闲时开始采集，采集中则取消。"""
        if self.mode == "collect":
            self.cancel_collect("采集已取消")
        else:
            self.start_collect()

    def start_collect(self):
        if self.mode == "collect":
            return
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("还差一步", "请先输入姓名")
            self.entry.focus_set()
            return
        if len(name) > 16:
            messagebox.showwarning("姓名太长", "请控制在 16 个字符以内")
            return
        if not self.cam_ok:
            messagebox.showerror("无法采集", "摄像头没有打开")
            return

        self._pending_name = name
        self.samples = []
        self._sample_q = []
        self._since_sample = 0
        self._miss = 0
        self._multi_on = False
        self._warn = ""
        self._banner = ""
        self.mode = "collect"
        self.btn_add.config(state="normal", text="取消采集", bg="#8a94a6")
        self.entry.config(state="disabled")
        if self.live_var.get():
            self.set_status(f"正在校验活体，请眨一下眼（录入「{name}」）…", ACCENT)
            self.set_hint("第一步：眨一次眼证明是真人。\n通过后自动开始采集，请保持画面里只有你一个人。", ACCENT)
        else:
            self.set_status(f"正在采集「{name}」，请正对镜头…", ACCENT)

    def _collect_step(self, frame, gray, faces, tracks, n_person):
        # ---------- 1. 画面里不止一个人：立刻暂停，一帧都不采 ----------
        if n_person > 1:
            if not self._multi_on:
                self._multi_on = True
                try:
                    self.root.bell()        # 响一下，避免用户只盯着屏幕没注意
                except Exception:
                    pass
            self._miss = 0
            self._since_sample = 0
            self._warn = "录入时请保持画面中只有一个人"
            self._banner = f"检测到 {n_person} 人 · 采集已暂停"
            self.set_status(f"检测到 {n_person} 个人：录入时请保持画面中只有一个人（已暂停采集）", WARN)
            self.set_hint("画面里出现了多个人，采集已暂停。\n其他人离开画面、只剩一个人后会自动继续。", WARN)
            return
        if self._multi_on:
            self._multi_on = False
            self._warn = ""
            self.hint.config(text="画面恢复为单人，继续采集…", fg=OK)

        # ---------- 2. 一个人都没有 ----------
        if n_person == 0:
            self._miss += 1
            self._banner = "没有检测到人脸"
            self.set_status("采集暂停：画面里没有检测到人脸", WARN)
            if self._miss > COLLECT_TIMEOUT:
                self.cancel_collect("没检测到人脸，采集已取消。请正对镜头再试一次")
            return
        self._miss = 0

        # ---------- 3. 锁定主脸（面积最大的那张）并挂到 track 上 ----------
        main_idx = int(np.argmax([float(f[2]) * float(f[3]) for f in faces]))
        track = next((t for t in tracks if t.idx == main_idx), None)
        if track is None:
            self._since_sample = 0
            return

        # ---------- 4. 活体：先眨一次眼，通过后才允许采特征 ----------
        if self.live_var.get():
            st = getattr(track, "blink", None)
            if st is None or not st.passed:
                self._since_sample = 0
                self._banner = "请眨一下眼睛完成活体校验"
                self.set_status(f"活体校验：{st.label() if st else '校验中…'}（录入「{self._pending_name}」）", ACCENT)
                self.set_hint("第一步：眨一次眼证明是真人。\n通过后自动开始采集，请保持画面里只有你一个人。", ACCENT)
                return

        # ---------- 5. 质量不够的帧不收：入库存的是「好样本」，库才不容易跑偏 ----------
        q, qd = face_quality(gray, faces[main_idx])
        track.quality, track.qdetail = q, qd
        if q < MIN_QUALITY:
            self._since_sample = 0
            why = self._quality_hint(qd)
            self._banner = f"画面质量不够 · {why}"
            self.set_status(f"采集暂停：{why}（质量 {q:.2f}，需要 ≥{MIN_QUALITY}）", WARN)
            self.set_hint(f"这一帧不适合入库：{why}。\n调整好之后会自动继续，不会白等。", WARN)
            return

        # ---------- 6. 正常采样 ----------
        self._banner = f"采集中 {len(self.samples)}/{COLLECT_N} · 可轻微换角度"
        self._since_sample += 1
        if self._since_sample < COLLECT_EVERY:
            return
        self._since_sample = 0

        self.samples.append(self.enc.encode(frame, faces[main_idx]))
        self._sample_q.append(q)
        self.set_status(f"采集中「{self._pending_name}」 {len(self.samples)}/{COLLECT_N}，"
                        f"可以稍微换换角度", ACCENT)
        self.set_hint(f"第二步：正在采特征，质量 {q:.2f} 合格。\n脸可以小幅左右转一下，多角度会更稳。", ACCENT)

        if len(self.samples) >= COLLECT_N:
            self.finish_collect()

    @staticmethod
    def _quality_hint(qd):
        """找出最拖后腿的那一项，直接告诉用户该怎么做。

        全黑画面亮度和清晰度会同时塌到 0，这时候提「光线」才对 ——
        没有光的时候，晃不晃根本看不出来。
        """
        if qd.get("亮度", 1.0) < 0.2:
            return "画面太黑，请开灯或换个亮一点的位置"
        if qd.get("清晰", 1.0) < 0.2:
            return "画面发虚，端稳一点或擦下镜头"
        worst = min(("尺寸", "清晰", "亮度", "正脸"), key=lambda k: qd.get(k, 1.0))
        return {"尺寸": "人脸太小，请坐近一点",
                "清晰": "画面发虚，端稳一点或擦下镜头",
                "亮度": "光线不好，换个位置或开灯",
                "正脸": "脸偏得太厉害，请正对镜头"}[worst]

    def finish_collect(self):
        feats = np.stack(self.samples)
        mean = feats.mean(axis=0)
        mean = mean / (np.linalg.norm(mean) + 1e-9)
        name = self._pending_name
        q_mean = float(np.mean(self._sample_q)) if self._sample_q else None
        self.lib.add(name, mean.astype(np.float32), quality=q_mean)

        # 同名样本之间的一致性：太散说明采集中晃得厉害
        spread = float(np.mean([float(f @ mean) for f in feats]))

        self.mode = "live"
        self.samples = []
        self._sample_q = []
        self._banner = ""
        self._warn = ""
        self._reset_add_button()
        self.refresh_list()

        live_note = "，已通过活体校验" if self.live_var.get() else ""
        self.set_hint(f"「{name}」已入库（一致性 {spread:.3f}）{live_note}。\n"
                      f"换个角度再点一次可补充样本，识别更稳。", OK)
        self.set_status(f"已添加「{name}」：{COLLECT_N} 帧平均特征入库，一致性 {spread:.3f}", OK)

    def cancel_collect(self, why):
        self.mode = "live"
        self.samples = []
        self._sample_q = []
        self._banner = ""
        self._warn = ""
        self._multi_on = False
        self._reset_add_button()
        self.set_status(why, WARN)
        self.set_hint("采集没有完成。请让脸正对镜头、光线充足，再点一次。", WARN)

    def _reset_add_button(self):
        self.btn_add.config(state="normal", text="采集并添加", bg=ACCENT)
        self.entry.config(state="normal")

    # ------------------------------------------------------------ 审计
    def _audit_recognitions(self, tracks, window=30.0):
        """把「认出了谁」写进审计表。同一身份 window 秒内只记一条。"""
        now = time.time()
        for t in tracks:
            name = t.name
            if name in ("未知", "未注册") or t.idx is None:
                continue
            last = self._audit_ts.get(name, 0.0)
            if now - last < window:
                continue
            self._audit_ts[name] = now
            try:
                self.lib.audit("recognize", name, f"score={t.score:.3f}")
            except Exception as e:               # 审计失败不能拖垮识别
                log.warning("写审计失败: %s", e)

    # ------------------------------------------------------------ 库管理
    def refresh_list(self):
        self.listbox.delete(0, tk.END)
        self._list_names = []
        cnt = self.lib.count_by_name()
        for nm, c in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])):
            self.listbox.insert(tk.END, f"  {nm}    ·  {c}")
            self._list_names.append(nm)
        self.count_label.config(text=f"{len(cnt)} 人 / {len(self.lib.names)} 样本")

    def delete_selected(self):
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showinfo("提示", "请先在列表里选中一个人")
            return
        name = self._list_names[sel[0]]
        if not messagebox.askyesno("确认删除", f"删除「{name}」的所有样本？"):
            return

        # 走接口而不是改快照：快照改了数据库不会变，而且不会留审计记录。
        n = self.lib.delete(name)
        self.refresh_list()
        self.set_status(f"已删除「{name}」（{n} 条样本）", SUB)

    def clear_all(self):
        if len(self.lib.names) == 0:
            messagebox.showinfo("提示", "人脸库已经是空的")
            return
        if not messagebox.askyesno("确认清空", "清空整个人脸库？此操作不可恢复。"):
            return
        p, s = self.lib.clear()
        self.refresh_list()
        self.set_status(f"已清空：{p} 人 / {s} 条样本（审计记录保留）", SUB)

    # ------------------------------------------------------------ 自检截图
    def selfshot(self, root, delay_note=""):
        """把窗口自身截下来存到 out/，用于确认界面渲染是否正常。"""
        try:
            from PIL import ImageGrab
            root.update_idletasks()
            root.update()
            time.sleep(0.35)
            x, y = root.winfo_rootx(), root.winfo_rooty()
            w, h = root.winfo_width(), root.winfo_height()
            full = ImageGrab.grab()
            d = os.path.dirname(os.path.abspath(__file__))
            os.makedirs(os.path.join(d, "out"), exist_ok=True)
            full.save(os.path.join(d, "out", "gui_shot_full.png"))
            # 全屏图是物理像素，Tk 是逻辑像素，按屏幕尺寸比换算缩放系数
            sx = full.size[0] / root.winfo_screenwidth()
            sy = full.size[1] / root.winfo_screenheight()
            box = (int(x * sx), int(y * sy), int((x + w) * sx), int((y + h) * sy))
            im = full.crop(box)
            p = os.path.join(d, "out", "gui_shot.png")
            im.save(p)
            print(f"SHOT {p} {im.size[0]}x{im.size[1]} | win={w}x{h}@{x},{y} "
                  f"| screen {full.size[0]}x{full.size[1]} vs tk "
                  f"{root.winfo_screenwidth()}x{root.winfo_screenheight()} scale={sx:.3f} {delay_note}")
        except Exception as e:
            print("SHOT FAILED:", e)
        finally:
            self.on_close()

    # ------------------------------------------------------------ 退出
    def on_close(self):
        try:
            self.cap.release()
        except Exception:
            pass
        self.root.destroy()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(here, "models", "yunet.onnx")):
        print("缺少模型，正在下载…")
        os.system(f'"{sys.executable}" "{os.path.join(here, "download_models.py")}"')

    root = tk.Tk()
    app = FaceApp(root)

    # python gui.py --shot  →  2.6 秒后自动截图并退出（自检用）
    if "--shot" in sys.argv:
        root.attributes("-topmost", True)
        root.after(2600, lambda: app.selfshot(root))

    # python gui.py --selftest=30  →  跑 30 秒后自动退出，用于验证长时间运行是否会崩
    for _a in sys.argv:
        if _a.startswith("--selftest"):
            secs = int(_a.split("=")[1]) if "=" in _a else 20
            print(f"SELFTEST start {secs}s", flush=True)

            def _done():
                print(f"SELFTEST OK 存活 {secs}s | 帧数 {app.frame_no} | "
                      f"fps {app.fps:.1f} | 库内 {len(app.lib.names)} 条", flush=True)
                app.on_close()

            root.after(secs * 1000, _done)

        # python gui.py --stress=600  →  隐藏窗口跑满 600 帧（不打扰人），验证渲染路径稳定
        elif _a.startswith("--stress"):
            n = int(_a.split("=")[1]) if "=" in _a else 600
            app._stress_limit = n
            root.withdraw()
            print(f"STRESS start {n} 帧（窗口隐藏）", flush=True)

    root.mainloop()


if __name__ == "__main__":
    main()
