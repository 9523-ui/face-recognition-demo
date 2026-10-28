"""
配置与日志
===========
配置从代码里搬出来，放到 `settings.json`。理由很实际：

  · 阈值这类参数**应该按现场数据改**，不该改源码。改源码意味着要重新测、
    重新打包，而且一旦有人手滑改错了代码逻辑，排查成本远高于改一个数字。
  · 部署到不同机器（不同摄像头、不同光线）需要不同阈值，同一份代码要能带不同配置跑。
  · 密码/路径这类东西不该硬编码在源码里。

优先级：**环境变量 > settings.json > 内置默认值**。
环境变量前缀 `FACE_DEMO_`，例如 `FACE_DEMO_MATCH_THRESH=0.55`，
方便 CI 或临时实验覆盖，不用改文件。

关于 `match_thresh` 的默认值，得说清楚：
  SFace 官方给的 0.363 来自 LFW **1:1 验证**（只比一对图）。
  本项目实际是 **1:N 检索**（从库里挑一个），两者风险差一个量级 ——
  `bench_scale.py` 实测：按 0.363 判定，陌生人误接受率高达 30%+。
  但把默认值直接提到 0.55 又会让「录入条件差的库」完全认不出人。
  所以默认取 0.45（比官方的严、比数据建议的松），
  **正确做法是跑一遍 diagnose.py 看你自己的同人相似度，再据此定值**。
"""
import json
import logging
import logging.handlers
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(BASE, "settings.json")
LOG_PATH = os.path.join(BASE, "data", "app.log")

# 内置默认值。settings.json 里只写需要覆盖的项。
DEFAULTS = {
    "match_thresh": 0.45,        # 判定为「就是这个人」的最低余弦相似度
    "match_margin": 0.045,       # 第一名领先第二名不足此值时弃权（只跟不同人比）
    "min_quality": 0.45,         # 质量分低于此值：不投票、不入库
    "det_thresh": 0.60,          # YuNet 检测置信度
    "recog_every": 2,            # 每 N 帧做一次特征提取
    "vote_n": 7,                 # 多帧投票窗口
    "feat_avg_n": 5,             # 特征滑动平均窗口
    "require_liveness": True,    # 是否强制眨眼通过才给结论
    "db_path": os.path.join("data", "faces.db"),
    "log_level": "INFO",
    "log_max_mb": 2,
    "log_backups": 3,
}

_DOC = {
    "_README": "本文件由程序自动生成。改完保存后重启程序生效。删除本文件会恢复默认值。",
    "_match_thresh": "1:N 检索的判定阈值。0.363 是 SFace 的 1:1 阈值，用在这里太松"
                     "（实测陌生人误接受 30%+）。先跑 diagnose.py 看同人相似度，"
                     "正常应 ≥0.7；再按 bench_scale.py 的建议定值（几十人库通常 0.55~0.65）。",
    "_require_liveness": "true = 必须眨眼通过才显示名字（防照片）。纯演示可设 false。",
}


class Settings:
    def __init__(self, path=SETTINGS_PATH, autoload=True):
        self.path = path
        self._data = dict(DEFAULTS)
        self._source = {}          # 记录每个键来自哪里，便于排查「为什么是这个值」
        if autoload:
            self.load()

    # ---------------------------------------------------------- 读写
    def load(self):
        for k, v in DEFAULTS.items():
            self._source[k] = "default"
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    user = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                # 配置坏了不能让程序起不来，退回默认值并明确告警
                self._data = dict(DEFAULTS)
                print(f"[配置] {os.path.basename(self.path)} 解析失败（{e}），已退回默认值",
                      file=sys.stderr)
                return self
            for k, v in user.items():
                if k.startswith("_"):
                    continue
                if k in DEFAULTS:
                    self._data[k] = v
                    # 文件里写了但值等于默认，说明是首次自动生成的，不算用户改过
                    self._source[k] = "default" if v == DEFAULTS[k] else "settings.json"
                else:
                    self._data[k] = v            # 允许自定义键，程序不认也无害
                    self._source[k] = "settings.json(extra)"
        self.apply_env()
        return self

    def apply_env(self):
        for k in list(self._data.keys()):
            env = os.environ.get("FACE_DEMO_" + k.upper())
            if env is None:
                continue
            cur = self._data[k]
            try:
                if isinstance(cur, bool):
                    val = env.strip().lower() in ("1", "true", "yes", "on")
                elif isinstance(cur, int):
                    val = int(env)
                elif isinstance(cur, float):
                    val = float(env)
                else:
                    val = env
            except ValueError:
                print(f"[配置] 环境变量 FACE_DEMO_{k.upper()}={env!r} 类型不对，已忽略",
                      file=sys.stderr)
                continue
            self._data[k] = val
            self._source[k] = "env"

    def save(self):
        """把当前值写回文件（保留 _README 说明，便于人工阅读）。"""
        out = dict(_DOC)
        out.update(self._data)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
            f.write("\n")
        return self.path

    def ensure_file(self):
        """首次运行时生成带说明的配置文件，之后不再覆盖用户的修改。"""
        if not os.path.exists(self.path):
            self.save()
            return True
        return False

    # ---------------------------------------------------------- 取值
    def get(self, key, default=None):
        return self._data.get(key, DEFAULTS.get(key, default))

    def __getitem__(self, key):
        return self.get(key)

    def abspath(self, key):
        """把配置里的相对路径解析成绝对路径（相对本文件所在目录）。"""
        p = str(self.get(key))
        return p if os.path.isabs(p) else os.path.join(BASE, p)

    def origin(self, key):
        return self._source.get(key, "?")

    def as_dict(self):
        return dict(self._data)


# 全局单例：各模块 `from config import S` 直接用
S = Settings()


# ---------------------------------------------------------------- 日志
def setup_logging(name="face_demo", level=None, console=True):
    """文件日志（按大小轮转）+ 可选控制台输出。

    GUI 程序没有控制台，出错时什么都看不到 —— 所以必须落盘。
    轮转上限 2MB × 3 份，避免长期运行把磁盘写满。
    """
    level = level or S.get("log_level", "INFO")
    logger = logging.getLogger(name)
    if logger.handlers:                    # 重复调用不重复挂 handler
        return logger
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = False

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        LOG_PATH, maxBytes=int(S.get("log_max_mb", 2)) * 1024 * 1024,
        backupCount=int(S.get("log_backups", 3)), encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    if console and sys.stdout is not None:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
        ch.setLevel(logging.WARNING)       # 控制台只出警告以上，别刷屏
        logger.addHandler(ch)
    return logger


def get_logger(name="face_demo"):
    return setup_logging(name)
