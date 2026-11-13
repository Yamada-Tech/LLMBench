"""GUI の実サーバー向けスモーク確認（開発用ユーティリティ）.

    python -X utf8 _gui_real_check.py <base_url> <model>

BenchApp を実描画し、ワーカールートを同期実行して結果反映まで見る。
"""

import queue
import sys
import tempfile
from pathlib import Path

from llmbench.cancel import CancelToken
from llmbench.gui import BenchApp, SettingsStore


def main() -> int:
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.1.189:11072/v1"
    model = sys.argv[2] if len(sys.argv) > 2 else "iq3_s"
    tmp = tempfile.mkdtemp()
    app = BenchApp(SettingsStore(Path(tmp) / "settings.json"))
    app.var_base.set(base_url)
    app.var_model.set(model)
    app.profile_vars["short"].set(True)
    app.var_repeats.set("1")
    app.var_warmup.set("0")
    app.var_max_tokens.set("128")
    app.category_vars["math"].set(True)
    app.category_vars["code"].set(True)
    app.var_out.set("results")
    app.var_label.set("gui-real")

    values = app._current_values()
    print("base_url =", values["base_url"], "| model =", values["model"])
    print("validate:", repr(app._validate(values)))
    app.cancel_token = CancelToken()
    app._worker_run(values)

    done = None
    while True:
        try:
            kind, data = app.queue.get_nowait()
        except queue.Empty:
            break
        if kind == "log":
            print("LOG", data)
        elif kind == "error":
            print("ERROR", data)
        elif kind == "done":
            done = data
    if done:
        app._finish(*done)
        print("HEADLINE:", app.headline.cget("text"))
        print("SPEED ROWS:", len(app.speed_tree.get_children()),
              "ACC ROWS:", len(app.acc_tree.get_children()))
        print("SAVED:", [str(p) for p in app.saved_paths])
    app.destroy()
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
