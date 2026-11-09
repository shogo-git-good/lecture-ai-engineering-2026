
"""第1回発展課題：Gemini APIの応答性能比較実験。"""

import argparse
import csv
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import errors, types

from app.config import get_api_key

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results.csv"
RESPONSES = ROOT / "responses"

# 質問は知識問題と推論問題の2種類
QUESTIONS = {
    "knowledge": (
        "LoRA（Low-Rank Adaptation）とは何ですか？"
        "説明してください。"
    ),
    "reasoning": (
        "ある工場では製品AとBを生産する。"
        "Aは1個あたり利益300円、作業時間2時間、"
        "Bは1個あたり利益400円、作業時間3時間かかる。"
        "合計作業時間は12時間以内であり、"
        "製品数は0以上の整数とする。"
        "利益が最大になる生産個数を求め、"
        "理由を説明してください。"
    ),
}

# 目標出力トークン数
TARGETS = [100, 300, 500]

# CSVに保存するデータ
FIELDS = [
    "model",
    "question",
    "target",
    "trial",
    "timestamp",
    "elapsed",
    "input_tokens",
    "output_tokens",
    "thinking_tokens",
    "total_tokens",
    "status",
    "error_code",
    "response_file",
]


def build_prompt(question, target):
    """目標トークン数を指定した質問文を作成する。"""
    return (
        f"{QUESTIONS[question]}\n\n"
        f"日本語で約{target}トークンを目安に回答してください。"
        "回答の詳しさを調整し、簡潔さと正確さを両立してください。"
        "同じ内容の不必要な繰り返しは避けてください。"
    )


def completed_trials():
    """すでに成功した実験を確認する。"""
    if not RESULTS.exists():
        return set()

    with RESULTS.open("r", encoding="utf-8", newline="") as f:
        return {
            (
                row["model"],
                row["question"],
                int(row["target"]),
                int(row["trial"]),
            )
            for row in csv.DictReader(f)
            if row["status"] == "success"
        }


def save_row(row):
    """測定結果をCSVに追記する。"""
    new_file = not RESULTS.exists()

    with RESULTS.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)

        if new_file:
            writer.writeheader()

        writer.writerow(row)


def measure(client, model, question, target, trial):
    """Gemini APIを実行して応答時間・トークン数を測定する。"""

    row = {
        "model": model,
        "question": question,
        "target": target,
        "trial": trial,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed": "",
        "input_tokens": "",
        "output_tokens": "",
        "thinking_tokens": "",
        "total_tokens": "",
        "status": "success",
        "error_code": "",
        "response_file": "",
    }

    prompt = build_prompt(question, target)

    # 応答時間の測定開始
    start = time.perf_counter()

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
            ),
        )

        # 応答時間の測定終了
        row["elapsed"] = round(
            time.perf_counter() - start, 4
        )

        # トークン数を取得
        usage = response.usage_metadata

        if usage:
            row["input_tokens"] = usage.prompt_token_count
            row["output_tokens"] = usage.candidates_token_count
            row["thinking_tokens"] = usage.thoughts_token_count
            row["total_tokens"] = usage.total_token_count

        # 回答全文をテキストファイルに保存
        filename = (
            f"{model}_{question}_{target}_{trial}.txt"
        )

        path = RESPONSES / filename

        path.write_text(
            response.text or "",
            encoding="utf-8"
        )

        row["response_file"] = str(
            path.relative_to(ROOT)
        )

    except errors.APIError as exc:
        row["elapsed"] = round(
            time.perf_counter() - start, 4
        )

        row["status"] = "error"
        row["error_code"] = exc.code

    # 1回ごとにCSVへ保存
    save_row(row)

    return row


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--interval", type=float, default=15)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.repeat < 1 or args.interval < 0:
        parser.error(
            "repeatは1以上、intervalは0以上にしてください"
        )

    if args.limit is not None and args.limit < 1:
        parser.error("limitは1以上にしてください")

    # 実験条件を自動生成
    tasks = [
        (question, target, trial)
        for question in QUESTIONS
        for target in TARGETS
        for trial in range(1, args.repeat + 1)
    ]

    # 実行順序をランダム化
    random.Random(42).shuffle(tasks)

    # 実行済みの実験を除外
    done = completed_trials()

    tasks = [
        task for task in tasks
        if (args.model, *task) not in done
    ]

    # 予備実験では実行回数を制限できる
    if args.limit is not None:
        tasks = tasks[:args.limit]

    print(f"モデル: {args.model}")
    print(f"実行予定: {len(tasks)}回")
    print(f"待機時間: {args.interval}秒")

    # APIを呼ばず実行予定だけ確認
    if args.dry_run:
        for task in tasks:
            print(task)
        return

    if not tasks:
        print("未実行の条件はありません")
        return

    # 保存先を作成
    RESPONSES.mkdir(parents=True, exist_ok=True)

    # Geminiクライアント
    client = genai.Client(api_key=get_api_key())

    for index, (question, target, trial) in enumerate(
        tasks, 1
    ):
        print(
            f"[{index}/{len(tasks)}] "
            f"{question}, {target}トークン, "
            f"{trial}回目",
            flush=True,
        )

        result = measure(
            client, model=args.model,
            question=question,
            target=target,
            trial=trial,
        )

        print(
            f"状態: {result['status']} / "
            f"応答時間: {result['elapsed']}秒 / "
            f"出力: {result['output_tokens']}トークン",
            flush=True,
        )

        # APIエラーが出たら停止
        if result["status"] != "success":
            print("APIエラーが発生したため停止します。")
            break

        # 次のAPI呼び出しまで15秒待機
        if index < len(tasks):
            time.sleep(args.interval)

    print(f"結果保存先: {RESULTS}")


if __name__ == "__main__":
    main()
