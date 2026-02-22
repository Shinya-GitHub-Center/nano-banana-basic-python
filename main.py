import os
import sys
import asyncio
from datetime import datetime
from pathlib import Path
import base64
from google import genai

# ====================================
# Configuration - ここで設定を変更
# ====================================

# 画像保存先ディレクトリ (uv runが.envを自動的に読み込みます)
OUTPUT_DIR = os.getenv("IMAGE_OUTPUT_DIR")

# 一回で生成する画像数
IMAGE_COUNT = 1

# 画像アスペクト比（サポート: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9）
IMAGE_ASPECT_RATIO = "3:2"

# 画像サイズ（サポート: 1K, 2K, 4K）
IMAGE_SIZE = "1K"

# 生成パラメータ
TEMPERATURE = 0.2
TOP_P = 0.95

# システムインストラクション
SYSTEM_INSTRUCTION = "You are a professional image creator. Generate high-quality images based on the user's request."

# Gemini API Key (環境変数から取得)
API_KEY = os.getenv("GEMINI_API_KEY")


# ====================================
# Main Logic
# ====================================


def validate_args():
    """引数のバリデーション"""
    if not API_KEY:
        print(
            "❌ Error: GEMINI_API_KEY environment variable is not set.", file=sys.stderr
        )
        print("", file=sys.stderr)
        print("Usage:", file=sys.stderr)
        print("  Set GEMINI_API_KEY in .env file", file=sys.stderr)
        print('  uv run main.py "your prompt here"', file=sys.stderr)
        sys.exit(1)

    if not OUTPUT_DIR:
        print(
            "❌ Error: IMAGE_OUTPUT_DIR environment variable is not set.",
            file=sys.stderr,
        )
        print("", file=sys.stderr)
        print("Usage:", file=sys.stderr)
        print("  Set IMAGE_OUTPUT_DIR in .env file", file=sys.stderr)
        sys.exit(1)

    if len(sys.argv) < 2:
        print("❌ Error: Image prompt is required.", file=sys.stderr)
        print("", file=sys.stderr)
        print("Usage:", file=sys.stderr)
        print('  uv run main.py "your prompt here"', file=sys.stderr)
        print("", file=sys.stderr)
        print("Example:", file=sys.stderr)
        print(
            '  uv run main.py "futuristic city at sunset with neon lights"',
            file=sys.stderr,
        )
        sys.exit(1)


def ensure_output_directory():
    """出力ディレクトリを作成"""
    output_path = Path(OUTPUT_DIR)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created output directory: {OUTPUT_DIR}")


def create_timestamped_directory():
    """タイムスタンプ付きのサブディレクトリを作成"""
    timestamp = datetime.now().isoformat().replace(":", "-").replace(".", "-")[:19]
    session_dir = Path(OUTPUT_DIR) / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


async def generate_image(prompt, index, output_path):
    """Gemini API を使って画像を生成"""
    try:
        client = genai.Client(api_key=API_KEY)

        print(f"   Processing image {index}...")

        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                response_modalities=["IMAGE"],
                image_config=genai.types.ImageConfig(
                    aspect_ratio=IMAGE_ASPECT_RATIO,
                    image_size=IMAGE_SIZE,
                ),
            ),
        )

        # レスポンスから画像データを取得
        for part in response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data:
                image_data = part.inline_data.data

                # データがすでにbytesの場合はそのまま使用、strの場合はBase64デコード
                if isinstance(image_data, bytes):
                    buffer = image_data
                else:
                    buffer = base64.b64decode(image_data)

                filename = f"image-{str(index).zfill(2)}.png"
                filepath = output_path / filename

                filepath.write_bytes(buffer)
                print(f"   ✅ Generated: {filename} ({len(buffer)} bytes)")
                return str(filepath)

        raise Exception("No image data in response")
    except Exception as error:
        print(f"   ❌ Failed to generate image {index}: {error}", file=sys.stderr)
        return None


def save_prompt_file(session_dir, prompt):
    """プロンプトファイルを保存"""
    prompt_file = session_dir / "prompt.txt"
    metadata = [
        f"Prompt: {prompt}",
        f"Generated: {datetime.now().isoformat()}",
        f"Model: gemini-2.5-flash-image",
        f"Count: {IMAGE_COUNT}",
        f"Aspect Ratio: {IMAGE_ASPECT_RATIO}",
        f"Image Size: {IMAGE_SIZE}",
        f"Temperature: {TEMPERATURE}",
        f"Top P: {TOP_P}",
        f"Output Directory: {session_dir}",
    ]

    prompt_file.write_text("\n".join(metadata))
    print("📝 Saved metadata to: prompt.txt")


async def main():
    """メイン処理"""
    print("🎨 Gemini Blog Image Generator")
    print("================================\n")

    # バリデーション
    validate_args()

    prompt = sys.argv[1]

    print(f'📝 Prompt: "{prompt}"')
    print(f"🔢 Image Count: {IMAGE_COUNT}")
    print(f"📐 Aspect Ratio: {IMAGE_ASPECT_RATIO}")
    print(f"📐 Image Size: {IMAGE_SIZE}")
    print(f"📂 Output Directory: {OUTPUT_DIR}\n")

    # ディレクトリ準備
    ensure_output_directory()
    session_dir = create_timestamped_directory()
    print(f"📁 Session Directory: {session_dir}\n")

    # プロンプトファイルを保存
    save_prompt_file(session_dir, prompt)

    # 画像生成
    print("🚀 Starting image generation...\n")

    successful_images = []

    for i in range(1, IMAGE_COUNT + 1):
        result = await generate_image(prompt, i, session_dir)
        if result:
            successful_images.append(result)

        # API レート制限を考慮して少し待機（最後の画像以外）
        if i < IMAGE_COUNT:
            await asyncio.sleep(1)

    success_count = len(successful_images)

    print("\n================================")
    print("✨ Generation Complete!")
    print(f"   Success: {success_count}/{IMAGE_COUNT}")
    print(f"   Location: {session_dir}")
    print("================================\n")

    if success_count < IMAGE_COUNT:
        print("⚠️  Some images failed to generate. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as error:
        print(f"\n❌ Fatal Error: {error}", file=sys.stderr)
        sys.exit(1)
