import argparse
import sys
import logging
from .agents import Coordinator


def build_command(pdf_path: str, title: str, api_key: str, model: str, max_feedback_loops: int, length_level: int, difficulty_level: int, formatting: str) -> int:
	logging.info("CLI: build 시작")
	logging.info(f"입력 PDF: {pdf_path}")
	logging.info(f"모델: {model}, 최대 피드백 루프: {max_feedback_loops}, 길이 수준: {length_level}, 난이도: {difficulty_level}, 포맷팅: {formatting}")
	coordinator = Coordinator(
		pdf_path=pdf_path,
		title=title,
		api_key=api_key,
		model=model,
		max_feedback_loops=max_feedback_loops,
		length_level=length_level,
		difficulty_level=difficulty_level,
		formatting=formatting,
	)
	output_path = coordinator.run_build()
	print(output_path)
	return 0


def parse_args(argv=None):
	parser = argparse.ArgumentParser(prog="cwd_v2", description="CWD V2 강의 자료 생성기")
	subparsers = parser.add_subparsers(dest="command", required=True)

	build_parser = subparsers.add_parser("build", help="PDF로부터 강의 자료 생성")
	build_parser.add_argument("pdf_path", type=str)
	build_parser.add_argument("--title", type=str, required=True)
	build_parser.add_argument("--api-key", type=str, required=True)
	build_parser.add_argument("--model", type=str, default="gemini-2.5-flash")
	build_parser.add_argument("--max-feedback-loops", type=int, default=2)
	build_parser.add_argument("--length-level", type=int, default=5, help="출력 길이 수준 (0=최소~10=최대)")
	build_parser.add_argument("--difficulty-level", type=int, default=3, help="난이도 (1=매우 쉬움 ~ 5=매우 심화)")
	build_parser.add_argument("--formatting", type=str, default="light", help="포맷팅 모드 (none|light)")

	return parser.parse_args(argv)


def main(argv=None):
	# 실시간 로그 설정 (stdout)
	logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
	args = parse_args(argv)
	if args.command == "build":
		return build_command(
			pdf_path=args.pdf_path,
			title=args.title,
			api_key=args.api_key,
			model=args.model,
			max_feedback_loops=args.max_feedback_loops,
			length_level=max(0, min(10, int(args.length_level))),
			difficulty_level=max(1, min(5, int(args.difficulty_level))),
			formatting=(str(args.formatting).lower() if str(args.formatting).lower() in ("none", "light") else "light"),
		)
	return 1


if __name__ == "__main__":
	sys.exit(main())
