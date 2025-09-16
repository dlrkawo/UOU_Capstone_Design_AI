import os
import time
import concurrent.futures
from typing import List, Dict, Any, Tuple, Optional

# google genai SDK (gemini)
try:
	from google import genai
	exists_genai = True
except Exception:
	exists_genai = False


def get_attr_safe(obj: Any, name: str, default: Optional[Any] = None) -> Any:
	if isinstance(obj, dict):
		return obj.get(name, default)
	return getattr(obj, name, default)


class GeminiClient:
	"""간단 래퍼: 파일 업로드와 generate_content 호출을 지원."""

	def __init__(self, api_key: str, model: str):
		if not exists_genai:
			raise RuntimeError("google-genai 패키지가 필요합니다. requirements 설치 후 재시도하세요.")
		self.api_key = api_key
		self.model = model
		self.client = genai.Client(api_key=api_key)

	def upload_pdf(self, pdf_path: str) -> Any:
		if not os.path.exists(pdf_path):
			raise FileNotFoundError(pdf_path)
		# 최신 SDK는 mime_type 인자를 받지 않음
		return self.client.files.upload(file=pdf_path)

	def wait_for_active(self, file_obj: Any, timeout_sec: int = 180) -> Any:
		start = time.time()
		name = get_attr_safe(file_obj, "name")
		while True:
			refreshed = self.client.files.get(name=name)
			state = get_attr_safe(refreshed, "state")
			if state == "ACTIVE":
				return refreshed
			if time.time() - start > timeout_sec:
				raise TimeoutError("Gemini file activation timeout")
			time.sleep(1.0)

	def ask(self, system_instructions: str, user_text: str, file_uri: str) -> str:
		resp = self.client.models.generate_content(
			model=self.model,
			config={"system_instruction": system_instructions},
			contents=[
				{
					"role": "user",
					"parts": [
						{"file_data": {"file_uri": file_uri}},
						{"text": user_text},
					],
				}
			],
		)
		return getattr(resp, "text", "") or ""


def parallel_map(tasks: List[Tuple[str, str, str]], worker_fn, max_workers: int = 4) -> List[Tuple[str, str]]:
	"""tasks: list of (task_id, section_prompt, section_key). Returns [(section_key, text)]."""
	results: List[Tuple[str, str]] = []
	with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
		future_to_key = {executor.submit(worker_fn, t): t[2] for t in tasks}
		for future in concurrent.futures.as_completed(future_to_key):
			key = future_to_key[future]
			try:
				text = future.result()
				results.append((key, text))
			except Exception as e:
				results.append((key, f"__ERROR__: {e}"))
	return results


def md_join(title: str, parts: List[Tuple[str, str]]) -> str:
	parts_sorted = sorted(parts, key=lambda x: x[0])
	sections = [f"## {k}\n\n{v}" for k, v in parts_sorted]
	return f"# {title}\n\n" + "\n\n".join(sections)
