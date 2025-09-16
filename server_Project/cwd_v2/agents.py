from __future__ import annotations

import os
import logging
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from .utils import GeminiClient, parallel_map, md_join, get_attr_safe


@dataclass
class SectionPlan:
	key: str  # e.g. "2.1 주요 개념"
	prompt: str


class PDFStructureAnalyst:
	"""PDF 구조 분석가: Gemini가 PDF를 직접 읽어 목차/핵심 개념 후보를 추출."""

	SYSTEM = (
		"[역할] PDF 구조 분석가. 원본 PDF를 직접 읽고 강의용 논리 단위를 설계한다.\n"
		"[출력 형식] 줄바꿈으로 구분된 항목 리스트만 출력. 불필요한 서론/해설 금지.\n"
		"[요건] 최대 8개. 각 항목은 '번호. 제목' 형식(예: '2.2. 오차 역전파').\n"
		"[원칙] (1) 원문 근거 없는 가설/목차 추가 금지, (2) 중복/과세분화 지양, (3) 강의 흐름상 자연스러운 순서로 정렬,\n"
		"(4) 가능한 한 PDF의 실제 구조(챕터/절/소절)를 반영."
	)

	def __init__(self, client: GeminiClient, file_uri: str):
		self.client = client
		self.file_uri = file_uri

	def analyze(self) -> List[SectionPlan]:
		logging.info("[PDF 구조 분석가] 작업 시작: 석션/핵심 개념 추출")
		user = (
			"원본 PDF를 기반으로 강의에 사용할 작업 단위를 추출하라.\n"
			"형식: 줄바꿈으로 구분한 항목 목록. 각 항목은 간결한 제목."
		)
		text = self.client.ask(self.SYSTEM, user, self.file_uri)
		lines = [l.strip() for l in text.splitlines() if l.strip()]
		if not lines:
			logging.warning("[PDF 구조 분석가] 비어있는 결과 → 기본 섹션 사용")
			lines = ["1. 개요", "2. 핵심 개념", "3. 요약"]
		plans = [SectionPlan(key=line, prompt=f"다음 항목에 대한 강의 텍스트를 작성하라: {line}") for line in lines]
		logging.info(f"[PDF 구조 분석가] 추출된 작업 단위 수: {len(plans)}")
		return plans


class ContentCreatorWorker:
	"""콘텐츠 생성가: 전체 PDF를 참조하여 특정 섹션만 생성."""

	SYSTEM = (
		"[역할] 강의 자료 작성자. 요청된 섹션만 다룬다.\n"
		"[원칙] (1) 원문 기반 사실만 사용(환각 금지), (2) 핵심 개념을 먼저 요약 후 예시/설명 확장,\n"
		"(3) 불확실하거나 모호한 내용은 그대로 두고 '확인 필요' 질문을 덧붙인다,\n"
		"(4) 직접 인용은 사용하지 말고, 필요 시 개념만 요약해 설명한다.\n"
		"[형식] 마크다운. 소제목/목록/강조를 활용해 가독성 강화. 다른 섹션 내용은 참조만."
	)

	def __init__(self, client: GeminiClient, file_uri: str, section: SectionPlan, length_level: int = 5, difficulty_level: int = 3):
		self.client = client
		self.file_uri = file_uri
		self.section = section
		self.length_level = max(0, min(10, int(length_level)))
		self.difficulty_level = max(1, min(5, int(difficulty_level)))

	def generate(self) -> str:
		logging.info(f"[콘텐츠 생성가] 수임: '{self.section.key}' 생성")
		# 길이 제어 힌트 구성
		length_hint = (
			"아주 짧게" if self.length_level <= 1 else
			"짧게" if self.length_level <= 3 else
			"보통 길이" if self.length_level <= 6 else
			"자세히" if self.length_level <= 8 else
			"매우 자세히"
		)
		bullet_density = (
			"핵심 bullet 1~2개, 문장 2~3줄" if self.length_level <= 2 else
			"bullet 3~4개, 문장 4~6줄" if self.length_level <= 5 else
			"bullet 5~7개, 소제목 포함" if self.length_level <= 8 else
			"bullet 8개+, 소제목/예시/수식 가능"
		)
		# 난이도별 예시/설명/수식 가이드
		difficulty_label = {1: "매우 쉬움", 2: "쉬움", 3: "보통", 4: "심화", 5: "매우 심화"}[self.difficulty_level]
		examples_guide = (
			"아주 쉬운 비유/일상 예시 1개, 수식 없음" if self.difficulty_level == 1 else
			"쉬운 예시 1~2개, 간단한 표/도식 가능" if self.difficulty_level == 2 else
			"표준 예시 2개, 간단한 공식/정의 포함" if self.difficulty_level == 3 else
			"도출 과정/반례/엣지케이스 포함, 수식/의사코드 가능" if self.difficulty_level == 4 else
			"깊은 통찰/복잡 사례/증명 스케치, 수식/의사코드/복잡 비교 포함"
		)
		user = (
			f"섹션 제목: {self.section.key}\n"
			f"지시사항: {self.section.prompt}\n"
			f"길이 가이드: {length_hint}. ({bullet_density})\n"
			f"난이도: {self.difficulty_level} ({difficulty_label}). 예시/설명 가이드: {examples_guide}.\n"
			"형식: 마크다운. 소단락, 목록, 핵심 포인트 강조."
		)
		text = self.client.ask(self.SYSTEM, user, self.file_uri)
		logging.info(f"[콘텐츠 생성가] 완료: '{self.section.key}'")
		return text


class MarkdownFormatterAgent:
	"""마크다운 포맷터: 여러 조각을 일관된 톤으로 정리."""
	SYSTEM = (
		"[역할] 기술 편집자. 여러 섹션을 하나의 문서로 자연스럽게 연결한다.\n"
		"[원칙] (1) 내용 왜곡/의미 변경 금지, (2) 인용/페이지 표기 보존, (3) 중복 제거 및 용어 일관성 유지,\n"
		"(4) 섹션 간 전환문 추가 가능하되 사실에 근거하지 않은 새로운 주장 금지.\n"
		"[형식] 일관된 마크다운 레벨과 스타일 유지."
	)

	def __init__(self, client: GeminiClient, file_uri: str):
		self.client = client
		self.file_uri = file_uri

	def format(self, title: str, parts: List[Tuple[str, str]]) -> str:
		logging.info("[마크다운 포맷터] 취합/포맷팅 시작")
		base = md_join(title, parts)
		user = (
			"다음 마크다운 초안을 자연스럽게 연결하고 일관된 스타일로 다듬어라.\n"
			"가능하면 섹션간 연결 문장을 추가하되 내용 왜곡은 금지.\n\n"
			f"초안:\n{base}"
		)
		text = self.client.ask(self.SYSTEM, user, self.file_uri)
		logging.info("[마크다운 포맷터] 포맷팅 완료")
		return text


class FeedbackFactCheckerAgent:
	"""피드백 및 팩트체커: 초안과 원본 PDF를 교차 검증."""
	SYSTEM = (
		"[역할] 품질 검증 및 환각 방지 전문가. 초안과 원본 PDF를 교차 검증한다.\n"
		"[판정 기준] (1) 원문과의 내용 일치/불일치, (2) 인용/페이지 근거 유무, (3) 환각 수.\n"
		"[출력 제한] 반드시 JSON 한 덩어리만 출력. 다른 텍스트 금지.\n"
		"[형식] {decision: 'pass'|'revise'|'abort', overlap_ratio: 0..1, hallucination_count: int, reasons: string[], targets: [{key: string, notes: string}]}\n"
		"[타겟 규칙] targets.key 는 제공된 섹션 키 목록(정확 일치) 중에서만 선택. 최소 필요 범위만 선정."
	)

	def __init__(self, client: GeminiClient, file_uri: str):
		self.client = client
		self.file_uri = file_uri

	def _decide(self, payload: Dict[str, Any]) -> Dict[str, Any]:
		# 완화된 임계치
		overlap = float(payload.get("overlap_ratio", 0.0) or 0.0)
		hcount = int(payload.get("hallucination_count", 0) or 0)
		decision = str(payload.get("decision", "")).lower()
		if decision in ("pass", "ok"):
			return {"status": "ok", "metrics": {"overlap_ratio": overlap, "hallucination_count": hcount}}
		if decision == "abort" or hcount >= 3:
			return {"status": "abort", "metrics": {"overlap_ratio": overlap, "hallucination_count": hcount}}
		# 완화 기준: 일부 포함(>=0.25) 이고 환각 적음(<=1) → ok
		if overlap >= 0.25 and hcount <= 1:
			return {"status": "ok", "metrics": {"overlap_ratio": overlap, "hallucination_count": hcount}}
		return {"status": "needs_revision", "metrics": {"overlap_ratio": overlap, "hallucination_count": hcount}}

	def verify(self, draft_md: str, section_keys: List[str]) -> Dict[str, Any]:
		logging.info("[피드백/팩트체커] 검증 시작")
		user = (
			"다음 초안의 사실 검증을 수행하라. 원문과의 일치 여부, 인용/페이지 근거, 환각 여부를 평가하라.\n"
			"초안이 원문 일부를 포함하고 명백한 오류가 없다면 pass. 일부 불일치/불명확이 있으면 revise.\n"
			"형식: JSON 하나. {decision, overlap_ratio, hallucination_count, reasons, targets}\n"
			"targets: [{key, notes}]이며 key 는 반드시 아래 섹션 키 중 하나여야 한다. 최소 필요 섹션만 지정.\n\n"
			f"섹션 키 목록: {section_keys}\n\n"
			f"초안:\n{draft_md}"
		)
		text = self.client.ask(self.SYSTEM, user, self.file_uri)
		# JSON 파싱 시도
		payload: Dict[str, Any]
		try:
			payload = json.loads(text)
		except Exception:
			# 폴백: 키워드 기반(완화)
			lower = text.lower()
			if "abort" in lower:
				logging.error("[피드백/팩트체커] 치명적 환각 다수 → ABORT (fallback)")
				return {"status": "abort", "raw": text, "targets": []}
			if "pass" in lower or "승인" in text or "ok" in lower:
				logging.info("[피드백/팩트체커] 승인 OK (fallback)")
				return {"status": "ok", "raw": text, "targets": []}
			# 애매하면 수정 필요로 분류
			logging.warning("[피드백/팩트체커] 수정 필요 (fallback)")
			return {"status": "needs_revision", "raw": text, "targets": []}

		decision = self._decide(payload)
		status = decision["status"]

		# targets 정규화: 제공되지 않으면 reasons 기반으로 유추
		targets = payload.get("targets") or []
		if not targets:
			reasons = payload.get("reasons") or []
			if isinstance(reasons, str):
				reasons = [reasons]
			matched: List[Dict[str, Any]] = []
			for sk in section_keys:
				for r in reasons:
					if isinstance(r, str) and sk in r:
						matched.append({"key": sk, "notes": r})
						break
			targets = matched

		if status == "abort":
			logging.error("[피드백/팩트체커] 치명적 환각 판정")
		elif status == "needs_revision":
			logging.warning("[피드백/팩트체커] 수정 필요")
		else:
			logging.info("[피드백/팩트체커] 승인 OK")
		return {"status": status, "raw": text, "targets": targets, **decision}


class Delegator:
	"""작업 분해, 병렬 처리, 재시도 관리."""

	def __init__(self, client: GeminiClient, file_uri: str, length_level: int = 5, difficulty_level: int = 3):
		self.client = client
		self.file_uri = file_uri
		self.length_level = max(0, min(10, int(length_level)))
		self.difficulty_level = max(1, min(5, int(difficulty_level)))

	def plan(self) -> List[SectionPlan]:
		logging.info("[델리게이터] 작업 계획 수립 시작")
		analyst = PDFStructureAnalyst(self.client, self.file_uri)
		plans = analyst.analyze()
		logging.info("[델리게이터] 작업 계획 수립 완료")
		return plans

	def run_parallel_generation(self, plans: List[SectionPlan], max_workers: int = 4) -> List[Tuple[str, str]]:
		logging.info(f"[델리게이터] 병렬 생성 시작 (workers={max_workers}, tasks={len(plans)})")
		def do_one(task: Tuple[str, str, str]) -> str:
			_, section_prompt, key = task
			worker = ContentCreatorWorker(self.client, self.file_uri, SectionPlan(key=key, prompt=section_prompt), length_level=self.length_level, difficulty_level=self.difficulty_level)
			try:
				return worker.generate()
			except Exception as e:
				logging.warning(f"[델리게이터] 1차 실패 재시도: {key}: {e}")
				return worker.generate()

		tasks = [(str(i + 1), p.prompt, p.key) for i, p in enumerate(plans)]
		parts = parallel_map(tasks, do_one, max_workers=max_workers)
		logging.info("[델리게이터] 병렬 생성 완료")
		return parts

	def format_markdown(self, title: str, parts: List[Tuple[str, str]]) -> str:
		return MarkdownFormatterAgent(self.client, self.file_uri).format(title, parts)

	def revise_parts(self, parts: List[Tuple[str, str]], feedback_raw: str) -> List[Tuple[str, str]]:
		logging.info("[델리게이터] 피드백 반영 재생성 시작")
		revised: List[Tuple[str, str]] = []
		for key, text in parts:
			user = (
				f"섹션 '{key}' 내용을 다음 피드백을 반영해 수정하라.\n"
				f"피드백:\n{feedback_raw}\n\n현재 섹션 텍스트:\n{text}"
			)
			section = SectionPlan(key=key, prompt=user)
			worker = ContentCreatorWorker(self.client, self.file_uri, section, length_level=self.length_level, difficulty_level=self.difficulty_level)
			try:
				revised_text = worker.generate()
			except Exception as e:
				logging.warning(f"[델리게이터] 재생성 실패 유지: {key}: {e}")
				revised_text = text
			revised.append((key, revised_text))
		logging.info("[델리게이터] 피드백 반영 재생성 완료")
		return revised

	def revise_selected_parts(self, parts: List[Tuple[str, str]], feedback: Dict[str, Any]) -> List[Tuple[str, str]]:
		logging.info("[델리게이터] 선택적 재생성 시작")
		targets = feedback.get("targets") or []
		target_keys: List[str] = []
		notes_map: Dict[str, str] = {}
		for t in targets:
			if isinstance(t, dict):
				k = str(t.get("key") or t.get("section") or t.get("name") or "").strip()
				if not k:
					continue
				target_keys.append(k)
				notes = str(t.get("notes") or "").strip()
				if notes:
					notes_map[k] = notes
			elif isinstance(t, str):
				target_keys.append(t)

		if not target_keys:
			logging.warning("[델리게이터] 대상 섹션이 비어있음 → 전체 재생성으로 폴백")
			return self.revise_parts(parts, feedback.get("raw", ""))

		revised: List[Tuple[str, str]] = []
		for key, text in parts:
			if key in target_keys:
				notes = notes_map.get(key, "")
				user = (
					f"섹션 '{key}' 내용을 다음 피드백을 반영해 수정하라.\n"
					f"피드백 원본:\n{feedback.get('raw', '')}\n"
					f"해당 섹션 메모: {notes}\n\n"
					f"현재 섹션 텍스트:\n{text}"
				)
				section = SectionPlan(key=key, prompt=user)
				worker = ContentCreatorWorker(self.client, self.file_uri, section, length_level=self.length_level, difficulty_level=self.difficulty_level)
				try:
					revised_text = worker.generate()
				except Exception as e:
					logging.warning(f"[델리게이터] 선택 재생성 실패 유지: {key}: {e}")
					revised_text = text
				revised.append((key, revised_text))
			else:
				revised.append((key, text))
		logging.info("[델리게이터] 선택적 재생성 완료")
		return revised


class Coordinator:
	"""프로젝트 총괄 및 최종 품질 책임자: 피드백 루프 관리."""

	def __init__(self, pdf_path: str, title: str, api_key: str, model: str, max_feedback_loops: int = 2, length_level: int = 5, difficulty_level: int = 3, formatting: str = "light"):
		self.pdf_path = pdf_path
		self.title = title
		resolved_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
		self.api_key = resolved_key
		self.model = model
		self.max_feedback_loops = max(0, int(max_feedback_loops))
		self.length_level = max(0, min(10, int(length_level)))
		self.difficulty_level = max(1, min(5, int(difficulty_level)))
		self.formatting = (str(formatting).lower() if str(formatting).lower() in ("none", "light") else "light")

	def run_build(self) -> str:
		logging.info("[코디네이터] 빌드 시작")
		client = GeminiClient(self.api_key, self.model)
		logging.info("[코디네이터] PDF 업로드")
		file_obj = client.upload_pdf(self.pdf_path)
		active = client.wait_for_active(file_obj)
		file_uri = get_attr_safe(active, "uri") or get_attr_safe(active, "file_uri")
		logging.info("[코디네이터] 업로드 완료, 작업 지시 → 델리게이터")

		delegator = Delegator(client, file_uri, length_level=self.length_level, difficulty_level=self.difficulty_level)
		plans = delegator.plan()
		parts = delegator.run_parallel_generation(plans, max_workers=min(6, max(2, len(plans))))
		# 포맷팅 모드: none → 단순 병합, light → 포맷터로 폴리싱
		if self.formatting == "none":
			md = md_join(self.title, parts)
		else:
			md = delegator.format_markdown(self.title, parts)

		feedback_agent = FeedbackFactCheckerAgent(client, file_uri)
		loops = 0
		while True:
			section_keys = [k for k, _ in parts]
			feedback = feedback_agent.verify(md, section_keys)
			if feedback.get("status") == "abort":
				logging.warning("[코디네이터] 치명적 환각 다수 → 전체 엄격 재작업으로 회귀")
				strict_msg = (
					"STRICT_REWRITE: 원문 직접 근거(페이지/직접 인용) 없는 내용은 모두 제거/비워둘 것. "
					"추론/외삽 금지. 불확실 부분은 '확인 필요'로 표시하고 질문 제시."
				)
				forced_targets = [{"key": k, "notes": "치명적 환각: 근거 있는 내용만 유지/복원"} for k, _ in parts]
				feedback = {"raw": (feedback.get("raw", "") + "\n" + strict_msg), "targets": forced_targets}
			if feedback.get("status") == "ok":
				logging.info("[코디네이터] 품질 승인 완료")
				break
			loops += 1
			logging.info(f"[코디네이터] 피드백 루프 {loops}/{self.max_feedback_loops}")
			if loops > self.max_feedback_loops:
				logging.warning("[코디네이터] 최대 피드백 루프 초과 → 현재 결과물 확정")
				break
			if feedback.get("targets"):
				parts = delegator.revise_selected_parts(parts, feedback)
			else:
				parts = delegator.revise_parts(parts, feedback.get("raw", ""))
			if self.formatting == "none":
				md = md_join(self.title, parts)
			else:
				md = delegator.format_markdown(self.title, parts)

		out_dir = os.path.join(os.path.dirname(self.pdf_path), "outputs")
		os.makedirs(out_dir, exist_ok=True)
		out_path = os.path.join(out_dir, f"{self.title}.md")
		with open(out_path, "w", encoding="utf-8") as f:
			f.write(md)
		logging.info(f"[코디네이터] 결과 저장: {out_path}")
		return out_path
