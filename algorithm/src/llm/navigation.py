from dataclasses import dataclass
from typing import Optional, List
import re

from .memory import ImageMemory, ImageEntry
from .prompting import StructuredPrompt
from .logging import PromptLogger
from .llm_client import LLMClient
from .vlm import VLMClient


@dataclass
class Goal:
    text: Optional[str] = None
    image_index: Optional[int] = None
    image_description: Optional[str] = None
    prompt: Optional[StructuredPrompt] = None


class NavigationGPrompt:
    def __init__(self, image_memory: ImageMemory, logger: Optional[PromptLogger] = None) -> None:
        self.image_memory = image_memory
        self._logger = logger

    def build(
        self,
        scene_description: str,
        goal_text: Optional[str] = None,
        goal_image: Optional[object] = None,
        goal_image_description: Optional[str] = None,
    ) -> Goal:
        if not goal_text and goal_image is None:
            raise ValueError("Provide goal_text or goal_image.")

        goal = Goal(text=goal_text, image_description=goal_image_description)
        if goal_image is not None:
            goal.image_index = self.image_memory.add_image(goal_image, tag="goal")

        prompt = StructuredPrompt(
            system="You are a navigation assistant that summarizes the scene and goal.",
            logger=self._logger,
        )
        prompt.set_section("Scene", scene_description)
        if goal.text:
            prompt.set_section("GoalText", goal.text)
        if goal.image_index is not None:
            prompt.set_section("GoalImageIndex", str(goal.image_index))
        if goal.image_description:
            prompt.set_section("GoalImageDescription", goal.image_description)

        goal.prompt = prompt
        return goal


class SubgoalSelector:
    def __init__(
        self,
        llm_client: LLMClient,
        image_memory: ImageMemory,
        logger: Optional[PromptLogger] = None,
    ) -> None:
        self.llm_client = llm_client
        self.image_memory = image_memory
        self._logger = logger

    def _memory_lines(self, entries: List[ImageEntry]) -> List[str]:
        lines: List[str] = []
        for entry in entries:
            desc = entry.description or ""
            tag = f" tag={entry.tag}" if entry.tag else ""
            lines.append(f"[{entry.index}] {desc}{tag}".strip())
        return lines

    def build_prompt(
        self,
        goal: Goal,
        current_observation: str,
    ) -> StructuredPrompt:
        prompt = StructuredPrompt(
            system=(
                "Select a subgoal image index that helps reach the goal. "
                "If none are suitable, reply with 'none'. "
                "Output only an integer index or 'none'."
            ),
            logger=self._logger,
        )
        if goal.text:
            prompt.set_section("GoalText", goal.text)
        if goal.image_description:
            prompt.set_section("GoalImageDescription", goal.image_description)
        if goal.image_index is not None:
            prompt.set_section("GoalImageIndex", str(goal.image_index))
        prompt.set_section("CurrentObservation", current_observation)
        entries = self.image_memory.list_entries()
        prompt.set_section("MemoryImages", "\n".join(self._memory_lines(entries)))
        return prompt

    def select_subgoal(self, goal: Goal, current_observation: str) -> Optional[int]:
        if not self.image_memory.list_entries():
            return None
        prompt = self.build_prompt(goal, current_observation)
        response = self.llm_client.chat(prompt.to_messages(), temperature=0.0)
        text = response.strip().lower()
        if "none" in text:
            return None
        match = re.search(r"-?\d+", text)
        if not match:
            return None
        idx = int(match.group(0))
        return idx if self.image_memory.get_entry(idx) else None


class NavigationController:
    def __init__(
        self,
        llm_client: LLMClient,
        vlm_client: VLMClient,
        image_memory: ImageMemory,
        logger: Optional[PromptLogger] = None,
    ) -> None:
        self.llm_client = llm_client
        self.vlm_client = vlm_client
        self.image_memory = image_memory
        self.current_subgoal_index: Optional[int] = None
        self._logger = logger

    def decide_next_step(
        self,
        goal: Goal,
        current_observation: str,
        subgoal_reached: bool,
    ) -> str:
        prompt = StructuredPrompt(
            system=(
                "Decide the next step. Reply only with 'select_subgoal' or 'final_goal'."
            ),
            logger=self._logger,
        )
        if goal.text:
            prompt.set_section("GoalText", goal.text)
        if goal.image_description:
            prompt.set_section("GoalImageDescription", goal.image_description)
        prompt.set_section("CurrentObservation", current_observation)
        prompt.set_section("SubgoalReached", "yes" if subgoal_reached else "no")
        response = self.llm_client.chat(prompt.to_messages(), temperature=0.0)
        text = response.strip().lower()
        return "select_subgoal" if "select" in text else "final_goal"

    def choose_subgoal(self, goal: Goal, current_observation: str) -> Optional[int]:
        selector = SubgoalSelector(self.llm_client, self.image_memory, logger=self._logger)
        self.current_subgoal_index = selector.select_subgoal(goal, current_observation)
        return self.current_subgoal_index

    def check_subgoal_reached(self, current_image) -> bool:
        if self.current_subgoal_index is None:
            return False
        entry = self.image_memory.get_entry(self.current_subgoal_index)
        if not entry:
            return False
        return self.vlm_client.judge_relevance(
            current_image=current_image,
            target_image=entry.data_url,
            target_description=entry.description,
        )

    def check_final_goal_reached(self, current_image, goal: Goal) -> bool:
        if goal.image_index is not None:
            entry = self.image_memory.get_entry(goal.image_index)
            target_image = entry.data_url if entry else None
        else:
            target_image = None
        return self.vlm_client.judge_relevance(
            current_image=current_image,
            target_image=target_image,
            target_description=goal.image_description or goal.text,
        )
