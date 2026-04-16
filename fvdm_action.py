from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple

class ActionType(Enum):
    MOVE = "move"
    STAY = "stay"
    COMBAT = "combat"
    TRADE = "trade"
    MATE = "mate"
    CREDIT = "credit"
    TAGGING = "tagging"

@dataclass(frozen=True)
class ActionCandidate:
    action_type: ActionType
    target_cell_coords: Optional[Tuple[int, int]] = None
    target_agent_id: Optional[int] = None
    tag_index: Optional[int] = None
