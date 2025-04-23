import torch

from isaaclab.managers import ActionManager
from isaaclab.envs import ManagerBasedRLEnv
from collections.abc import Sequence


class CustomActionManager(ActionManager):
    """Custom action manager for the UR10 environment."""

    def __init__(self, cfg: object, env):
        super().__init__(cfg, env)
        self._prev_2_action = torch.zeros_like(self._action)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        # super().reset(env_ids)
        self._prev_2_action[env_ids] = 0.0
        return super().reset(env_ids)

    def process_action(self, action: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function should be called once per environment step.

        Args:
            action: The actions to process.
        """
        # check if action dimension is valid
        if self.total_action_dim != action.shape[1]:
            raise ValueError(f"Invalid action shape, expected: {self.total_action_dim}, received: {action.shape[1]}.")
        # store the input actions
        self._prev_2_action[:] = self._prev_action
        self._prev_action[:] = self._action
        self._action[:] = action.to(self.device)

        # split the actions and apply to each tensor
        idx = 0
        for term in self._terms.values():
            term_actions = action[:, idx : idx + term.action_dim]
            term.process_actions(term_actions)
            idx += term.action_dim

    @property
    def prev_2_action(self):
        """Get the previous action."""
        return self._prev_2_action
    
class CustomManagerBasedRLEnv(ManagerBasedRLEnv):
    """Custom manager-based RL environment for the UR10 environment."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.action_manager = CustomActionManager(cfg.actions, self)

