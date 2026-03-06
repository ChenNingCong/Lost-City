"""
lost_cities_env_fast2.py
════════════════════════════════════════════════════════════════════
Drop-in accelerated replacement for lost_cities_env.py.

Architecture
────────────
• Game state lives in a C struct (LCGame) allocated via CFFI.
  Python never touches inner arrays — it only calls C functions.
• Deck shuffling stays in Python (random.Random) for RNG compatibility.
• opponent.act_batch() stays in Python (neural network calls).
• Everything else — valid actions, obs building, mask filling,
  apply_play, apply_draw, scoring — runs entirely in compiled C.

Public API is 100% backward-compatible with lost_cities_env.py.
"""

from __future__ import annotations

import enum
import math
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sys, os

# ── Load C engine ─────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import lc_core2_cffi
_ffi = lc_core2_cffi.ffi
_lib = lc_core2_cffi.lib

# Constants from C
TOTAL_ACTIONS = _lib.lc2_total_actions()   # 600
CARD_POOL     = _lib.lc2_card_pool()       # 50
OBS_SIZE      = _lib.lc2_obs_size()        # 301
NUM_COLORS    = _lib.lc2_num_colors()      # 5
_GAME_SIZE    = _lib.lc2_sizeof_game()

# ── Optional Gymnasium ─────────────────────────────────────────────────────
try:
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.vector.utils import batch_space
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    class _FakeSpaces:
        class Box:
            def __init__(self, low, high, dtype=None):
                self.low = np.asarray(low); self.high = np.asarray(high)
                self.dtype = dtype; self.shape = self.low.shape
        class Discrete:
            def __init__(self, n): self.n = int(n); self.shape = ()
        class MultiDiscrete:
            def __init__(self, nvec): self.nvec = nvec; self.shape = (len(nvec),)
    class _FakeGym:
        class Env:
            def reset(self, seed=None, options=None): pass
    gym = _FakeGym(); spaces = _FakeSpaces()
    def batch_space(space, n):
        if isinstance(space, spaces.Box):
            return spaces.Box(low=np.stack([space.low]*n),
                              high=np.stack([space.high]*n), dtype=space.dtype)
        elif isinstance(space, spaces.Discrete):
            return spaces.MultiDiscrete([int(space.n)]*n)
        else: raise NotImplementedError

Card = Tuple[int, int]

class ActionType(enum.Enum):
    Play    = 0
    DISCARD = 1

# ── Card value table (for deck generation) ───────────────────────────────
_CARD_VALUES = [0, 0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
_COLORS      = ["Blue", "Yellow", "White", "Green", "Red"]

# Pre-build the 60-card deck (unshuffled) as a uint8 numpy array
_DECK_TEMPLATE = np.array(
    [_lib.lc2_card_encode(ci, v)
     for ci in range(NUM_COLORS)
     for v in _CARD_VALUES],
    dtype=np.uint8)


# ═════════════════════════════════════════════════════════════════════════════
# 1.  C-BACKED GAME  (wraps LCGame struct)
# ═════════════════════════════════════════════════════════════════════════════

class LostCitiesGame:
    """
    Lost Cities game whose state lives entirely in a C struct.
    Python-facing API is identical to the original implementation.
    """

    COLORS             = _COLORS
    CARD_VALUES        = _CARD_VALUES
    UNIQUE_CARD_VALUE  = 10
    NUM_PLAYERS        = 2
    STARTING_HAND_SIZE = 8
    COLOR_TO_IDX       = {c: i for i, c in enumerate(_COLORS)}

    def __init__(self):
        # Allocate one LCGame struct — zero-initialised
        self._buf = _ffi.new("char[]", _GAME_SIZE)
        self._g   = _ffi.cast("LCGame *", self._buf)
        self._rng = random.Random()
        # scratch buffer for get_valid_actions
        self._act_buf = _ffi.new("int[128]")

    # ── Setup ────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = random.Random(seed)
        deck = _DECK_TEMPLATE.copy()
        # Shuffle using Python rng (preserves seeding semantics)
        idx = list(range(60))
        self._rng.shuffle(idx)
        shuffled = deck[idx]
        ptr = _ffi.cast("uint8_t *", _ffi.from_buffer(shuffled))
        _lib.init_game_from_deck(self._g, ptr, 60)

    # ── Properties (read from C struct) ──────────────────────────────────

    @property
    def current_player(self) -> int:
        return _lib.lc2_current_player(self._g)

    @property
    def game_over(self) -> bool:
        return bool(_lib.lc2_game_over(self._g))

    @property
    def scores(self) -> List[float]:
        return [float(_lib.lc2_score(self._g, 0)),
                float(_lib.lc2_score(self._g, 1))]

    # ── Card helpers ─────────────────────────────────────────────────────

    def encode_card(self, color_idx: int, card_value: int) -> Card:
        return color_idx, card_value

    def encode_card_to_id(self, color_idx: int, value: int) -> int:
        return int(_lib.lc2_card_encode(color_idx, value))

    def decode_card_id(self, i: int) -> Card:
        c = _ffi.cast("uint8_t", i)
        return _lib.lc2_card_color(c), _lib.lc2_card_value(c)

    def get_card_color(self, card: Card) -> int:
        return card[0]

    # ── Rule checks ──────────────────────────────────────────────────────

    def _is_valid_expedition_play(self, player_id: int, card: Card) -> bool:
        color_idx, value = card
        expedition = self.expeditions[player_id][color_idx]
        if not expedition: return True
        last_value = expedition[-1][1]
        if value == 0:
            if sum(1 for _, v in expedition if v == 0) >= 3: return False
            return last_value == 0
        return value > (last_value if last_value != 0 else 0)

    # ── Valid actions (C-accelerated, returns Python tuples for compat) ───

    def get_valid_actions(self, player_id: int) -> List[Tuple[int, int, int]]:
        n = _lib.get_valid_actions_c(self._g, player_id, self._act_buf)
        _u = _lib.lc2_action_unflatten
        card_id_p = _ffi.new("int*"); atype_p = _ffi.new("int*"); ds_p = _ffi.new("int*")
        result = []
        for i in range(n):
            flat = self._act_buf[i]
            _u(flat, card_id_p, atype_p, ds_p)
            result.append((int(card_id_p[0]), int(atype_p[0]), int(ds_p[0])))
        return result

    # ── Action execution ─────────────────────────────────────────────────

    def _apply_play(self, player_id: int, card: Card, action_type: ActionType):
        c_card = _ffi.cast("uint8_t", _lib.lc2_card_encode(card[0], card[1]))
        ret = _lib.apply_play_c(self._g, player_id, c_card, action_type.value)
        if ret != 0:
            raise ValueError(f"apply_play failed: {ret} card={card} atype={action_type}")

    def _apply_draw(self, player_id: int, draw_source: int):
        ret = _lib.apply_draw_c(self._g, player_id, draw_source)
        if ret != 0:
            raise ValueError(f"apply_draw failed: {ret} draw_source={draw_source}")

    def take_action(self, player_id: int, card: Card,
                    action_type: ActionType, draw_source: int):
        # draw_source here is -1=deck, 0..4=discard pile index
        # C's take_action_c uses env convention: 0=deck, 1..5=discard+1
        draw_source_env = 0 if draw_source == -1 else draw_source + 1
        c_card = _ffi.cast("uint8_t", _lib.lc2_card_encode(card[0], card[1]))
        ret = _lib.take_action_c(self._g, player_id, c_card,
                                  action_type.value, draw_source_env)
        if ret != 0:
            raise ValueError(f"take_action failed: ret={ret} player={player_id} "
                             f"card={card} atype={action_type} draw={draw_source}")

    # ── Scoring ──────────────────────────────────────────────────────────

    def _calculate_final_scores(self) -> List[float]:
        return self.scores

    # ── Python-list views (for backward compat, tests, etc.) ─────────────

    @property
    def hands(self) -> List[List[Card]]:
        result = []
        for p in range(2):
            hand = []
            for i in range(_lib.lc2_hand_size(self._g, p)):
                c = _lib.lc2_hand_card(self._g, p, i)
                hand.append((_lib.lc2_card_color(c), _lib.lc2_card_value(c)))
            result.append(hand)
        return result

    @property
    def expeditions(self) -> List[List[List[Card]]]:
        result = []
        for p in range(2):
            player_exps = []
            for c in range(NUM_COLORS):
                exp = []
                for i in range(_lib.lc2_exp_size(self._g, p, c)):
                    card = _lib.lc2_exp_card(self._g, p, c, i)
                    exp.append((_lib.lc2_card_color(card), _lib.lc2_card_value(card)))
                player_exps.append(exp)
            result.append(player_exps)
        return result

    @property
    def discard_piles(self) -> List[List[Card]]:
        result = []
        for c in range(NUM_COLORS):
            pile = []
            for i in range(_lib.lc2_dp_size(self._g, c)):
                card = self._g.discard[c][i]
                pile.append((_lib.lc2_card_color(card), _lib.lc2_card_value(card)))
            result.append(pile)
        return result

    @property
    def deck(self):
        """Returns a list-like with correct len() for compatibility."""
        return _DeckProxy(_lib.lc2_deck_top(self._g))

    def get_public_state(self, player_id: int) -> Dict[str, Any]:
        s = self.scores
        return {
            "player_id":          player_id,
            "current_player":     self.current_player,
            "game_over":          self.game_over,
            "scores":             s,
            "deck_size":          _lib.lc2_deck_top(self._g),
            "hand":               self.hands[player_id],
            "opponent_hand_size": _lib.lc2_hand_size(self._g, 1 - player_id),
            "expeditions":        self.expeditions,
            "discard_piles":      [self.discard_piles[c][-1]
                                   if _lib.lc2_dp_size(self._g, c) > 0 else None
                                   for c in range(NUM_COLORS)],
            "full_discard_piles": self.discard_piles,
            "color_names":        self.COLORS,
        }


class _DeckProxy:
    """Minimal deck proxy — only len() needed externally."""
    def __init__(self, size): self._size = size
    def __len__(self): return self._size
    def __bool__(self): return self._size > 0


# ═════════════════════════════════════════════════════════════════════════════
# 2.  FLAT ACTION HELPERS  (unchanged)
# ═════════════════════════════════════════════════════════════════════════════

def flatten_action(action_tuple: Tuple[int, int, int], dimensions: List[int]) -> int:
    N1, N2, N3 = dimensions; a1, a2, a3 = action_tuple
    if not (0 <= a1 < N1 and 0 <= a2 < N2 and 0 <= a3 < N3):
        raise ValueError(f"Action {action_tuple} out of bounds {dimensions}.")
    return a1 * (N2 * N3) + a2 * N3 + a3

def unflatten_action(flat_index: int, dimensions: List[int]) -> Tuple[int, int, int]:
    N1, N2, N3 = dimensions; total = math.prod(dimensions)
    if not (0 <= flat_index < total):
        raise ValueError(f"flat_index {flat_index} out of [0, {total - 1}].")
    a3 = flat_index % N3;  flat_index //= N3
    a2 = flat_index % N2;  a1 = flat_index // N2
    return a1, a2, a3


# ═════════════════════════════════════════════════════════════════════════════
# 3.  AGENTS
# ═════════════════════════════════════════════════════════════════════════════

class Agent(ABC):
    @abstractmethod
    def act(self, obs: np.ndarray, action_mask: np.ndarray) -> int: pass

class BatchedAgent(ABC):
    @abstractmethod
    def act_batch(self, obs_batch: np.ndarray, mask_batch: np.ndarray) -> np.ndarray: pass

class RandomAgent(Agent):
    def act(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        return int(np.random.choice(action_mask.nonzero()[0]))

class RandomBatchedAgent(BatchedAgent):
    def act_batch(self, obs_batch: np.ndarray, mask_batch: np.ndarray) -> np.ndarray:
        N = obs_batch.shape[0]; actions = np.empty(N, dtype=np.int64)
        for i in range(N):
            valid = mask_batch[i].nonzero()[0]; actions[i] = np.random.choice(valid)
        return actions


# ═════════════════════════════════════════════════════════════════════════════
# 4.  SINGLE-AGENT GYM ENVIRONMENT
# ═════════════════════════════════════════════════════════════════════════════

class LostCitiesEnv(gym.Env if HAS_GYM else object):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, opponent_agent: Agent):
        super().__init__()
        self.game           = LostCitiesGame()
        self.opponent_agent = opponent_agent

        self._action_dims = [CARD_POOL, 2, NUM_COLORS + 1]
        self.action_space = spaces.Discrete(TOTAL_ACTIONS)

        # Pre-allocated buffers
        self._obs_buf  = np.zeros(OBS_SIZE,      dtype=np.int8)
        self._mask_buf = np.zeros(TOTAL_ACTIONS, dtype=np.uint8)

        self.game.reset()
        low  = np.zeros(OBS_SIZE, dtype=np.int8)
        high = np.full(OBS_SIZE, 3, dtype=np.int8)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int8)
        self._scores = (0.0, 0.0)

    def _get_obs(self, player_id: int) -> np.ndarray:
        _lib.build_obs_c(self.game._g, player_id,
                         _ffi.cast("int8_t *", _ffi.from_buffer(self._obs_buf)))
        return self._obs_buf.copy()

    def get_action_mask(self, player_id: int) -> np.ndarray:
        self._mask_buf[:] = 0
        _lib.fill_action_mask_c(self.game._g, player_id,
                                _ffi.cast("uint8_t *", _ffi.from_buffer(self._mask_buf)))
        return self._mask_buf.view(bool)

    def _decode_flat(self, flat: int) -> Tuple[Card, ActionType, int]:
        card_id_p = _ffi.new("int*"); atype_p = _ffi.new("int*"); ds_p = _ffi.new("int*")
        _lib.lc2_action_unflatten(flat, card_id_p, atype_p, ds_p)
        card_id = int(card_id_p[0]); atype = int(atype_p[0]); ds_env = int(ds_p[0])
        c = _ffi.cast("uint8_t", card_id)
        card = (_lib.lc2_card_color(c), _lib.lc2_card_value(c))
        return card, ActionType(atype), ds_env - 1

    def set_opponent(self, opponent: Agent): self.opponent_agent = opponent

    def reset(self, seed=None, options=None):
        if HAS_GYM: super().reset(seed=seed)
        self.game.reset(); self._scores = (0.0, 0.0)
        return self._get_obs(0), {}

    def step(self, action: int):
        card, atype, draw = self._decode_flat(action)
        self.game.take_action(0, card, atype, draw)
        terminated = self.game.game_over

        if not terminated:
            obs1  = self._get_obs(1)
            mask1 = self.get_action_mask(1)
            a1    = int(self.opponent_agent.act(obs1, mask1))
            card1, atype1, draw1 = self._decode_flat(a1)
            self.game.take_action(1, card1, atype1, draw1)
            terminated = self.game.game_over

        new_scores = (float(_lib.lc2_score(self.game._g, 0)),
                      float(_lib.lc2_score(self.game._g, 1)))
        reward = (new_scores[0] - new_scores[1]) - (self._scores[0] - self._scores[1])
        self._scores = new_scores
        return self._get_obs(0), reward, terminated, False, {}

    def render(self, mode="human"):
        g = self.game
        print(f"--- Lost Cities | Deck={_lib.lc2_deck_top(g._g)} | "
              f"P{g.current_player}'s turn ---")
        if g.game_over:
            print(f"GAME OVER  P0={g.scores[0]}  P1={g.scores[1]}")


# ═════════════════════════════════════════════════════════════════════════════
# 5.  VECTORISED ENVIRONMENT
# ═════════════════════════════════════════════════════════════════════════════

class VecLostCitiesEnv:
    """
    Runs num_envs games in parallel with all game logic in C.

    The C collect_obs_masks_batch() function is called with a pointer array
    — it fills all N obs and mask rows in a single C loop with no Python
    overhead between games.
    """

    def __init__(self, num_envs: int, opponent_agent: BatchedAgent):
        self.num_envs = num_envs
        self.opponent = opponent_agent

        # Allocate N C game structs
        self._game_bufs = [_ffi.new("char[]", _GAME_SIZE) for _ in range(num_envs)]
        self._game_ptrs = [_ffi.cast("LCGame *", buf) for buf in self._game_bufs]
        # C-level pointer array for batch calls
        self._ptr_array = _ffi.new(f"LCGame *[{num_envs}]")
        for i, p in enumerate(self._game_ptrs):
            self._ptr_array[i] = p
        # Python-facing LostCitiesGame wrappers (share the C buffer)
        self.games = [_CGameWrapper(buf) for buf in self._game_bufs]

        self._action_dims = [CARD_POOL, 2, NUM_COLORS + 1]
        self._action_size = TOTAL_ACTIONS
        self._obs_size    = OBS_SIZE

        # Pre-allocated batch buffers
        self._obs_p0  = np.zeros((num_envs, OBS_SIZE),      dtype=np.int8)
        self._obs_p1  = np.zeros((num_envs, OBS_SIZE),      dtype=np.int8)
        self._mask_p0 = np.zeros((num_envs, TOTAL_ACTIONS), dtype=np.uint8)
        self._mask_p1 = np.zeros((num_envs, TOTAL_ACTIONS), dtype=np.uint8)
        self._scores  = np.zeros((num_envs, 2),             dtype=np.float32)
        self._new_scores_buf = np.zeros((num_envs, 2),      dtype=np.float32)

        # CFFI pointers for batch buffers
        self._p_obs_p0  = _ffi.cast("int8_t *",  _ffi.from_buffer(self._obs_p0))
        self._p_obs_p1  = _ffi.cast("int8_t *",  _ffi.from_buffer(self._obs_p1))
        self._p_mask_p0 = _ffi.cast("uint8_t *", _ffi.from_buffer(self._mask_p0))
        self._p_mask_p1 = _ffi.cast("uint8_t *", _ffi.from_buffer(self._mask_p1))
        self._p_new_scores = _ffi.cast("float *", _ffi.from_buffer(self._new_scores_buf))

        # Scratch for unflatten
        self._card_id_p = _ffi.new("int*")
        self._atype_p   = _ffi.new("int*")
        self._ds_p      = _ffi.new("int*")

        self.single_observation_space = spaces.Box(
            low=np.zeros(OBS_SIZE, dtype=np.int8),
            high=np.full(OBS_SIZE, 3, dtype=np.int8), dtype=np.int8)
        self.single_action_space = spaces.Discrete(TOTAL_ACTIONS)
        self.observation_space   = batch_space(self.single_observation_space, num_envs)
        self.action_space        = batch_space(self.single_action_space, num_envs)

    def _decode_flat_into(self, flat: int):
        """Decode into pre-allocated scratch pointers."""
        _lib.lc2_action_unflatten(flat, self._card_id_p, self._atype_p, self._ds_p)

    def _collect_batch(self, player_id: int,
                       obs_buf: np.ndarray, mask_buf: np.ndarray,
                       p_obs, p_mask) -> None:
        """Single C call fills all N obs+mask rows."""
        _lib.collect_obs_masks_batch(
            self._ptr_array, self.num_envs, player_id, p_obs, p_mask)

    # For backward compat with tests
    def _get_obs_single(self, game_wrapper, player_id: int) -> np.ndarray:
        buf = np.zeros(OBS_SIZE, dtype=np.int8)
        _lib.build_obs_c(game_wrapper._g, player_id,
                         _ffi.cast("int8_t *", _ffi.from_buffer(buf)))
        return buf

    def _get_mask_single(self, game_wrapper, player_id: int) -> np.ndarray:
        buf = np.zeros(TOTAL_ACTIONS, dtype=np.uint8)
        _lib.fill_action_mask_c(game_wrapper._g, player_id,
                                _ffi.cast("uint8_t *", _ffi.from_buffer(buf)))
        return buf.view(bool)

    def reset(self, seed=None, options=None):
        if seed is not None:
            seed_seq    = np.random.SeedSequence(seed)
            child_seeds = seed_seq.spawn(self.num_envs)
            for i, child in enumerate(child_seeds):
                self.games[i].reset(seed=int(child.generate_state(1)[0]))
        else:
            for game in self.games:
                game.reset()
        self._scores[:] = 0.0
        _lib.collect_obs_masks_batch(
            self._ptr_array, self.num_envs, 0, self._p_obs_p0, self._p_mask_p0)
        return self._obs_p0.copy(), {}

    def get_action_masks(self, player_id: int = 0) -> np.ndarray:
        p_obs  = self._p_obs_p0  if player_id == 0 else self._p_obs_p1
        p_mask = self._p_mask_p0 if player_id == 0 else self._p_mask_p1
        _lib.collect_obs_masks_batch(
            self._ptr_array, self.num_envs, player_id, p_obs, p_mask)
        m = self._mask_p0 if player_id == 0 else self._mask_p1
        return m.view(bool).copy()

    def step(self, actions_p0: np.ndarray):
        N          = self.num_envs
        terminated = np.zeros(N, dtype=bool)

        # ── Phase 1: all p0 turns ────────────────────────────────────────
        for i in range(N):
            flat = int(actions_p0[i])
            _lib.lc2_action_unflatten(flat, self._card_id_p, self._atype_p, self._ds_p)
            card_id  = int(self._card_id_p[0])
            atype    = int(self._atype_p[0])
            draw_src = int(self._ds_p[0]) - 1   # env→internal: 0→-1, 1..5→0..4
            c_card   = _ffi.cast("uint8_t", card_id)
            _lib.apply_play_c(self._game_ptrs[i], 0, c_card, atype)
            _lib.apply_draw_c(self._game_ptrs[i], 0, draw_src)
            if _lib.lc2_game_over(self._game_ptrs[i]):
                terminated[i] = True

        # ── Phase 2: collect p1 obs in one C call ───────────────────────
        _lib.collect_obs_masks_batch(
            self._ptr_array, N, 1, self._p_obs_p1, self._p_mask_p1)
        actions_p1 = self.opponent.act_batch(self._obs_p1, self._mask_p1.view(bool))

        # ── Phase 3: all p1 turns ────────────────────────────────────────
        for i in range(N):
            if terminated[i]: continue
            flat = int(actions_p1[i])
            _lib.lc2_action_unflatten(flat, self._card_id_p, self._atype_p, self._ds_p)
            card_id  = int(self._card_id_p[0])
            atype    = int(self._atype_p[0])
            draw_src = int(self._ds_p[0]) - 1
            c_card   = _ffi.cast("uint8_t", card_id)
            _lib.apply_play_c(self._game_ptrs[i], 1, c_card, atype)
            _lib.apply_draw_c(self._game_ptrs[i], 1, draw_src)
            if _lib.lc2_game_over(self._game_ptrs[i]):
                terminated[i] = True

        # ── Rewards ──────────────────────────────────────────────────────
        _lib.scores_batch(self._ptr_array, N, self._p_new_scores)
        rewards = ((self._new_scores_buf[:, 0] - self._new_scores_buf[:, 1])
                   - (self._scores[:, 0] - self._scores[:, 1]))
        np.copyto(self._scores, self._new_scores_buf)

        # ── Auto-reset ────────────────────────────────────────────────────
        for i in terminated.nonzero()[0]:
            self.games[i].reset()
            self._scores[i] = 0.0

        # ── Next p0 obs ───────────────────────────────────────────────────
        _lib.collect_obs_masks_batch(
            self._ptr_array, N, 0, self._p_obs_p0, self._p_mask_p0)
        return self._obs_p0.copy(), rewards, terminated, np.zeros(N, dtype=bool), {}


class _CGameWrapper:
    """
    Thin Python wrapper around an LCGame * that also holds an RNG.
    Exposes enough of the LostCitiesGame interface for the step loop.
    """
    COLORS             = _COLORS
    CARD_VALUES        = _CARD_VALUES
    UNIQUE_CARD_VALUE  = 10
    NUM_PLAYERS        = 2
    STARTING_HAND_SIZE = 8

    def __init__(self, buf):
        self._buf = buf
        self._g   = _ffi.cast("LCGame *", buf)
        self._rng = random.Random()
        self._act_buf = _ffi.new("int[128]")

    def reset(self, seed=None):
        if seed is not None:
            self._rng = random.Random(seed)
        deck = _DECK_TEMPLATE.copy()
        idx = list(range(60))
        self._rng.shuffle(idx)
        shuffled = deck[idx]
        ptr = _ffi.cast("uint8_t *", _ffi.from_buffer(shuffled))
        _lib.init_game_from_deck(self._g, ptr, 60)

    @property
    def current_player(self):
        return _lib.lc2_current_player(self._g)

    @property
    def game_over(self):
        return bool(_lib.lc2_game_over(self._g))

    @property
    def scores(self):
        return [float(_lib.lc2_score(self._g, 0)),
                float(_lib.lc2_score(self._g, 1))]

    def decode_card_id(self, i):
        c = _ffi.cast("uint8_t", i)
        return _lib.lc2_card_color(c), _lib.lc2_card_value(c)

    def _calculate_final_scores(self):
        return self.scores

    # Property stubs used by test helpers only
    @property
    def hands(self):
        result = []
        for p in range(2):
            hand = []
            for i in range(_lib.lc2_hand_size(self._g, p)):
                c = _lib.lc2_hand_card(self._g, p, i)
                hand.append((_lib.lc2_card_color(c), _lib.lc2_card_value(c)))
            result.append(hand)
        return result

    @property
    def deck(self):
        return _DeckProxy(_lib.lc2_deck_top(self._g))

    @property
    def expeditions(self):
        result = []
        for p in range(2):
            player_exps = []
            for c in range(NUM_COLORS):
                exp = []
                for i in range(_lib.lc2_exp_size(self._g, p, c)):
                    card = _lib.lc2_exp_card(self._g, p, c, i)
                    exp.append((_lib.lc2_card_color(card), _lib.lc2_card_value(card)))
                player_exps.append(exp)
            result.append(player_exps)
        return result

    @property
    def discard_piles(self):
        result = []
        for c in range(NUM_COLORS):
            pile = []
            for i in range(_lib.lc2_dp_size(self._g, c)):
                card = self._g.discard[c][i]
                pile.append((_lib.lc2_card_color(card), _lib.lc2_card_value(card)))
            result.append(pile)
        return result
