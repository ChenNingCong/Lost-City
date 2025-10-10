
# --- CORE GAME LOGIC (Adapted from previous response) ---
from typing import *
import random
import enum
Card = Tuple[int, int]
class ActionType(enum.Enum):
    Play = 0
    DISCARD = 1

class LostCitiesGame:
    """Encapsulates the core rules and state of the Lost Cities card game."""

    COLORS = ["Blue", "Yellow", "White", "Green", "Red"]
    CARD_VALUES = [0, 0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    UNIQUE_CARD_VALUE = len(set(CARD_VALUES))
    NUM_PLAYERS = 2
    STARTING_HAND_SIZE = 8
    
    COLOR_TO_IDX = {color: i for i, color in enumerate(COLORS)}
    # (Color Index, Value) is the card format
    def __init__(self):
        self.deck: List[Card] = []
        self.hands: List[List[Card]] = [[] for _ in range(self.NUM_PLAYERS)]
        self.discard_piles: List[List[Card]] = [[] for _ in range(len(self.COLORS))]
        self.expeditions: List[List[List[Card]]] = [[[] for _ in range(len(self.COLORS))] for _ in range(self.NUM_PLAYERS)]
        self.current_player: int = 0
        self.game_over: bool = False
        self.scores: List[float] = [0.0, 0.0]
    def encode_card(self, color_idx : int, card_value : int) -> Card:
        return color_idx, card_value
    
    def _generate_deck(self):
        """Creates and shuffles the 60-card deck."""
        deck = [self.encode_card(color_idx, value) for color_idx, _ in enumerate(self.COLORS) for value in self.CARD_VALUES]
        random.shuffle(deck)
        self.deck = deck

    def reset(self):
        """Initializes the game state for a new game."""
        self._generate_deck()
        self.hands = [[] for _ in range(self.NUM_PLAYERS)]
        self.discard_piles = [[] for _ in range(len(self.COLORS))]
        self.expeditions = [[[] for _ in range(len(self.COLORS))] for _ in range(self.NUM_PLAYERS)]
        self.current_player = 0
        self.game_over = False
        self.scores = [0.0, 0.0]

        for _ in range(self.STARTING_HAND_SIZE):
            for p in range(self.NUM_PLAYERS):
                self.hands[p].append(self.deck.pop())
    
    def _is_valid_expedition_play(self, player_id: int, card: Card) -> bool:
        """Checks if a card can be played to an expedition."""
        color_idx, value = card
        expedition = self.expeditions[player_id][color_idx]
        
        if not expedition:
            return True 
        
        last_value = expedition[-1][1] 
        
        if value == 0: # Investment Card
            num_investments = sum(1 for _, v in expedition if v == 0)
            if num_investments >= 3:
                return False 
            if last_value != 0:
                return False # Cannot play investment after a numbered card
            return True
            
        else: # Numbered card (2-10)
            effective_last_value = last_value if last_value != 0 else 0
            return value > effective_last_value
    def get_valid_actions_frontend(self, player_id: int) -> Dict[str, Any]:
        """
        Returns a dictionary of valid moves for a player.
        
        Valid actions are:
        1. Play to Expedition (5 colors * 1 card)
        2. Discard (5 colors * 1 card)
        3. Draw from Draw Pile (1 option)
        4. Draw from Discard Pile (5 colors)
        
        The full action space combines (Play/Discard) + Draw.
        """
        valid_plays = [] # (card_idx_in_hand, action_type: 'E'/'D', color_idx)
        hand = self.hands[player_id]
        
        for i, card in enumerate(hand):
            color_idx, value = card
            
            # 1. Expedition Play check
            if self._is_valid_expedition_play(player_id, card):
                valid_plays.append((i, 'E', color_idx))
            
            # 2. Discard is always valid
            valid_plays.append((i, 'D', color_idx))
            
        # Draw options
        valid_draws = [0] # 0 = Draw Pile
        for color_idx in range(len(self.COLORS)):
            if self.discard_piles[color_idx]:
                valid_draws.append(color_idx + 1) # 1-5 = Discard Piles
                
        return {'plays': valid_plays, 'draws': valid_draws}

    def get_valid_actions(self, player_id: int):
        """Returns a list of valid (card_idx, action_type, color_idx) plays and valid draw sources."""
        valid_plays : List[Tuple[int,int,int]] = [] # (card_idx, play_type, draw_source)
        hand = self.hands[player_id]
        
        for i, card in enumerate(hand):
            color_idx, value_idx = card
            
            # 1. Expedition Play check
            if self._is_valid_expedition_play(player_id, card):
                for draw_source in range(len(self.COLORS) + 1):
                    if draw_source == 0 or len(self.discard_piles[draw_source-1]) > 0:
                        valid_plays.append((self.encode_card_to_id(color_idx, value_idx), ActionType.Play.value, draw_source))
            
            # 2. Discard is always valid
            for draw_source in range(len(self.COLORS) + 1):
                # either we draw a card or we draw from a card that's no this color
                if draw_source == 0 or (len(self.discard_piles[draw_source-1]) > 0 and draw_source - 1 != color_idx):
                    valid_plays.append((self.encode_card_to_id(color_idx, value_idx), ActionType.DISCARD.value, draw_source))

        return valid_plays
    
    def get_card_color(self, card : Card):
        return card[0]
    
    def take_action(self, player_id: int, card: Card, action_type: ActionType, draw_source: int):
        """Executes a full turn (Play + Draw)."""
        assert self.current_player == player_id
        # 1. Validation (Essential for a robust server)
        hand = self.hands[player_id]
        if card not in hand:
            raise ValueError(f"Invalid card {card}.")
        
        # Check Play validity
        if action_type == ActionType.Play and not self._is_valid_expedition_play(player_id, card):
            raise ValueError("Invalid expedition play.")
        color_idx = self.get_card_color(card) 
        # Check Draw validity
        # -1 == deck, else from discard pile
        if draw_source != -1:
            if not self.discard_piles[draw_source]:
                raise ValueError(f"The discard pipe is empty {draw_source}")
            if action_type == ActionType.DISCARD and color_idx == draw_source:
                raise ValueError(f"Cannot draw from discard pile {draw_source}")
        hand.remove(card)
        if action_type == ActionType.Play:
            self.expeditions[player_id][color_idx].append(card)
        else:
            self.discard_piles[color_idx].append(card)
            
        # 3. Draw Card
        if draw_source == -1: # Draw from main deck
            assert self.deck
            self.hands[player_id].append(self.deck.pop())
        else: # Draw from a discard pile
            self.hands[player_id].append(self.discard_piles[draw_source].pop())
        
        # 4. Check for game end and switch player
        if not self.deck:
            self.game_over = True
            self.scores = self._calculate_final_scores()
        else:
            self.current_player = 1 - player_id

    def _calculate_final_scores(self) -> List[float]:
        """Calculates the final score for both players."""
        final_scores = [0.0, 0.0]
        
        for p in range(self.NUM_PLAYERS):
            total_score = 0
            for color_idx in range(len(self.COLORS)):
                expedition = self.expeditions[p][color_idx]
                if not expedition:
                    continue
                
                base_value = sum(value for _, value in expedition)
                num_investments = sum(1 for _, value in expedition if value == 0)
                multiplier = num_investments + 1
                
                expedition_score = (base_value - 20) * multiplier
                
                if len(expedition) >= 8:
                    expedition_score += 20
                    
                total_score += expedition_score
                
            final_scores[p] = total_score
        return final_scores

    def get_public_state(self, player_id: int) -> Dict[str, Any]:
        """Returns the state visible to a specific player (including their hand)."""
        s = self._calculate_final_scores()
        state = {
            'player_id': player_id,
            'current_player': self.current_player,
            'game_over': self.game_over,
            'scores': s,
            'deck_size': len(self.deck),
            'hand': self.hands[player_id],
            'opponent_hand_size': len(self.hands[1-player_id]),
            'expeditions': self.expeditions,
            'discard_piles': [pile[-1] if pile else None for pile in self.discard_piles], # Only top card is visible
            'full_discard_piles': self.discard_piles, # For a server where opponents don't cheat, we can send all
            'color_names': self.COLORS,
        }
        return state
    def encode_card_to_id(self, color_idx : int, value_idx : int):
        return color_idx * self.UNIQUE_CARD_VALUE + (0 if value_idx == 0 else value_idx - 1)
    def decode_card_id(self, i : int) -> Card:
        color_idx = i // self.UNIQUE_CARD_VALUE
        value_idx = i % self.UNIQUE_CARD_VALUE
        return color_idx, value_idx + (value_idx != 0)
        

# --- 2. GYMNASIUM ENVIRONMENT CLASS ---
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Callable, Tuple, Dict, Any, Optional
from collections import defaultdict
from abc import ABC, abstractmethod
class Agent(ABC):
    @abstractmethod
    def act(self, obs : np.ndarray, action_mask : np.ndarray) -> int:
        pass
class RandomAgent(Agent):
    def __init__(self, id) -> None:
        self.id = id
    def act(self, obs: np.ndarray, action_mask : np.ndarray) -> int:
        action_set = action_mask.nonzero()[0]
        return int(np.random.choice(action_set))

import math
def flatten_action(action_tuple: Tuple[int, int, int], dimensions: List[int]) -> int:
    """Converts a multi-dimensional tuple action into a single integer index."""
    if len(action_tuple) != len(dimensions):
        raise ValueError("Action tuple and dimensions list must have the same length.")

    N1, N2, N3 = dimensions
    a1, a2, a3 = action_tuple

    if not (0 <= a1 < N1 and 0 <= a2 < N2 and 0 <= a3 < N3):
        raise ValueError(f"Action component {action_tuple} out of bounds.")

    # I = a1 * (N2 * N3) + a2 * N3 + a3
    flat_index = a1 * (N2 * N3) + a2 * N3 + a3
    return flat_index

def unflatten_action(flat_index: int, dimensions: List[int]) -> Tuple[int, ...]:
    """
    Converts a single integer index back into the multi-dimensional tuple action
    using direct, unrolled calculations for fixed 3 dimensions (N1, N2, N3).
    This is generally faster than iterating for a fixed, small number of dimensions.
    """
    total_size = math.prod(dimensions)
    if not (0 <= flat_index < total_size):
        raise ValueError(f"Flat index {flat_index} is out of bounds [0, {total_size - 1}].")

    # Dimensions: N1 (Card ID, Slowest), N2 (Play Type), N3 (Draw Source, Fastest)
    N1, N2, N3 = dimensions
    temp_index = flat_index
    
    # 1. Extract a3 (Fastest changing component, Draw Source)
    a3 = temp_index % N3
    temp_index = temp_index // N3
    
    # 2. Extract a2 (Play Type)
    a2 = temp_index % N2
    temp_index = temp_index // N2
    
    # 3. Extract a1 (Slowest changing component, Card ID)
    # The remainder of the index after two divisions is the value of a1.
    a1 = temp_index 

    return (a1, a2, a3)

class LostCitiesEnv(gym.Env):
    """
    A single-agent Gymnasium environment for Lost Cities, 
    with a provided agent acting as the opponent.
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}
    def _flatten_action(self, action):
        return flatten_action(action, self._action_dimension)
    def _unflatten_action(self, action : int):
        return unflatten_action(action, self._action_dimension)
    def __init__(self, opponent_agent: 'Agent'):
        super().__init__()
        self.game = LostCitiesGame()
        self.opponent_agent = opponent_agent # Function for Player 1 actions
        # Note: This abstract action space is HUGE and requires internal masking/logic in `step`.
        self.unflatten_action_space = spaces.Tuple((
            spaces.Discrete(len(self.game.COLORS) * self.game.UNIQUE_CARD_VALUE),   
                                                          # Play Card Id, which equals to id of unique cards
            spaces.Discrete(2),                           # Play type 0=Expedition, 1=Discard
            spaces.Discrete(len(self.game.COLORS) + 1)    # Draw source 0=Deck, 1-5=Discard
        ))
        self._action_dimension : List[int] = [int(i.n) for i in self.unflatten_action_space.spaces]
        # closed interval, so we minius one here...
        self.action_space = spaces.Discrete(math.prod(self._action_dimension))
        
        # --- Observation Space ---
        # State includes: Agent Hand, Opponent Expeditions, Discard Piles, Deck Size, Agent Expeditions.
        # A full observation space is also complex. We'll define a simplified, fixed-size **Vector** space.
        
        # 1. Agent Hand (8 cards * (color, value) -> 8 * 2 = 16 features)
        # 2. Agent Expeditions (5 colors * (top_card_value, num_investments) -> 5 * 2 = 10 features)
        # 3. Opponent Expeditions (5 colors * (top_card_value, num_investments) -> 5 * 2 = 10 features)
        # 4. Discard Piles (5 colors * top_card_value -> 5 features)
        # 5. Deck Size (1 feature)
        # Total Features: 16 + 10 + 10 + 5 + 1 = 42
        
        # Feature vector will be:
        # [Hand_C1, Hand_V1, ..., Hand_C8, Hand_V8, 
        #  AE_TopV1, AE_Inv1, ..., AE_TopV5, AE_Inv5, 
        #  OE_TopV1, OE_Inv1, ..., OE_TopV5, OE_Inv5, 
        #  DP_TopV1, ..., DP_TopV5, Deck_Size]
        self.reset()
        obs = self._get_obs(0)
        self.scores = (0,0)
        low = np.array(np.zeros_like(obs), dtype=np.int8)
        high = np.array(np.zeros_like(obs) + 3, dtype=np.int8)
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int8)


    def _get_obs(self, player_id: int) -> np.ndarray:
        """Converts the game state into the fixed-size observation vector for the given player."""
        state :List[int] = []
        def encode_card_subset(x : List[Card]):
            x_vector = [0 for i in range(len(self.game.COLORS) * self.game.UNIQUE_CARD_VALUE)]
            for (color_idx, value_idx) in x:
                x_vector[self.game.encode_card_to_id(color_idx, value_idx)] += 1
            return x_vector
        state.extend(encode_card_subset(self.game.hands[player_id]))
        def flatten(x : List[List[Card]]):
            result = []
            for i in x:
                result.extend(i)
            return result
        # 2. Agent Expeditions (5 colors)
        state.extend(encode_card_subset(flatten(self.game.expeditions[player_id])))
            
        # 3. Opponent Expeditions (5 colors)
        opponent_id = 1 - player_id
        state.extend(encode_card_subset(flatten(self.game.expeditions[opponent_id])))

        # 4. Discard Piles (5 colors, top card value, 0 if empty)
        # Encode only the top cards
        for i in range(3):
            state.extend(encode_card_subset([pile[-i-1] for pile in self.game.discard_piles if len(pile) > i]))

        # 5. Deck Size
        state.append(len(self.game.deck))
        
        return np.array(state, dtype=np.int8)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment for a new episode."""
        super().reset(seed=seed)
        self.game.reset()
        self.scores = (0,0)
        
        # If the opponent is Player 0, let them play first (not applicable here, P0 is the agent)
        observation = self._get_obs(player_id=0)
        info = {}
        
        return observation, info
    def set_opponent(self, opponent : Agent):
        self.opponent_agent = opponent
    def get_valid_action_set(self, player_id:int)->List[int]:
        return [self._flatten_action(i) for i in self.game.get_valid_actions(player_id)]
    
    def get_action_mask(self, player_id:int):
        # Create a mask filled with zeros
        int_list = self.get_valid_action_set(player_id)
        mask = np.zeros(int(self.action_space.n), dtype=np.bool)
        # Set the positions specified in int_list to 1
        mask[int_list] = 1
        return mask
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes a turn for Player 0 and the opponent's turn (Player 1).
        
        Args:
            action: (card_index_in_hand (0-7), play_type (0/1), color_index (0-4), draw_source (0-5))
        """
        assert self.game.current_player == 0, "Not the agent's turn."

        # --- Player 0 (Agent) Turn ---
        card_idx, play_type, draw_source = self._unflatten_action(action)
        
        # Translate action components into game format
        card = self.game.decode_card_id(card_idx)
        
        # Note: In a real-world setting, we would need to check if the action is valid (masking)
        # For this design, we assume the agent only submits valid moves based on the valid action space.
        self.game.take_action(player_id=0, card=card, action_type=ActionType(play_type), draw_source=draw_source-1)

        # --- Player 1 (Opponent) Turn ---
        reward = 0.0
        terminated = self.game.game_over
        truncated = False # Gymnasium standard

        if not terminated:
            # Get opponent action from the provided agent function
            opponent_state = self._get_obs(1)
            # The opponent agent must return: (card_index, action_type ('E'/'D'), color_index), draw_source (0-5)
            opponent_card_idx, opponent_play_type, opponent_draw_source = self._unflatten_action(int(self.opponent_agent.act(opponent_state, self.get_action_mask(1))))
            opponent_card = self.game.decode_card_id(opponent_card_idx)
            
            # Execute opponent's move
            self.game.take_action(player_id=1, card=opponent_card, action_type=ActionType(opponent_play_type), draw_source=opponent_draw_source-1)
            
            # Check for game end again
            terminated = self.game.game_over
        
        if terminated:
            # Reward is the difference in final scores
            reward = self.game.scores[0] - self.game.scores[1]
            
        observation = self._get_obs(player_id=0)
        info = {}
        # compute score for the new state
        new_scores = self.game._calculate_final_scores()
        reward = (new_scores[0] - new_scores[1]) - (self.scores[0] - self.scores[1])
        self.scores = new_scores
        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        """Prints the current game state to the console."""
        print("--- Lost Cities Game State ---")
        print(f"Current Player: P{self.game.current_player}")
        print(f"Deck Size: {len(self.game.deck)}")
        
        # Helper to format cards
        def format_card(card):
            return f"({self.game.COLORS[card[0]][0]}{card[1]})"
            
        for p in range(self.game.NUM_PLAYERS):
            print(f"\n--- Player {p} ---")
            print(f"  Hand: {[format_card(c) for c in self.game.hands[p]]}")
            print("  Expeditions:")
            for c_idx, exp in enumerate(self.game.expeditions[p]):
                if exp:
                    print(f"    {self.game.COLORS[c_idx]}: {[format_card(c) for c in exp]}")
        
        print("\n--- Discard Piles ---")
        for c_idx, pile in enumerate(self.game.discard_piles):
            top_card = format_card(pile[-1]) if pile else "Empty"
            print(f"  {self.game.COLORS[c_idx]}: Top Card: {top_card} (Size: {len(pile)})")
            
        if self.game.game_over:
            print("\n!!! GAME OVER !!!")
            print(f"Final Scores: P0: {self.game.scores[0]}, P1: {self.game.scores[1]}")
            print(f"Result: P0 wins by {self.game.scores[0] - self.game.scores[1]:.2f}" if self.game.scores[0] > self.game.scores[1] else "Tie" if self.game.scores[0] == self.game.scores[1] else f"P1 wins by {self.game.scores[1] - self.game.scores[0]:.2f}")

