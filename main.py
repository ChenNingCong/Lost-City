# main.py

from dataclasses import dataclass
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
# --- FASTAPI SETUP ---
from game import RandomAgent, LostCitiesEnv, ActionType
app = FastAPI()
env = LostCitiesEnv(RandomAgent(1))
@dataclass
class GameServerState:
    env: LostCitiesEnv
game_server_state = GameServerState(env=env)

# Configure CORS to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Schemas for API requests ---

class PlayAction(BaseModel):
    player_id: int # 0 or 1
    card_index: int # Index in the player's hand (0-7)
    action_type: str # 'E' for Expedition, 'D' for Discard
    draw_source: int # 0 for Deck, 1-5 for Discard Piles

# --- API Endpoints ---

@app.get("/game/reset")
def reset_game():
    """Resets the game state and starts a new game."""
    game_server_state.env.reset()
    return {"message": "Game reset successfully. New game started."}

@app.get("/game/opponent")
def get_opponent():
    """Retrieves the public game state and the player's private hand."""
    import glob
    return [{"id" : i, 'name' : path} for i,path in enumerate(glob.glob("model/*.pkt"))]

@app.get("/game/state/{player_id}")
def get_game_state(player_id: int):
    """Retrieves the public game state and the player's private hand."""
    game = game_server_state.env.game
    if player_id not in [0, 1]:
        raise HTTPException(status_code=400, detail="Invalid player ID.")
    return game.get_public_state(player_id)
class OpponentPath(BaseModel):
    path: str # Defines that 'path' must be in the JSON body

@app.post("/game/set_opponent")
def set_opponent(path : OpponentPath):
    print(f"Set path {path}")
    import torch
    agent_state_dict = torch.load(path.path)
    from ppo import Agent
    def make_agent():
        # only use to get the action space and observation space, which are needed to initialize the agent
        from lost_cities_env_fast2 import VecLostCitiesEnv, RandomBatchedAgent
        fake_env = VecLostCitiesEnv(1, RandomBatchedAgent())
        return Agent(fake_env).cuda()
    agent = make_agent()
    agent.load_state_dict(agent_state_dict)
    game_server_state.env.set_opponent(agent)
    return {"message": f"Opponent set to model at {path.path}"}

@app.post("/game/play")
def handle_play_action(action: PlayAction):
    """Handles a player's move (play card + draw card)."""
    env = game_server_state.env
    game = env.game

    if game.game_over:
        return {"message": "Game is over. Reset to start a new game."}

    player_id = action.player_id
    assert player_id == 0
    if player_id != game.current_player:
        raise HTTPException(status_code=400, detail=f"It is not Player {player_id}'s turn. It is Player {game.current_player}'s turn.")

    try:
        # Get card details before the list changes
        card_to_play = game.hands[player_id][action.card_index]

        # Execute the move
        # game.take_action(
        #     player_id=player_id, 
        #     card=card_to_play, 
        #     action_type=ActionType.Play if action.action_type == 'E' else ActionType.DISCARD, 
        #     draw_source=action.draw_source - 1
        # )
        card_id = game.encode_card_to_id(*card_to_play)
        action_t = (card_id, ActionType.Play.value if action.action_type == 'E' else ActionType.DISCARD.value, action.draw_source)
        env.step(action=env._flatten_action(action_t))
        message = (f"P{player_id} played {game.COLORS[card_to_play[0]]} {card_to_play[1]} to "
                   f"{'expedition' if action.action_type == 'E' else 'discard'}. "
                   f"Drew from {'Deck' if action.draw_source == 0 else game.COLORS[action.draw_source - 1] + ' Discard'}.")
        
        if game.game_over:
            message += f" Game Over! Scores: P0: {game.scores[0]}, P1: {game.scores[1]}"
        return {"message": message, "state": game.get_public_state(player_id)}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IndexError:
        raise HTTPException(status_code=400, detail="Invalid card index in hand.")


@app.get("/game/valid_actions/{player_id}")
def get_valid_moves(player_id: int):
    """Retrieves all valid moves for the current player."""
    game = game_server_state.env.game
    if player_id != game.current_player:
        return {"message": f"It is not P{player_id}'s turn."}
    
    return game.get_valid_actions_frontend(player_id)

# To run this file: uvicorn main:app --reload