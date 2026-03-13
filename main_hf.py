from dataclasses import dataclass
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from game import RandomAgent, LostCitiesEnv, ActionType

app = FastAPI()
env = LostCitiesEnv(RandomAgent(1))

@dataclass
class GameServerState:
    env: LostCitiesEnv

game_server_state = GameServerState(env=env)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def serve_index():
    return FileResponse("lost-cities.html")

class PlayAction(BaseModel):
    player_id: int
    card_index: int
    action_type: str
    draw_source: int

class OpponentPath(BaseModel):
    path: str

@app.get("/game/reset")
def reset_game():
    game_server_state.env.reset()
    return {"message": "Game reset successfully. New game started."}

@app.get("/game/opponent")
def get_opponent():
    import glob
    return [{"id": i, "name": path} for i, path in enumerate(glob.glob("model/*.pkt"))]

@app.get("/game/state/{player_id}")
def get_game_state(player_id: int):
    game = game_server_state.env.game
    if player_id not in [0, 1]:
        raise HTTPException(status_code=400, detail="Invalid player ID.")
    return game.get_public_state(player_id)

@app.post("/game/set_opponent")
def set_opponent(path: OpponentPath):
    import torch
    from agent import Agent
    agent_state_dict = torch.load(path.path, map_location="cpu")
    agent = Agent(device="cpu")
    agent.load_state_dict(agent_state_dict)
    game_server_state.env.set_opponent(agent)
    return {"message": f"Opponent set to model at {path.path}"}

@app.post("/game/play")
def handle_play_action(action: PlayAction):
    env = game_server_state.env
    game = env.game
    if game.game_over:
        return {"message": "Game is over. Reset to start a new game."}
    player_id = action.player_id
    assert player_id == 0
    if player_id != game.current_player:
        raise HTTPException(status_code=400,
            detail=f"It is not Player {player_id}'s turn.")
    try:
        card_to_play = game.hands[player_id][action.card_index]
        card_id = game.encode_card_to_id(*card_to_play)
        action_t = (card_id,
                    ActionType.Play.value if action.action_type == 'E' else ActionType.DISCARD.value,
                    action.draw_source)
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
    game = game_server_state.env.game
    if player_id != game.current_player:
        return {"message": f"It is not P{player_id}'s turn."}
    return game.get_valid_actions_frontend(player_id)
