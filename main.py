import os
import uuid
import time
import tracemalloc
from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from database import db, create_document, get_documents
from schemas import Strategy, Generation, GameResult
import random

app = FastAPI(title="TicTacToe AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Tic-Tac-Toe Core
# -------------------------
Board = List[str]  # length 9, values in {"X", "O", ""}

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6)              # diags
]

DEFAULT_WEIGHTS = {
    "win": 100.0,
    "block_win": 90.0,
    "fork": 15.0,
    "block_fork": 14.0,
    "center": 3.0,
    "corner": 2.0,
    "line2": 1.5,
    "line1": 0.5,
}

class MoveRequest(BaseModel):
    board: Board
    ai_symbol: str = Field("O", pattern="^[XO]$")
    human_symbol: str = Field("X", pattern="^[XO]$")
    use_strategy: Optional[str] = Field(None, description="Strategy name to use; if None, use best available or default")
    return_tree: bool = True

class MoveResponse(BaseModel):
    move: int
    board: Board
    winner: str
    metrics: Dict[str, float]
    nodes_explored: int
    depth: int
    tree: Optional[dict] = None

class GARequest(BaseModel):
    population_size: int = 30
    generations: int = 20
    mutation_rate: float = 0.2
    tournament_k: int = 3
    run_id: Optional[str] = None

class GAResponse(BaseModel):
    run_id: str
    generations: List[Generation]
    best_strategy: Strategy

# Helpers

def check_winner(board: Board) -> str:
    for a,b,c in WIN_LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    if all(cell for cell in board):
        return "draw"
    return ""


def available_moves(board: Board) -> List[int]:
    return [i for i,c in enumerate(board) if not c]


def evaluate(board: Board, ai: str, human: str, w: Dict[str, float]) -> float:
    # Terminal checks
    winner = check_winner(board)
    if winner == ai:
        return w.get("win", 100.0)
    if winner == human:
        return -w.get("win", 100.0)
    if winner == "draw":
        return 0.0

    score = 0.0
    # Center preference
    if board[4] == ai:
        score += w.get("center", 3.0)
    elif board[4] == human:
        score -= w.get("center", 3.0)

    # Corners
    corners = [0,2,6,8]
    score += sum(1 for i in corners if board[i] == ai) * w.get("corner", 2.0)
    score -= sum(1 for i in corners if board[i] == human) * w.get("corner", 2.0)

    # Line patterns
    for a,b,c in WIN_LINES:
        line = [board[a], board[b], board[c]]
        ai_count = line.count(ai)
        human_count = line.count(human)
        if ai_count and not human_count:
            if ai_count == 2:
                score += w.get("line2", 1.5)
            elif ai_count == 1:
                score += w.get("line1", 0.5)
        if human_count and not ai_count:
            if human_count == 2:
                score -= w.get("line2", 1.5)
            elif human_count == 1:
                score -= w.get("line1", 0.5)

    # Simple fork heuristics: two non-blocked 2-in-lines
    def potential_twos(sym: str) -> int:
        count = 0
        for a,b,c in WIN_LINES:
            line = [board[a], board[b], board[c]]
            if line.count(sym) == 2 and line.count("") == 1:
                count += 1
        return count
    score += potential_twos(ai) * w.get("fork", 15.0)
    score -= potential_twos(human) * w.get("block_fork", 14.0)

    return score


def minimax(board: Board, ai: str, human: str, w: Dict[str, float], is_max: bool, alpha: float, beta: float, depth: int = 0, tree: Optional[dict] = None) -> Tuple[float, Optional[int], int, int, dict]:
    winner = check_winner(board)
    nodes = 1
    max_depth = depth

    if winner:
        val = evaluate(board, ai, human, w)
        return val, None, nodes, max_depth, (tree or {})

    moves = available_moves(board)
    if not moves:
        val = evaluate(board, ai, human, w)
        return val, None, nodes, max_depth, (tree or {})

    best_move = None
    if is_max:
        value = float('-inf')
        for m in moves:
            board[m] = ai
            child = {"move": m, "children": []}
            val, _, child_nodes, child_depth, _ = minimax(board, ai, human, w, False, alpha, beta, depth+1, child)
            board[m] = ""
            nodes += child_nodes
            max_depth = max(max_depth, child_depth)
            if val > value:
                value, best_move = val, m
                if tree is not None:
                    (tree.setdefault("children", [])).append({"move": m, "value": val})
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return value, best_move, nodes, max_depth, (tree or {})
    else:
        value = float('inf')
        for m in moves:
            board[m] = human
            child = {"move": m, "children": []}
            val, _, child_nodes, child_depth, _ = minimax(board, ai, human, w, True, alpha, beta, depth+1, child)
            board[m] = ""
            nodes += child_nodes
            max_depth = max(max_depth, child_depth)
            if val < value:
                value, best_move = val, m
                if tree is not None:
                    (tree.setdefault("children", [])).append({"move": m, "value": val})
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value, best_move, nodes, max_depth, (tree or {})


def get_strategy_weights(name: Optional[str]) -> Dict[str, float]:
    if name:
        doc = db["strategy"].find_one({"name": name})
        if doc:
            return {**DEFAULT_WEIGHTS, **doc.get("weights", {})}
    # try best by fitness
    best = db["strategy"].find_one(sort=[("fitness", -1)])
    if best:
        return {**DEFAULT_WEIGHTS, **best.get("weights", {})}
    return DEFAULT_WEIGHTS

# -------------------------
# API Endpoints
# -------------------------

@app.get("/")
def read_root():
    return {"message": "TicTacToe AI Backend Running"}

@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set",
        "database_name": "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        collections = db.list_collection_names()
        response["collections"] = collections[:10]
        response["database"] = "✅ Connected & Working"
        response["connection_status"] = "Connected"
    except Exception as e:
        response["database"] = f"⚠️  Issue: {str(e)[:80]}"
    return response

@app.post("/api/game/move", response_model=MoveResponse)
def api_game_move(req: MoveRequest):
    board = req.board.copy()
    ai = req.ai_symbol
    human = req.human_symbol
    weights = get_strategy_weights(req.use_strategy)

    start_time = time.perf_counter()
    tracemalloc.start()
    tree_container = {"root": {"children": []}} if req.return_tree else None
    val, move, nodes, depth, tree_info = minimax(board, ai, human, weights, True, float('-inf'), float('inf'), 0, tree_container["root"] if tree_container else None)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed = (time.perf_counter() - start_time) * 1000.0

    if move is None:
        # if terminal or no moves, return as is
        winner = check_winner(board)
        return MoveResponse(move=-1, board=board, winner=winner or "", metrics={"time_ms": elapsed, "mem_kb": peak/1024.0, "nodes": nodes}, nodes_explored=nodes, depth=depth, tree=tree_container)

    board[move] = ai
    winner = check_winner(board) or ""

    metrics = {
        "time_ms": round(elapsed, 3),
        "mem_kb": round(peak/1024.0, 3),
        "nodes": nodes,
        "eval": val,
    }

    return MoveResponse(move=move, board=board, winner=winner, metrics=metrics, nodes_explored=nodes, depth=depth, tree=tree_container)

# --------- GA ----------

def random_weights() -> Dict[str, float]:
    return {k: max(0.0, v + random.uniform(-0.5*v, 0.5*v)) for k,v in DEFAULT_WEIGHTS.items()}


def mutate(w: Dict[str, float], rate: float) -> Dict[str, float]:
    out = dict(w)
    for k in out:
        if random.random() < rate:
            out[k] = max(0.0, out[k] * (1.0 + random.uniform(-0.3, 0.3)))
    return out


def crossover(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    child = {}
    for k in DEFAULT_WEIGHTS.keys():
        child[k] = a[k] if random.random() < 0.5 else b[k]
        # small blend
        child[k] = 0.9*child[k] + 0.1*((a[k]+b[k])/2)
    return child


def play_match(weights: Dict[str,float], games: int = 4) -> float:
    # Fitness vs baseline (default heuristic). alternate first player.
    ai = 'O'
    human = 'X'
    wins = 0
    draws = 0
    losses = 0
    for g in range(games):
        board = [""]*9
        turn = 'X' if g % 2 == 0 else 'O'
        while True:
            winner = check_winner(board)
            if winner:
                if winner == 'draw':
                    draws += 1
                elif (winner == 'O' and turn == 'X') or (winner == 'X' and turn == 'O'):
                    # previous player won
                    if winner == 'O':
                        wins += 1
                    else:
                        losses += 1
                else:
                    if winner == 'O':
                        wins += 1
                    else:
                        losses += 1
                break
            if turn == 'X':
                # baseline uses DEFAULT_WEIGHTS as 'X'
                _, move, _, _, _ = minimax(board, 'X', 'O', DEFAULT_WEIGHTS, True, float('-inf'), float('inf'))
                if move is None:
                    move = random.choice(available_moves(board))
                board[move] = 'X'
                turn = 'O'
            else:
                # candidate plays as O using weights
                _, move, _, _, _ = minimax(board, 'O', 'X', weights, True, float('-inf'), float('inf'))
                if move is None:
                    move = random.choice(available_moves(board))
                board[move] = 'O'
                turn = 'X'
    return wins + 0.5*draws - 0.0*losses


@app.post("/api/ga/run", response_model=GAResponse)
def api_ga_run(req: GARequest):
    pop_size = max(6, req.population_size)
    gens = max(1, req.generations)
    rate = min(1.0, max(0.0, req.mutation_rate))
    k = max(2, req.tournament_k)
    run_id = req.run_id or str(uuid.uuid4())

    # init population
    population = [random_weights() for _ in range(pop_size)]
    # include default
    population[0] = DEFAULT_WEIGHTS.copy()

    history: List[Generation] = []

    prev_best = None
    for g in range(gens):
        fitnesses = [play_match(w, games=6) for w in population]
        best_idx = max(range(pop_size), key=lambda i: fitnesses[i])
        best = fitnesses[best_idx]
        mean = sum(fitnesses)/len(fitnesses)
        std = (sum((f-mean)**2 for f in fitnesses)/len(fitnesses))**0.5
        best_w = population[best_idx]

        # store generation stats
        gen_doc = Generation(
            run_id=run_id,
            generation=g,
            population_size=pop_size,
            best_fitness=best,
            mean_fitness=mean,
            std_fitness=std,
            best_weights=best_w,
            convergence=(0.0 if prev_best is None else (best - prev_best))
        )
        create_document('generation', gen_doc)
        history.append(gen_doc)

        # update best strategy collection
        create_document('strategy', Strategy(name=f"run-{run_id}-g{g}", weights=best_w, fitness=best, generation=g))

        # selection - tournament
        selected = []
        for _ in range(pop_size):
            cand = random.sample(list(zip(population, fitnesses)), k)
            selected.append(max(cand, key=lambda c: c[1])[0])

        # next generation via crossover + mutation
        new_pop = []
        for i in range(0, pop_size, 2):
            a = selected[i]
            b = selected[(i+1) % pop_size]
            child1 = mutate(crossover(a,b), rate)
            child2 = mutate(crossover(b,a), rate)
            new_pop.extend([child1, child2])
        population = new_pop[:pop_size]
        prev_best = best

    # final best
    final_fit = [play_match(w, games=8) for w in population]
    best_idx = max(range(pop_size), key=lambda i: final_fit[i])
    best_w = population[best_idx]
    best_strategy = Strategy(name=f"run-{run_id}-best", weights=best_w, fitness=final_fit[best_idx], generation=gens)
    create_document('strategy', best_strategy)

    return GAResponse(run_id=run_id, generations=history, best_strategy=best_strategy)


@app.get('/api/strategies')
def api_strategies():
    out = []
    for s in db['strategy'].find().sort('fitness', -1).limit(50):
        s['_id'] = str(s['_id'])
        out.append(s)
    return out

@app.get('/api/ga/generations')
def api_generations(run_id: Optional[str] = None):
    query = {"run_id": run_id} if run_id else {}
    out = []
    for g in db['generation'].find(query).sort('generation', 1):
        g['_id'] = str(g['_id'])
        out.append(g)
    return out

@app.post('/api/games/result')
def api_save_game(res: GameResult):
    create_document('gameresult', res)
    return {"status": "ok"}

@app.get('/api/metrics/summary')
def api_metrics_summary():
    # summarize best strategy, last generations
    best = db['strategy'].find_one(sort=[('fitness', -1)])
    gens = list(db['generation'].find().sort('generation', 1))
    for g in gens:
        g['_id'] = str(g['_id'])
    return {
        "best_strategy": {"name": best.get('name'), "fitness": best.get('fitness'), "weights": best.get('weights')} if best else None,
        "generations": gens[-20:],
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
