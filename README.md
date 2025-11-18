# Adaptive Tic-Tac-Toe AI (Backend)

This backend provides the AI for a Tic-Tac-Toe application featuring:
- Minimax with Alpha-Beta pruning
- Genetic Algorithm (GA) to evolve evaluation weights for the heuristic function
- Metrics collection: execution time, memory, nodes explored, depth
- MongoDB persistence for strategies, generations, and game results

## API

- `GET /test` – backend and database health
- `POST /api/game/move` – compute AI move; returns new board, winner, metrics, and a shallow decision tree for visualization
- `POST /api/ga/run` – run GA to evolve strategy weights; persists generation stats and best strategy
- `GET /api/strategies` – list stored strategies
- `GET /api/ga/generations?run_id=...` – list GA generations
- `POST /api/games/result` – store a played game and metrics
- `GET /api/metrics/summary` – summary with best strategy and recent generations

## Running locally

- Ensure `DATABASE_URL` and `DATABASE_NAME` are set in environment; MongoDB is preconfigured in this environment.
- The server runs on port 8000.

