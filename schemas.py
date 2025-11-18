"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
These schemas are used for data validation in your application.

Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- User -> "user" collection
- Product -> "product" collection
- BlogPost -> "blogs" collection
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime

# Tic-Tac-Toe AI related schemas

class Strategy(BaseModel):
    """
    GA-evolved evaluation strategy weights for Tic-Tac-Toe.
    Collection: "strategy"
    """
    name: str = Field(..., description="Strategy name or label")
    weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Evaluation weights e.g., line2, center, corner, block, fork"
    )
    fitness: float = Field(0.0, description="Fitness score from GA")
    generation: int = Field(0, description="Generation index from GA run")

class Generation(BaseModel):
    """
    Stores GA generation statistics.
    Collection: "generation"
    """
    run_id: str = Field(..., description="Identifier for a GA run")
    generation: int = Field(..., description="Generation number")
    population_size: int = Field(..., description="Population size")
    best_fitness: float = Field(..., description="Best fitness this generation")
    mean_fitness: float = Field(..., description="Mean fitness")
    std_fitness: float = Field(..., description="Std dev of fitness")
    best_weights: Dict[str, float] = Field(default_factory=dict)
    convergence: float = Field(0.0, description="Improvement over last N gens")

class GameResult(BaseModel):
    """
    Stores results for played games for evaluation.
    Collection: "gameresult"
    """
    game_id: str = Field(...)
    player_symbol: str = Field(..., description="'X' or 'O' for human")
    winner: str = Field(..., description="'X', 'O', or 'draw'")
    moves: List[int] = Field(default_factory=list)
    ai_strategy: Optional[str] = Field(None, description="Strategy name used by AI")
    metrics: Dict[str, float] = Field(default_factory=dict, description="timing, memory, nodes, depth")

# Example schemas kept for reference
class User(BaseModel):
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    address: str = Field(..., description="Address")
    age: Optional[int] = Field(None, ge=0, le=120, description="Age in years")
    is_active: bool = Field(True, description="Whether user is active")

class Product(BaseModel):
    title: str = Field(..., description="Product title")
    description: Optional[str] = Field(None, description="Product description")
    price: float = Field(..., ge=0, description="Price in dollars")
    category: str = Field(..., description="Product category")
    in_stock: bool = Field(True, description="Whether product is in stock")
