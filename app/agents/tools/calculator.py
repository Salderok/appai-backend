"""Safe arithmetic calculator tool (no eval())."""

from __future__ import annotations

import ast
import operator

from pydantic import BaseModel, Field

from app.agents.tools.base import BaseTool, ToolError

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.FloorDiv: operator.floordiv,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ToolError("Unsupported expression.")


class CalculatorArgs(BaseModel):
    expression: str = Field(description="Arithmetic expression, e.g. '2 * (3 + 4)'.")


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Evaluate a math expression and return the numeric result."
    args_model = CalculatorArgs

    async def run(self, args: CalculatorArgs) -> float:
        try:
            tree = ast.parse(args.expression, mode="eval")
            return float(_eval(tree))
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"Invalid expression: {exc}") from exc
