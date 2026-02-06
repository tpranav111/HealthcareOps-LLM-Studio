from dataclasses import dataclass


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    handler: callable


class ToolRegistry:
    def __init__(self, retriever=None):
        self._tools = {}
        self._retriever = retriever

    def register(self, tool):
        self._tools[tool.name] = tool

    def list_tools(self):
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "schema": tool.schema,
            }
            for tool in self._tools.values()
        ]

    def call(self, name, arguments):
        if name not in self._tools:
            return {"error": f"Unknown tool: {name}"}
        return self._tools[name].handler(arguments)


def _draft_prior_auth(args):
    return {
        "type": "prior_auth_request",
        "payer": args.get("payer", ""),
        "code": args.get("code", ""),
        "status": "draft",
        "notes": "Provide ordering provider NPI and clinical justification."
    }


def _lookup_policy(args, retriever):
    query = args.get("query", "")
    if not retriever:
        return {"results": []}
    results = retriever.retrieve(query, top_k=3)
    return {"results": [r["doc"] for r in results]}


def default_registry(retriever=None):
    registry = ToolRegistry(retriever=retriever)
    registry.register(
        Tool(
            name="draft_prior_auth",
            description="Draft a prior authorization request template.",
            schema={"type": "object", "properties": {"payer": {"type": "string"}, "code": {"type": "string"}}},
            handler=_draft_prior_auth,
        )
    )
    registry.register(
        Tool(
            name="lookup_policy",
            description="Search internal healthcare policy content.",
            schema={"type": "object", "properties": {"query": {"type": "string"}}},
            handler=lambda args: _lookup_policy(args, retriever),
        )
    )
    return registry
