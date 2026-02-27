from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from conductor.gh_sync import detect_phase, parse_blockers


@dataclass
class DAGNode:
    """A node in the dependency graph."""

    number: int
    title: str
    blocked_by: list[int] = field(default_factory=list)
    phase: str = "pending"


class CycleError(Exception):
    """Raised when a dependency cycle is detected."""

    def __init__(self, cycle: list[int]) -> None:
        self.cycle = cycle
        super().__init__(f"Dependency cycle detected: {' -> '.join(map(str, cycle))}")


class DAG:
    def __init__(self) -> None:
        self._nodes: dict[int, DAGNode] = {}

    def add_node(
        self,
        number: int,
        title: str,
        blocked_by: list[int] | None = None,
        phase: str = "pending",
    ) -> None:
        """Add or update a node in the DAG."""
        self._nodes[number] = DAGNode(
            number=number,
            title=title,
            blocked_by=blocked_by or [],
            phase=phase,
        )

    def get_node(self, number: int) -> DAGNode | None:
        return self._nodes.get(number)

    @property
    def nodes(self) -> list[DAGNode]:
        """All nodes sorted by number."""
        return sorted(self._nodes.values(), key=lambda n: n.number)

    def dependents(self, number: int) -> list[int]:
        """Issues that depend on (are blocked by) this issue."""
        return sorted(n.number for n in self._nodes.values() if number in n.blocked_by)

    def is_blocked(self, number: int, completed: set[int] | None = None) -> bool:
        """Check if an issue is blocked. Uses completed set to filter resolved deps."""
        node = self._nodes.get(number)
        if node is None:
            return False
        resolved = completed or set()
        return any(b not in resolved for b in node.blocked_by)

    def ready_issues(self, completed: set[int] | None = None) -> list[DAGNode]:
        """Return issues whose blockers are all resolved (or have no blockers)."""
        resolved = completed or set()
        return [
            n
            for n in self.nodes
            if n.number not in resolved and not self.is_blocked(n.number, resolved)
        ]

    def topological_sort(self) -> list[int]:
        """Return issue numbers in dependency order. Raises CycleError on cycles."""
        in_degree: dict[int, int] = {n: 0 for n in self._nodes}
        for node in self._nodes.values():
            for dep in node.blocked_by:
                if dep in self._nodes:
                    in_degree[node.number] += 1

        queue: deque[int] = deque(sorted(n for n, d in in_degree.items() if d == 0))
        result: list[int] = []

        while queue:
            current = queue.popleft()
            result.append(current)
            for dependent in self.dependents(current):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(self._nodes):
            cycle = self._find_cycle(in_degree)
            raise CycleError(cycle)

        return result

    def _find_cycle(self, in_degree: dict[int, int]) -> list[int]:
        """Extract one cycle from remaining nodes with non-zero in-degree."""
        remaining = {n for n, d in in_degree.items() if d > 0}
        if not remaining:
            return []

        start = next(iter(remaining))
        visited: dict[int, int] = {}
        current = start
        step = 0
        while current not in visited:
            visited[current] = step
            step += 1
            node = self._nodes[current]
            next_node = next((b for b in node.blocked_by if b in remaining), current)
            current = next_node

        cycle_start = current
        cycle = [cycle_start]
        node = self._nodes[cycle_start]
        nxt = next((b for b in node.blocked_by if b in remaining), cycle_start)
        while nxt != cycle_start:
            cycle.append(nxt)
            node = self._nodes[nxt]
            nxt = next((b for b in node.blocked_by if b in remaining), cycle_start)
        cycle.append(cycle_start)
        return cycle

    def execution_tiers(self) -> list[list[int]]:
        """Group issues into tiers for parallel execution.

        Tier N depends only on Tier <N.
        """
        order = self.topological_sort()
        depth: dict[int, int] = {}

        for num in order:
            node = self._nodes[num]
            deps_in_dag = [b for b in node.blocked_by if b in self._nodes]
            if not deps_in_dag:
                depth[num] = 0
            else:
                depth[num] = max(depth[b] for b in deps_in_dag) + 1

        if not depth:
            return []

        max_depth = max(depth.values())
        tiers: list[list[int]] = [[] for _ in range(max_depth + 1)]
        for num in order:
            tiers[depth[num]].append(num)

        return tiers


def build_dag_from_issues(issues: list[dict]) -> DAG:
    """Build DAG from issue data dicts.

    Each dict should have: number, title, body (for blocker parsing),
    labels (for phase detection).
    """
    dag = DAG()
    for issue in issues:
        blockers = parse_blockers(issue.get("body", ""))
        phase = detect_phase(issue.get("labels", []))
        dag.add_node(
            number=issue["number"],
            title=issue["title"],
            blocked_by=blockers,
            phase=phase,
        )
    return dag
