from __future__ import annotations

import pytest

from conductor.dag import DAG, CycleError, build_dag_from_issues


class TestAddAndGetNode:
    def test_add_and_retrieve(self) -> None:
        dag = DAG()
        dag.add_node(1, "First issue")
        node = dag.get_node(1)
        assert node is not None
        assert node.number == 1
        assert node.title == "First issue"
        assert node.blocked_by == []
        assert node.phase == "pending"

    def test_add_with_blockers_and_phase(self) -> None:
        dag = DAG()
        dag.add_node(2, "Second", blocked_by=[1], phase="design")
        node = dag.get_node(2)
        assert node is not None
        assert node.blocked_by == [1]
        assert node.phase == "design"

    def test_update_existing_node(self) -> None:
        dag = DAG()
        dag.add_node(1, "Original")
        dag.add_node(1, "Updated", blocked_by=[5])
        node = dag.get_node(1)
        assert node is not None
        assert node.title == "Updated"
        assert node.blocked_by == [5]

    def test_get_nonexistent(self) -> None:
        dag = DAG()
        assert dag.get_node(999) is None

    def test_nodes_sorted_by_number(self) -> None:
        dag = DAG()
        dag.add_node(3, "Third")
        dag.add_node(1, "First")
        dag.add_node(2, "Second")
        numbers = [n.number for n in dag.nodes]
        assert numbers == [1, 2, 3]


class TestDependents:
    def test_single_dependent(self) -> None:
        dag = DAG()
        dag.add_node(1, "Base")
        dag.add_node(2, "Depends on 1", blocked_by=[1])
        assert dag.dependents(1) == [2]

    def test_multiple_dependents(self) -> None:
        dag = DAG()
        dag.add_node(1, "Base")
        dag.add_node(2, "A", blocked_by=[1])
        dag.add_node(3, "B", blocked_by=[1])
        assert dag.dependents(1) == [2, 3]

    def test_no_dependents(self) -> None:
        dag = DAG()
        dag.add_node(1, "Standalone")
        assert dag.dependents(1) == []

    def test_dependents_of_unknown_node(self) -> None:
        dag = DAG()
        dag.add_node(1, "Only node")
        assert dag.dependents(999) == []


class TestIsBlocked:
    def test_not_blocked_no_deps(self) -> None:
        dag = DAG()
        dag.add_node(1, "Free")
        assert dag.is_blocked(1) is False

    def test_blocked_by_unresolved(self) -> None:
        dag = DAG()
        dag.add_node(1, "Base")
        dag.add_node(2, "Blocked", blocked_by=[1])
        assert dag.is_blocked(2) is True

    def test_unblocked_with_completed(self) -> None:
        dag = DAG()
        dag.add_node(1, "Base")
        dag.add_node(2, "Blocked", blocked_by=[1])
        assert dag.is_blocked(2, completed={1}) is False

    def test_partially_blocked(self) -> None:
        dag = DAG()
        dag.add_node(1, "A")
        dag.add_node(2, "B")
        dag.add_node(3, "C", blocked_by=[1, 2])
        assert dag.is_blocked(3, completed={1}) is True
        assert dag.is_blocked(3, completed={1, 2}) is False

    def test_unknown_node_not_blocked(self) -> None:
        dag = DAG()
        assert dag.is_blocked(999) is False


class TestReadyIssues:
    def test_all_ready_no_deps(self) -> None:
        dag = DAG()
        dag.add_node(1, "A")
        dag.add_node(2, "B")
        ready = dag.ready_issues()
        assert [n.number for n in ready] == [1, 2]

    def test_ready_with_completion(self) -> None:
        dag = DAG()
        dag.add_node(1, "A")
        dag.add_node(2, "B", blocked_by=[1])
        dag.add_node(3, "C", blocked_by=[1])

        assert [n.number for n in dag.ready_issues()] == [1]
        assert [n.number for n in dag.ready_issues(completed={1})] == [2, 3]

    def test_completed_issues_excluded(self) -> None:
        dag = DAG()
        dag.add_node(1, "A")
        dag.add_node(2, "B", blocked_by=[1])
        ready = dag.ready_issues(completed={1})
        assert 1 not in [n.number for n in ready]

    def test_chain_progression(self) -> None:
        dag = DAG()
        dag.add_node(1, "A")
        dag.add_node(2, "B", blocked_by=[1])
        dag.add_node(3, "C", blocked_by=[2])

        assert [n.number for n in dag.ready_issues()] == [1]
        assert [n.number for n in dag.ready_issues(completed={1})] == [2]
        assert [n.number for n in dag.ready_issues(completed={1, 2})] == [3]


class TestTopologicalSort:
    def test_linear_chain(self) -> None:
        dag = DAG()
        dag.add_node(1, "A")
        dag.add_node(2, "B", blocked_by=[1])
        dag.add_node(3, "C", blocked_by=[2])
        result = dag.topological_sort()
        assert result.index(1) < result.index(2) < result.index(3)

    def test_diamond(self) -> None:
        dag = DAG()
        dag.add_node(1, "Root")
        dag.add_node(2, "Left", blocked_by=[1])
        dag.add_node(3, "Right", blocked_by=[1])
        dag.add_node(4, "Join", blocked_by=[2, 3])
        result = dag.topological_sort()
        assert result.index(1) < result.index(2)
        assert result.index(1) < result.index(3)
        assert result.index(2) < result.index(4)
        assert result.index(3) < result.index(4)

    def test_independent_issues(self) -> None:
        dag = DAG()
        dag.add_node(1, "A")
        dag.add_node(2, "B")
        dag.add_node(3, "C")
        result = dag.topological_sort()
        assert sorted(result) == [1, 2, 3]

    def test_cycle_raises_error(self) -> None:
        dag = DAG()
        dag.add_node(1, "A", blocked_by=[2])
        dag.add_node(2, "B", blocked_by=[1])
        with pytest.raises(CycleError) as exc_info:
            dag.topological_sort()
        assert 1 in exc_info.value.cycle
        assert 2 in exc_info.value.cycle

    def test_cycle_three_nodes(self) -> None:
        dag = DAG()
        dag.add_node(1, "A", blocked_by=[3])
        dag.add_node(2, "B", blocked_by=[1])
        dag.add_node(3, "C", blocked_by=[2])
        with pytest.raises(CycleError) as exc_info:
            dag.topological_sort()
        assert len(exc_info.value.cycle) >= 3

    def test_cycle_error_message(self) -> None:
        dag = DAG()
        dag.add_node(1, "A", blocked_by=[2])
        dag.add_node(2, "B", blocked_by=[1])
        with pytest.raises(CycleError, match="Dependency cycle detected"):
            dag.topological_sort()

    def test_external_blocker_ignored(self) -> None:
        dag = DAG()
        dag.add_node(1, "A")
        dag.add_node(2, "B", blocked_by=[1, 999])
        result = dag.topological_sort()
        assert result == [1, 2]


class TestExecutionTiers:
    def test_single_tier(self) -> None:
        dag = DAG()
        dag.add_node(1, "A")
        dag.add_node(2, "B")
        tiers = dag.execution_tiers()
        assert len(tiers) == 1
        assert sorted(tiers[0]) == [1, 2]

    def test_linear_chain(self) -> None:
        dag = DAG()
        dag.add_node(1, "A")
        dag.add_node(2, "B", blocked_by=[1])
        dag.add_node(3, "C", blocked_by=[2])
        tiers = dag.execution_tiers()
        assert tiers == [[1], [2], [3]]

    def test_diamond(self) -> None:
        dag = DAG()
        dag.add_node(1, "Root")
        dag.add_node(2, "Left", blocked_by=[1])
        dag.add_node(3, "Right", blocked_by=[1])
        dag.add_node(4, "Join", blocked_by=[2, 3])
        tiers = dag.execution_tiers()
        assert len(tiers) == 3
        assert tiers[0] == [1]
        assert sorted(tiers[1]) == [2, 3]
        assert tiers[2] == [4]

    def test_cycle_propagates(self) -> None:
        dag = DAG()
        dag.add_node(1, "A", blocked_by=[2])
        dag.add_node(2, "B", blocked_by=[1])
        with pytest.raises(CycleError):
            dag.execution_tiers()


class TestBuildDagFromIssues:
    def test_basic_build(self) -> None:
        issues = [
            {
                "number": 1,
                "title": "Setup",
                "body": "",
                "labels": ["phase:plan"],
            },
            {
                "number": 2,
                "title": "Implement",
                "body": "Blocked by: #1",
                "labels": ["phase:design"],
            },
        ]
        dag = build_dag_from_issues(issues)
        assert len(dag.nodes) == 2

        node1 = dag.get_node(1)
        assert node1 is not None
        assert node1.phase == "plan"
        assert node1.blocked_by == []

        node2 = dag.get_node(2)
        assert node2 is not None
        assert node2.phase == "design"
        assert node2.blocked_by == [1]

    def test_no_labels_defaults_pending(self) -> None:
        issues = [{"number": 1, "title": "X", "body": "", "labels": []}]
        dag = build_dag_from_issues(issues)
        node = dag.get_node(1)
        assert node is not None
        assert node.phase == "pending"

    def test_missing_body_and_labels(self) -> None:
        issues = [{"number": 1, "title": "Bare"}]
        dag = build_dag_from_issues(issues)
        node = dag.get_node(1)
        assert node is not None
        assert node.blocked_by == []
        assert node.phase == "pending"

    def test_multiple_blockers(self) -> None:
        issues = [
            {"number": 1, "title": "A", "body": "", "labels": []},
            {"number": 2, "title": "B", "body": "", "labels": []},
            {
                "number": 3,
                "title": "C",
                "body": "Blocked by: #1, #2",
                "labels": [],
            },
        ]
        dag = build_dag_from_issues(issues)
        node3 = dag.get_node(3)
        assert node3 is not None
        assert node3.blocked_by == [1, 2]


class TestEmptyDAG:
    def test_empty_nodes(self) -> None:
        dag = DAG()
        assert dag.nodes == []

    def test_empty_topological_sort(self) -> None:
        dag = DAG()
        assert dag.topological_sort() == []

    def test_empty_execution_tiers(self) -> None:
        dag = DAG()
        assert dag.execution_tiers() == []

    def test_empty_ready_issues(self) -> None:
        dag = DAG()
        assert dag.ready_issues() == []

    def test_empty_dependents(self) -> None:
        dag = DAG()
        assert dag.dependents(1) == []
