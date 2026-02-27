from __future__ import annotations

import json

import pytest

from conductor.models import (
    BlockerStatus,
    FileOutput,
    ImplAssignment,
    ImplementationPlan,
    IssueContext,
    PlanTask,
    StubFile,
    StubFunction,
    StubManifest,
    TestAssignment,
    TestCategory,
    TestMatrix,
    TestMatrixEntry,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_function() -> StubFunction:
    return StubFunction(
        name="do_thing",
        docstring="Does the thing.",
        signature="def do_thing(x: int) -> str",
    )


@pytest.fixture
def stub_file(stub_function: StubFunction) -> StubFile:
    return StubFile(path="src/conductor/engine.py", functions=[stub_function])


@pytest.fixture
def plan_task() -> PlanTask:
    return PlanTask(
        id="task-1",
        description="Create the engine module",
        files=["src/conductor/engine.py"],
        verification="pytest tests/test_engine.py passes",
    )


@pytest.fixture
def implementation_plan(plan_task: PlanTask) -> ImplementationPlan:
    return ImplementationPlan(tasks=[plan_task])


@pytest.fixture
def blocker() -> BlockerStatus:
    return BlockerStatus(number=42, resolved=False)


@pytest.fixture
def issue_context(
    blocker: BlockerStatus, implementation_plan: ImplementationPlan
) -> IssueContext:
    return IssueContext(
        number=7,
        title="Add engine",
        body="We need an engine.",
        labels=["enhancement", "priority:high"],
        phase="plan",
        blocked_by=[blocker],
        branch="feat/7-add-engine",
        design="Some design notes",
        plan=implementation_plan,
    )


@pytest.fixture
def test_category() -> TestCategory:
    return TestCategory(
        name="happy_path",
        applies=True,
        reasoning="Core functionality must be covered",
    )


@pytest.fixture
def test_matrix_entry(test_category: TestCategory) -> TestMatrixEntry:
    return TestMatrixEntry(
        task_id="task-1",
        function="test_engine_start",
        file="tests/test_engine.py",
        categories=[test_category],
    )


@pytest.fixture
def test_matrix(test_matrix_entry: TestMatrixEntry) -> TestMatrix:
    return TestMatrix(entries=[test_matrix_entry])


# ---------------------------------------------------------------------------
# JSON round-trip helpers
# ---------------------------------------------------------------------------


def assert_round_trip(instance):
    """Serialize to JSON and back, assert equality."""
    json_str = instance.model_dump_json()
    rebuilt = type(instance).model_validate_json(json_str)
    assert rebuilt == instance

    data = json.loads(json_str)
    from_dict = type(instance).model_validate(data)
    assert from_dict == instance


# ---------------------------------------------------------------------------
# Round-trip tests for every model
# ---------------------------------------------------------------------------


class TestBlockerStatusRoundTrip:
    def test_round_trip(self, blocker: BlockerStatus):
        assert_round_trip(blocker)

    def test_resolved_true(self):
        assert_round_trip(BlockerStatus(number=1, resolved=True))


class TestPlanTaskRoundTrip:
    def test_round_trip(self, plan_task: PlanTask):
        assert_round_trip(plan_task)

    def test_empty_files(self):
        assert_round_trip(PlanTask(id="t", description="d", files=[], verification="v"))


class TestImplementationPlanRoundTrip:
    def test_round_trip(self, implementation_plan: ImplementationPlan):
        assert_round_trip(implementation_plan)

    def test_empty_tasks(self):
        assert_round_trip(ImplementationPlan(tasks=[]))


class TestIssueContextRoundTrip:
    def test_full_round_trip(self, issue_context: IssueContext):
        assert_round_trip(issue_context)

    def test_minimal_defaults(self):
        ctx = IssueContext(
            number=1,
            title="t",
            body="b",
            labels=[],
            phase="triage",
            blocked_by=[],
            branch="main",
        )
        assert ctx.design is None
        assert ctx.plan is None
        assert_round_trip(ctx)

    def test_design_without_plan(self):
        ctx = IssueContext(
            number=2,
            title="t",
            body="b",
            labels=["bug"],
            phase="design",
            blocked_by=[BlockerStatus(number=10, resolved=True)],
            branch="fix/2-bug",
            design="design doc here",
        )
        assert ctx.plan is None
        assert_round_trip(ctx)


class TestTestCategoryRoundTrip:
    def test_applies_true(self, test_category: TestCategory):
        assert_round_trip(test_category)

    def test_applies_false(self):
        assert_round_trip(
            TestCategory(
                name="edge_cases",
                applies=False,
                reasoning="No meaningful edge cases identified",
            )
        )


class TestTestMatrixEntryRoundTrip:
    def test_round_trip(self, test_matrix_entry: TestMatrixEntry):
        assert_round_trip(test_matrix_entry)

    def test_multiple_categories(self):
        entry = TestMatrixEntry(
            task_id="task-2",
            function="test_parse",
            file="tests/test_parse.py",
            categories=[
                TestCategory(name="happy_path", applies=True, reasoning="yes"),
                TestCategory(name="edge_cases", applies=True, reasoning="empty input"),
                TestCategory(name="error_handling", applies=False, reasoning="n/a"),
            ],
        )
        assert_round_trip(entry)


class TestTestMatrixRoundTrip:
    def test_round_trip(self, test_matrix: TestMatrix):
        assert_round_trip(test_matrix)

    def test_empty(self):
        assert_round_trip(TestMatrix(entries=[]))


class TestStubFunctionRoundTrip:
    def test_round_trip(self, stub_function: StubFunction):
        assert_round_trip(stub_function)


class TestStubFileRoundTrip:
    def test_round_trip(self, stub_file: StubFile):
        assert_round_trip(stub_file)

    def test_empty_functions(self):
        assert_round_trip(StubFile(path="empty.py", functions=[]))


class TestStubManifestRoundTrip:
    def test_round_trip(self, stub_file: StubFile):
        manifest = StubManifest(test_files=[stub_file], impl_files=[stub_file])
        assert_round_trip(manifest)

    def test_empty_manifest(self):
        assert_round_trip(StubManifest(test_files=[], impl_files=[]))


class TestTestAssignmentRoundTrip:
    def test_round_trip(self, stub_function: StubFunction, stub_file: StubFile):
        assignment = TestAssignment(
            test_file="tests/test_engine.py",
            stubs=[stub_function],
            related_impl_stubs=[stub_file],
        )
        assert_round_trip(assignment)


class TestImplAssignmentRoundTrip:
    def test_round_trip(self, stub_function: StubFunction):
        assignment = ImplAssignment(
            impl_file="src/conductor/engine.py",
            stubs=[stub_function],
            related_test_content="def test_engine(): ...",
            test_output=(
                "FAILED tests/test_engine.py::test_engine - NotImplementedError"
            ),
        )
        assert_round_trip(assignment)


class TestFileOutputRoundTrip:
    def test_round_trip(self):
        assert_round_trip(
            FileOutput(file="src/conductor/engine.py", content="print('hello')")
        )

    def test_empty_content(self):
        assert_round_trip(FileOutput(file="empty.py", content=""))


# ---------------------------------------------------------------------------
# JSON schema tests
# ---------------------------------------------------------------------------


class TestJsonSchemas:
    @pytest.mark.parametrize(
        "model_cls",
        [
            IssueContext,
            ImplementationPlan,
            TestMatrix,
            StubManifest,
            TestAssignment,
            ImplAssignment,
        ],
    )
    def test_schema_is_valid_json(self, model_cls):
        schema = model_cls.model_json_schema()
        assert isinstance(schema, dict)
        raw = json.dumps(schema)
        reloaded = json.loads(raw)
        assert reloaded == schema

    def test_issue_context_schema_has_defs(self):
        schema = IssueContext.model_json_schema()
        assert "$defs" in schema or "properties" in schema

    def test_test_matrix_schema_references_categories(self):
        schema = TestMatrix.model_json_schema()
        raw = json.dumps(schema)
        assert "TestCategory" in raw

    def test_stub_manifest_schema_references_stub_function(self):
        schema = StubManifest.model_json_schema()
        raw = json.dumps(schema)
        assert "StubFunction" in raw


# ---------------------------------------------------------------------------
# Nested serialization depth tests
# ---------------------------------------------------------------------------


class TestNestedSerialization:
    def test_issue_context_nested_plan_tasks(self, issue_context: IssueContext):
        data = issue_context.model_dump()
        assert data["plan"]["tasks"][0]["id"] == "task-1"
        assert data["blocked_by"][0]["number"] == 42

    def test_test_matrix_nested_categories(self, test_matrix: TestMatrix):
        data = test_matrix.model_dump()
        cat = data["entries"][0]["categories"][0]
        assert cat["name"] == "happy_path"
        assert cat["applies"] is True

    def test_stub_manifest_nested_functions(self, stub_file: StubFile):
        manifest = StubManifest(test_files=[stub_file], impl_files=[])
        data = manifest.model_dump()
        assert data["test_files"][0]["functions"][0]["name"] == "do_thing"

    def test_deeply_nested_json_round_trip(self):
        """Full pipeline: IssueContext -> plan -> tasks -> files."""
        ctx = IssueContext(
            number=99,
            title="Deep nesting test",
            body="body",
            labels=["a", "b", "c"],
            phase="implement",
            blocked_by=[
                BlockerStatus(number=1, resolved=True),
                BlockerStatus(number=2, resolved=False),
            ],
            branch="feat/99-deep",
            design="design",
            plan=ImplementationPlan(
                tasks=[
                    PlanTask(
                        id=f"task-{i}",
                        description=f"Task {i}",
                        files=[f"file_{i}.py"],
                        verification=f"check {i}",
                    )
                    for i in range(5)
                ]
            ),
        )
        json_str = ctx.model_dump_json()
        rebuilt = IssueContext.model_validate_json(json_str)
        assert rebuilt == ctx
        assert len(rebuilt.plan.tasks) == 5
        assert rebuilt.blocked_by[1].resolved is False
