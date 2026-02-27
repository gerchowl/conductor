"""Pydantic models defining the inter-step contracts for the conductor pipeline.

Each model represents the data exchanged between pipeline stages (triage,
planning, test-matrix generation, stub scaffolding, implementation). Models
are designed for JSON serialization so they can be passed as files between
agent invocations.
"""

from __future__ import annotations

from pydantic import BaseModel


class BlockerStatus(BaseModel):
    number: int
    resolved: bool


class PlanTask(BaseModel):
    id: str
    description: str
    files: list[str]
    verification: str


class ImplementationPlan(BaseModel):
    tasks: list[PlanTask]


class IssueContext(BaseModel):
    number: int
    title: str
    body: str
    labels: list[str]
    phase: str
    blocked_by: list[BlockerStatus]
    branch: str
    design: str | None = None
    plan: ImplementationPlan | None = None


class TestCategory(BaseModel):
    name: str
    applies: bool
    reasoning: str


class TestMatrixEntry(BaseModel):
    task_id: str
    function: str
    file: str
    categories: list[TestCategory]


class TestMatrix(BaseModel):
    entries: list[TestMatrixEntry]


class StubFunction(BaseModel):
    name: str
    docstring: str
    signature: str


class StubFile(BaseModel):
    path: str
    functions: list[StubFunction]


class StubManifest(BaseModel):
    test_files: list[StubFile]
    impl_files: list[StubFile]


class TestAssignment(BaseModel):
    test_file: str
    stubs: list[StubFunction]
    related_impl_stubs: list[StubFile]


class ImplAssignment(BaseModel):
    impl_file: str
    stubs: list[StubFunction]
    related_test_content: str
    test_output: str


class FileOutput(BaseModel):
    file: str
    content: str
