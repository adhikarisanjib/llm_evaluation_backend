from pydantic import BaseModel, ConfigDict, Field, field_validator


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class ModelCreate(BaseModel):
    name: str
    display_name: str
    adapter_type: str
    base_url: str
    api_key_env: str | None = None
    metadata_json: dict = Field(default_factory=dict)
    is_enabled: bool = True


class ModelRead(ORMModel):
    id: int
    name: str
    display_name: str
    adapter_type: str
    base_url: str
    api_key_env: str | None
    metadata_json: dict
    is_enabled: bool


class TaskCreate(BaseModel):
    id: int | None = Field(default=None, ge=1)
    title: str = Field(min_length=1)
    question: str = Field(min_length=1)
    starter_code: str | None = None
    difficulty: str | None = None
    task_type: str
    tests: list[str] = Field(default_factory=list)
    regression_tests: list[str] = Field(default_factory=list)
    programming_language: str = "python"

    @field_validator("task_type")
    @classmethod
    def validate_task_type(cls, value: str) -> str:
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "generation": "code_generation",
            "codegeneration": "code_generation",
            "code_generation": "code_generation",
            "debug": "debugging",
            "debugging": "debugging",
        }
        if normalized not in aliases:
            raise ValueError("task_type must be code_generation or debugging")
        return aliases[normalized]

    @field_validator("tests")
    @classmethod
    def tests_must_not_be_empty(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        if not cleaned:
            raise ValueError("at least one test is required")
        return cleaned

    @field_validator("regression_tests")
    @classmethod
    def clean_regression_tests(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item and item.strip()]

    @field_validator("programming_language")
    @classmethod
    def normalize_language(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("programming_language is required")
        return normalized


class TaskRead(ORMModel):
    id: int
    title: str = Field(min_length=1)
    question: str = Field(min_length=1)
    starter_code: str | None
    difficulty: str | None
    task_type: str
    tests: list[str]
    regression_tests: list[str]
    programming_language: str


class TaskExcelUploadResult(BaseModel):
    imported_count: int
    task_ids: list[int]


class ExperimentCreate(BaseModel):
    name: str
    model_ids: list[int]
    task_ids: list[int]
    attempts_per_task: int = Field(default=1, ge=1, le=100)
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 1024


class ExperimentCloneRequest(BaseModel):
    name: str | None = None
    attempts_per_task: int | None = Field(default=None, ge=1, le=100)


class ExperimentRead(ORMModel):
    id: int
    name: str
    status: str
    attempts_per_task: int
    temperature: float
    top_p: float
    max_tokens: int


class ExperimentDetail(ExperimentRead):
    models: list[ModelRead] = Field(default_factory=list)
    tasks: list[TaskRead] = Field(default_factory=list)


class RunTestResultRead(ORMModel):
    id: int
    test_group: str
    test_index: int
    test_code: str
    passed: bool
    duration_ms: float | None
    error: str | None


class RunRead(ORMModel):
    id: int
    experiment_id: int
    model_id: int
    task_id: int
    attempt_number: int
    status: str
    prompt: str
    raw_response: str | None
    extracted_code: str | None
    latency_ms: float | None
    tokens_per_second: float | None
    input_tokens: int | None
    output_tokens: int | None
    tests_total: int
    tests_passed: int
    all_tests_passed: bool
    bug_fixed: bool | None
    regression_count: int | None
    error: str | None
    test_results: list[RunTestResultRead] = Field(default_factory=list)
