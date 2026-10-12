
from langsmith import traceable


@traceable(name="healthcare_rag_connection_test")
def test_trace():
    return {
        "status": "success",
        "message": "LangSmith tracing is working."
    }


if __name__ == "__main__":
    result = test_trace()
    print(result)
