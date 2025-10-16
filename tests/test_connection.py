import pytest
from pytest import fixture, mark
from chaosllama.utils.utilities import get_spark_session


@fixture
def spark():
    return get_spark_session()


def test_spark_session(spark):
    assert spark is not None
    assert spark.version is not None
    print(f"Spark version: {spark.version}")

def test_spark_sql(spark):
    df = spark.createDataFrame(data=[(1, "Alice"), (2, "Bob")], schema=["id", "name"])
    assert df.count() == 2, f"Expected 2 rows, got {df.count()}"





if __name__ == "__main__":
    #pytest.main([__file__])
    test_spark_session(get_spark_session())