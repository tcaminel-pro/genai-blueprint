from beartype import beartype


@beartype
def process(items: list[dict]) -> bool:
    return all("id" in d for d in items)


process([{"id": 1}])  # OK
process([{"id": "x"}])  # raises clear BeartypeCallHintViolation

print("end")
