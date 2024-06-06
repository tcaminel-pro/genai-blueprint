from langserve import RemoteRunnable

chain = RemoteRunnable(
    "http://localhost:8000/joke/c/N4Igxg9gdgZglgcwK4CcCGAjANgUxALlCywFsCRi0S0BmAfQA46sIw0sQBfToA"
)
result = chain.invoke({"topic": "bear"})
print(result)
