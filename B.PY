def transitive_closure(graph):
    V = len(graph)
    reach = [[graph[i][j] for j in range(V)] for i in range(V)]

    for k in range(V):
        for i in range(V):
            for j in range(V):
                reach[i][j] = reach[i][j] or (reach[i][k] and reach[k][j])

    return reach

# Example
graph = [
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1]
]
result = transitive_closure(graph)
print("Transitive Closure:")
for row in result:
    print(row)


