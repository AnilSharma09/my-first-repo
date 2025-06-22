
from collections import defaultdict

def topological_sort(graph, V):
    visited = [False] * V
    stack = []

    def dfs(v):
        visited[v] = True
        for neighbour in graph[v]:
            if not visited[neighbour]:
                dfs(neighbour)
        stack.append(v)

    for i in range(V):
        if not visited[i]:
            dfs(i)

    return stack[::-1]

# Example usage
graph = defaultdict(list)
graph[0].extend([1, 2])
graph[1].append(3)
graph[2].append(3)
V = 4
print("Topological Sort:", topological_sort(graph, V))

