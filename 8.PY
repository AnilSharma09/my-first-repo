import heapq

def prims_algorithm(graph, V):
    visited = [False] * V
    min_heap = [(0, 0)]  # (cost, vertex)
    mst_cost = 0
    edges_used = []

    while min_heap:
        cost, u = heapq.heappop(min_heap)
        if visited[u]:
            continue
        visited[u] = True
        mst_cost += cost
        for v, weight in graph[u]:
            if not visited[v]:
                heapq.heappush(min_heap, (weight, v))
                edges_used.append((u, v, weight))

    print("Edges in MST:")
    for e in edges_used:
        print(e)
    print("Total Minimum Cost:", mst_cost)

# Graph as adjacency list
graph = {
    0: [(1, 2), (3, 6)],
    1: [(0, 2), (2, 3), (3, 8), (4, 5)],
    2: [(1, 3), (4, 7)],
    3: [(0, 6), (1, 8), (4, 9)],
    4: [(1, 5), (2, 7), (3, 9)]
}
V = 5
prims_algorithm(graph, V)

