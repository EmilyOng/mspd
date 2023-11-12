def get_nearest_neighbours(points):
    N = len(points)

    upper_right = [float("inf") for i in range(N)]
    upper_left = [float("-inf") for i in range(N)]
    lower_right = [float("inf") for i in range(N)]
    lower_left = [float("-inf") for i in range(N)]

    nearest_neighbours = [[] for i in range(N)]

    # Sort the points by y-coordinates (point_index = ((x, y), i))
    sorted_points = sorted(points, key=lambda point_index: point_index[0][1])

    for i in range(N):
        a_index = sorted_points[i][1]
        a_x = sorted_points[i][0][0]

        for j in range(i):
            b_index = sorted_points[j][1]
            b_x = sorted_points[j][0][0]

            if b_x <= a_x and a_x < upper_right[b_index]:
                upper_right[b_index] = a_x
                nearest_neighbours[b_index].append(a_index)
            elif upper_left[b_index] < a_x and a_x < b_x:
                upper_left[b_index] = a_x
                nearest_neighbours[b_index].append(a_index)

        for j in range(i - 1, -1, -1):
            b_index = sorted_points[j][1]
            b_x = sorted_points[j][0][0]
            if a_x <= b_x and b_x < lower_right[a_index]:
                lower_right[a_index] = b_x
                nearest_neighbours[a_index].append(b_index)
            elif lower_left[a_index] < b_x and b_x < a_x:
                lower_left[a_index] = b_x
                nearest_neighbours[a_index].append(b_index)

    return nearest_neighbours


def dfs(at, dist, adj_list, points, subtree, count, parent_count, pathlengths):
    subtree[at].add(at)
    pathlengths[at] = dist
    parent_count[at] += 1
    count[at] += 1

    for node in adj_list[at]:
        if node == at:
            continue
        dfs(node, dist + manhattan_distance(
            points[at][0][0], points[at][0][1],
            points[node][0][0], points[node][0][1]
        ), adj_list, points, subtree, count, parent_count, pathlengths)
        for v in subtree[node]:
            subtree[at].add(v)
        count[at] += count[node]


def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)



def calc_wl(adj_list, points):
    total = 0
    for i in range(len(points)):
        for node in adj_list[i]:
            total += manhattan_distance(
                points[i][0][0], points[i][0][1],
                points[node][0][0], points[node][0][1]
            )
    return total



def calc_skew(adj_list, points, N):
    parent_count = [0 for i in range(len(points))]
    count = [0 for i in range(len(points))]
    pathlengths = [0 for i in range(len(points))]

    subtree = [set() for i in range(len(points))]

    dfs(0, 0, adj_list, points, subtree, count, parent_count, pathlengths)

    mx = 0
    mn = 1000000000

    for i in range(1, N):
        mx = max(mx, pathlengths[i])
        mn = min(mn, pathlengths[i])

    return mx - mn


def steinerize(points, adj_list, parents):
    N = len(points)

    def best_steiner_node(index):
        a, b = 0, 0
        best = 0
        b_x, b_y = 0, 0

        for node_a in adj_list[index]:
            if node_a == index:
                continue

            for node_b in adj_list[index]:
                if node_b == index or node_a == node_b:
                    continue

            s_x = points[index][0]
            s_y = points[index][1]

            if min(points[node_a][0], points[node_b][0]) > points[index][0]:
                s_x = min(points[node_a][0], points[node_b][0])
            elif max(points[node_a][0], points[node_b][0]) < points[index][0]:
                s_x = max(points[node_a][0], points[node_b][0])

            if min(points[node_a][1], points[node_b][1]) > points[index][1]:
                s_y = min(points[node_a][1], points[node_b][1])
            elif max(points[node_a][1], points[node_b][1]) < points[index][1]:
                s_y = max(points[node_a][1], points[node_b][1])

            gain = abs(points[index][0] - s_x) + abs(points[index][1] - s_y)
            if gain > best:
                best = gain
                a, b = node_a, node_b
                b_x, b_y = s_x, s_y

        return (-best, index, a, b, b_x, b_y)


    pq = set()
    steiner_nodes = [[] for i in range(len(points))]
    for i in range(1, len(points)):
        new_node = best_steiner_node(i)
        pq.add(new_node)
        steiner_nodes[i] = new_node

    while len(pq) > 0:
        curr_node = next(iter(pq))
        pq.remove(curr_node)

        if curr_node[0] == 0:
            break

        a, b = curr_node[2], curr_node[3]
        steiner_node = N

        if points[a][0] == curr_node[4] and points[a][1] == curr_node[5]:
            steiner_node = a
        elif points[b][0] == curr_node[4] and points[b][1] == curr_node[5]:
            steiner_node = b
        else:
            points.append(((curr_node[4], curr_node[5]), steiner_node))
            points[steiner_node][0] = curr_node[4]
            points[steiner_node][1] = curr_node[5]

        if a != steiner_node:
            adj_list[curr_node[1]].remove(a)
            adj_list[curr_node[1]].add(steiner_node)
            parents[steiner_node] = curr_node[1]
            adj_list[steiner_node].add(a)
            parents[a] = steiner_node
            parents[steiner_node] = curr_node[1]
        if b != steiner_node:
            adj_list[curr_node[1]].remove(b)
            adj_list[curr_node[1]].add(steiner_node)
            parents[steiner_node] = curr_node[1]
            adj_list[steiner_node].add(b)
            parents[b] = steiner_node
            parents[steiner_node] = curr_node[1]

        steiner_nodes[curr_node[1]] = best_steiner_node(curr_node[1])
        pq.add(steiner_nodes[curr_node[1]])

        pq.discard(steiner_nodes[a])
        pq.discard(steiner_nodes[b])

        steiner_nodes[a] = best_steiner_node(a)
        pq.add(steiner_nodes[a])

        steiner_nodes[b] = best_steiner_node(b)
        pq.add(steiner_nodes[b])



def das(source_set, adj_list, parents, N, points): # Detour-Aware Steinerization
    nearest_neighbours = get_nearest_neighbours(points)

    subtree = [set() for i in range(len(points))]

    parent_count = [0 for i in range(len(points))]
    count = [0 for i in range(len(points))]
    pathlengths = [0 for i in range(len(points))]

    DT = [0 for i in range(len(points))]

    def get_dt(at, parent):
        DT[at] = pathlengths[at] - manhattan_distance(
            points[at][0][0], points[at][0][1],
            0, 0
        )
        for node in adj_list[at]:
            if node == at:
                continue
            get_dt(node, at)


    dfs(0, 0, adj_list, points, subtree, count, parent_count, pathlengths)

    max_pathlength = 0
    for i in range(N):
        max_pathlength = max(max_pathlength, pathlengths[i])

    for i in range(1, len(points)):
        if i in source_set:
            continue

        edge_length = manhattan_distance(
            points[parents[i]][0][0], points[parents[i]][0][1],
            points[i][0][0], points[i][0][1]
        )

        best_nn = parents[i]

        if best_nn >= N:
            continue

        for node in nearest_neighbours[i]:
            if node == 0:
                continue

            distance = manhattan_distance(
                points[node][0][0], points[node][0][1],
                points[i][0][0], points[i][0][1]
            )
            if distance <= edge_length and pathlengths[i] <= 0.5 * max_pathlength:
                if node not in subtree[i]:
                    best_nn = node
                    edge_length = distance

        adj_list[parents[i]].remove(i)
        parents[i] = best_nn
        adj_list[parents[i]].add(i)

        for j in range(len(points)):
            subtree[j].clear()
            pathlengths[j] = 0
            parent_count[j] = 0
            count[j] = 0

        dfs(0, 0, adj_list, points, subtree, count, parent_count, pathlengths)

    for i in range(len(points)):
        subtree[i].clear()
        parent_count[i] = 0
        count[i] = 0
        pathlengths[i] = 0

    dfs(0, 0, adj_list, points, subtree, count, parent_count, pathlengths)
    get_dt(0, 0)

    max_pathlength = 0
    curr_dt = 0
    curr_wl = calc_wl(adj_list, points)

    for i in range(N):
        max_pathlength = max(max_pathlength, pathlengths[i])
    for i in range(len(points)):
        curr_dt += DT[i]

    for i in range(1, len(points)):
        if i in source_set:
            continue

        new_parent = parents[i]
        if new_parent >= N:
            continue

        for node in nearest_neighbours[i]:
            if node == 0:
                continue

            new_wl = curr_wl - manhattan_distance(
                points[i][0][0], points[i][0][1],
                points[parents[i]][0][0], points[parents[i]][0][1]
            ) + manhattan_distance(
                points[i][0][0], points[i][0][1],
                points[node][0][0], points[node][0][1]
            )
            new_dt = curr_dt - manhattan_distance(
                points[i][0][0], points[i][0][1],
                points[parents[i]][0][0], points[parents[i]][0][1]
            ) * count[i] - manhattan_distance(
                points[i][0][0], points[i][0][1],
                points[node][0][0], points[node][0][1]
            ) * count[i]

            if new_wl <= curr_wl and new_dt <= curr_dt:
                if node not in subtree[i]:
                    curr_wl = new_wl
                    curr_dt = new_dt
                    new_parent = node

        adj_list[parents[i]].remove(i)
        parents[i] = new_parent
        adj_list[parents[i]].add(i)

        for j in range(len(points)):
            subtree[j].clear()
            parent_count[j] = 0
            count[j] = 0
            pathlengths[j] = 0

        dfs(0, 0, adj_list, points, subtree, count, parent_count, pathlengths)
        get_dt(0, 0)

    for i in range(len(points)):
        subtree[i].clear()
        parent_count[i] = 0
        count[i] = 0
        pathlengths[i] = 0

    dfs(0, 0, adj_list, points, subtree, count, parent_count, pathlengths)
    for i in range(len(points)):
        if parent_count[i] != 1:
            print("error")



def prim_dijkstra(alpha, points, source_set, nearest_neighbours, T):
    N = len(points)

    keys = [float("inf") for i in range(len(points))]
    pathlengths = [float("inf") for i in range(len(points))]
    parents = [0 for i in range(len(points))]
    visited = [False for i in range(len(points))]

    keys[0] = 0
    pathlengths[0] = 0
    parents[0] = 0

    pq = set()

    for source in source_set:
        keys[source] = 0
        pathlengths[source] = 0
        parents[source] = 0
        pq.add(((keys[source], pathlengths[source]), (source, 0)))

    if len(source_set) == 0:
        pq.add(((keys[0], pathlengths[0]), (0, 0)))
    else:
        visited[0] = True


    while len(pq) > 0:
        fr = next(iter(pq))
        pq.remove(fr)

        fr_index = fr[1][0]
        fr_pathlength = fr[0][1]

        visited[fr_index] = True

        for neighbour_index in nearest_neighbours[fr_index]:
            distance = manhattan_distance(
                points[fr_index][0][0], points[fr_index][0][1],
                points[neighbour_index][0][0], points[neighbour_index][0][1]
            )
            pathlength = distance + fr_pathlength
            weight = alpha * fr_pathlength + distance

            if not visited[neighbour_index] and weight <= keys[fr_pathlength]:
                pq.discard((
                    (keys[neighbour_index], pathlengths[neighbour_index]),
                    (neighbour_index, parents[neighbour_index])
                ))
                keys[neighbour_index] = weight
                pathlengths[neighbour_index] = pathlength
                parents[neighbour_index] = fr_index
                pq.add((
                    (keys[neighbour_index], pathlengths[neighbour_index]),
                    (neighbour_index, parents[neighbour_index])
                ))

    adj_list = [set() for i in range(N)]
    for i in range(N):
        adj_list[parents[i]].add(i)

    if T:
        steinerize(points, adj_list, parents)
        das(source_set, adj_list, parents, N, points)

    return adj_list, parents


def solve(N, source_set, input_df):
    # Store the x and y coordinates
    X = input_df[[f"x{i}" for i in range(N)]].values[0].tolist()
    Y = input_df[[f"y{i}" for i in range(N)]].values[0].tolist()

    points = [((X[i], Y[i]), i) for i in range(N)]

    nearest_neighbours = get_nearest_neighbours(points)
    alpha_values = [0.2, 0.4, 0.6, 0.8]

    best = None

    for alpha in alpha_values:
        # Calculate the normalized values
        n_adj_list, _ = prim_dijkstra(alpha, points, [], nearest_neighbours, False)
        n_wl = calc_wl(n_adj_list, points)
        n_skew = calc_skew(n_adj_list, points, N)

        adj_list, _ = prim_dijkstra(alpha, points, source_set, nearest_neighbours, True)
        wl = calc_wl(adj_list, points) / n_wl
        skew = calc_skew(adj_list, points, N) / n_skew

        if best is None:
            best = (wl, skew)
        else:
            if wl >= best[0] and skew >= best[1]:
                best = (wl, skew)

    return best
