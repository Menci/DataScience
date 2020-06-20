from typing import Dict, List

# 图结构，使用邻接表存储
class Graph:
    # 节点 ID 到节点名的映射
    map_id_to_name: List[str] = []
    # 节点名到节点 ID 的映射
    map_name_to_id: Dict[str, int] = {}
    # 节点 ID 到「从该节点引出的所有边的目标节点 ID」的映射
    edges_from_node: List[List[int]] = []
    # 节点 ID 到「以该节点终止的所有边的源节点 ID」的映射
    edges_to_node: List[List[int]] = []
    # 入度
    in_degree: List[int]
    # 出度
    out_degree: List[int]
    # 节点数量
    nodes_count: int

    def _ensure_node(self, name: str):
        if name not in self.map_name_to_id:
            i = len(self.map_id_to_name)
            self.map_name_to_id[name] = i
            self.map_id_to_name.append(name)
            self.edges_from_node.append([])
            self.edges_to_node.append([])
            return i
        return self.map_name_to_id[name]

    def __init__(self, input_file_name: str):
        with open(input_file_name, "r") as f:
            for line in f:
                if line.find(",") == -1:
                    continue
                s_name, t_name = line.strip().split(",")
                s, t = self._ensure_node(s_name), self._ensure_node(t_name)
                self.edges_from_node[s].append(t)
                self.edges_to_node[t].append(s)
        self.nodes_count = len(self.map_id_to_name)
        self.in_degree = [len(l) for l in self.edges_to_node]
        self.out_degree = [len(l) for l in self.edges_from_node]

# 各参数接口仿照 NetworkX.pagerank()
def page_rank_algorithm(
    g: Graph,
    damping_factor: float,
    personalization: List[float] = None,
    max_iter: int = 200,
    tol: float = 1e-6
) -> List[float]:
    n = g.nodes_count
    if personalization is None:
        p0 = [1.0 / n for _ in range(n)]
    else:
        # 归一化
        s = sum(personalization)
        p0 = [x / s for x in personalization]
    
    p_prev: List[float] = [] # 上次迭代的 p 向量
    p_curr: List[float] = p0 # 本次迭代的 p 向量
    delta: float = 0
    for _ in range(max_iter):
        p_prev = p_curr
        p_curr = [0 for _ in range(n)]
        for s in range(n):
            # 如果页面 s 有外链
            if g.out_degree[s] != 0:
                # 当用户以 alpha 的概率从页面 s 的外链中点击页面 t
                for t in g.edges_from_node[s]:
                    p_curr[t] += p_prev[s] / g.out_degree[s] * damping_factor
                # 当用户以 1 - alpha 的概率放弃页面 s 并随机访问新的页面 t
                for t in range(n):
                    p_curr[t] += p_prev[s] * p0[t] * (1 - damping_factor)
            else:
                # 用户只能随机访问新的页面 t
                for t in range(n):
                    p_curr[t] += p_prev[s] * p0[t]
        
        # 本次迭代的平均变化量
        delta = sum([abs(p_prev[i] - p_curr[i]) for i in range(n)]) / n
        if delta <= tol:
            return p_curr

        print("%d-th iteration, delta = %f" % (_, delta))

    print("Delta = %f after %d iterations" % (delta, max_iter))
    return p_curr

def get_node_ranklist(page_rank: List[float]) -> List[int]:
    return [i for _, i in sorted(zip(page_rank, range(len(page_rank))), reverse=True)]

def PageRank(input_file_name: str, damping_factor: float) -> List[str]:
    g = Graph(input_file_name)
    page_rank = page_rank_algorithm(g, damping_factor)
    ranklist = get_node_ranklist(page_rank)

    # for i in range(10):
    #     print("%d-th %s %f" % (i, g.map_id_to_name[ranklist[i]], page_rank[ranklist[i]]))

    return [g.map_id_to_name[i] for i in ranklist]

def PPR(input_file_name: str, input_seed: str, damping_factor: float) -> List[str]:
    g = Graph(input_file_name)

    personalization: List[float] = [0 for _ in range(g.nodes_count)]
    with open(input_seed, "r") as f:
        for line in f:
            if line.find(",") == -1:
                continue
            name, value_s = line.split(",")
            personalization[g.map_name_to_id[name]] = float(value_s)

    page_rank = page_rank_algorithm(g, damping_factor, personalization)
    ranklist = get_node_ranklist(page_rank)

    for i in range(10):
        print("%d-th %s %f" % (i, g.map_id_to_name[ranklist[i]], page_rank[ranklist[i]]))

    return [g.map_id_to_name[i] for i in ranklist]

PageRank("soc-Epinions1.txt", 0.85)
# PPR("soc-Epinions1.txt", "seeds", 0.85)
