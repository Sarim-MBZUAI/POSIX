import networkx as nx

def sensitivity_index(prompt_dict, calc_dist):
    """
    prompt_dict: dictionary with prompts as key and output as value
    calc_dist: a function which takes two tuples (x1, y1) and (x2,y2) as input params
    """

    n_nodes = len(prompt_dict)
    #create a graph from the list
    G = nx.complete_graph(n_nodes)
    prompts = list(prompt_dict.keys())
    #get list of nodes and edges
    node_list = list(G.nodes)
    edge_list = list(G.edges)

    node_attr_dict = {}
    edge_weight_dict = {}

    for i in range(len(node_list)):
        node_attr_dict[node_list[i]] = {"input": prompts[i], "output": prompt_dict[prompts[i]]}
    
    #set attributes for each node: attributes are input (x) and output (y)
    nx.set_node_attributes(G, node_attr_dict)

    #add edge weights
    for i in range(len(edge_list)):
        edge = edge_list[i]
        edge_weight_dict[edge] = {"weight": calc_dist((G.node[edge[0]]["input"], G.node[edge[0]]["output"]), 
                                           (G.node[edge[1]]["input"], G.node[edge[1]]["output"]))
                                }
    nx.set_edge_attributes(G, edge_weight_dict)

    #get average shortest path length of the graph
    spread = nx.average_shortest_path_length(G)

    return spread