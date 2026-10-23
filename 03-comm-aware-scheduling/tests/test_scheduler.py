from src.scheduler import Node, place_tasks, score_node


def test_prefers_low_comm():
    nodes = [
        Node("busy-net", 0.1, 0.9),
        Node("quiet-net", 0.5, 0.1),
    ]
    assert score_node(nodes[1]) < score_node(nodes[0])
    p = place_tasks(["job"], nodes)
    assert p[0].node == "quiet-net"
