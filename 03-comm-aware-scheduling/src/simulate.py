from .scheduler import Node, place_tasks


def main() -> None:
    nodes = [
        Node("n0", cpu_load=0.2, comm_load=0.8),
        Node("n1", cpu_load=0.5, comm_load=0.2),
        Node("n2", cpu_load=0.3, comm_load=0.4),
    ]
    placements = place_tasks(["tA", "tB", "tC", "tD"], nodes)
    for p in placements:
        print(f"{p.task} -> {p.node} (score={p.score:.3f})")


if __name__ == "__main__":
    main()
