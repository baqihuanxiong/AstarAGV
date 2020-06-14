import astar
import numpy as np
import asyncio
import random
import time


class Port:
    def __init__(self, shape: tuple):
        self.road = np.zeros(shape, dtype=int)
        self.cell_lock = [[asyncio.Lock() for _ in range(shape[1])] for _ in range(shape[0])]
        self.assigned_paths = {}

    def check_conflict(self, path: list):
        for i, position in enumerate(path):
            if i == 0:
                continue
            occupied = self.road[position[0]][position[1]]
            if occupied == 0:
                continue
            conflict_path = self.assigned_paths[occupied]
            j = conflict_path.index(position)
            if j == len(conflict_path) - 1:
                continue
            d1 = np.array(path[i]) - np.array(path[i-1])
            d2 = np.array(conflict_path[j+1]) - np.array(conflict_path[j])
            if np.cross(d1, d2) == 0 and np.dot(d1, d2) < 0:
                return True
        return False

    def display(self, key):
        print("(%.3f) AGV %s position updated:\n %s" % (time.time(), key, self.road))


async def agv_work(uid: int, port: Port, position: tuple, targets: list, speed=1, load_duration=3):
    current_position = position
    for target in targets:
        road = np.copy(port.road)
        road[target[0]][target[1]] = 0
        path = astar.search(road, current_position, target)
        if path is None:
            raise Exception("AGV %s deadlock" % uid)
        port.assigned_paths[uid] = path
        print("(%.3f) AGV %s current position %s, next target is %s" % (time.time(), uid, current_position, target))
        i = 0
        while current_position != target:
            if not port.check_conflict([current_position, path[i+1]]):
                i += 1
                async with port.cell_lock[path[i][0]][path[i][1]]:
                    port.road[path[i][0]][path[i][1]] = uid
                    port.display(uid)
                    await asyncio.sleep(astar.dist_func(path[i], path[i - 1]) / speed)
                    port.road[path[i-1][0]][path[i-1][1]] = 0
                    port.display(uid)
                    current_position = path[i]
            else:
                print("(%.3f) AGV %s path conflict, recalculating" % (time.time(), uid))
                road = np.copy(port.road)
                road[target[0]][target[1]] = 0
                change_path = astar.search(road, current_position, target)
                if change_path is None:
                    raise Exception("AGV %s deadlock" % uid)
                path[i:] = change_path
                port.assigned_paths[uid] = path
        print("(%.3f) AGV %s shifting container" % (time.time(), uid))
        await asyncio.sleep(load_duration)


async def simulation(port_shape: tuple, agv_targets: list, n_agv: int, n_task: int):
    random.seed(42)
    sim_port = Port(port_shape)
    sim_tasks = {}
    for i in range(n_agv):
        init_position = (port_shape[0] // 2, (i + 1) * (port_shape[1] // (n_agv + 1)))
        sim_tasks[i + 1] = [init_position] + random.sample(agv_targets, n_task // n_agv) + [init_position]
        sim_port.road[init_position[0]][init_position[1]] = i + 1
    print("Tasks: %s" % sim_tasks)
    sim_port.display("init")
    agv_coroutines = []
    for agv, targets in sim_tasks.items():
        agv_coroutine = asyncio.create_task(agv_work(agv, sim_port, targets[0], targets[1:]))
        agv_coroutines.append(agv_coroutine)
    start_time = time.time()
    for agv_coroutine in agv_coroutines:
        await agv_coroutine
    end_time = time.time()
    print("(%.3f) Use %.3f seconds." % (end_time, end_time - start_time))


if __name__ == "__main__":
    sim_port_shape = (5, 10)
    sim_agv_targets = [(0, 1), (0, 3),
                       (4, 2), (4, 4), (4, 6), (4, 8)]
    asyncio.run(simulation(sim_port_shape, sim_agv_targets, 5, 5))
