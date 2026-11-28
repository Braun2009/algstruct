"""Fifteen procedural tasks used for training and evaluation.

This file is the benchmark, not the architecture; it is imported by `train/run.py`
and by nothing in `model/`.

An event is (opcode, address, address/operand, payload). Same vocabulary and
output head across tasks; no task ID in the input. Only the final query is
supervised. The generator is seeded; the evaluator replays the visible program.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import itertools
import random

import torch


class Op(IntEnum):
    NOISE = 0
    WRITE = 1
    SWAP = 2
    COPY = 3
    READ = 4
    LINK = 5
    FOLLOW = 6
    IWRITE = 7
    PUSH = 8
    POP = 9
    PEEK = 10
    ENQUEUE = 11
    DEQUEUE = 12
    FRONT = 13
    OPEN = 14
    CLOSE = 15
    BIND = 16
    LOOKUP = 17
    SNAPSHOT = 18
    RESTORE = 19
    COMMIT = 20
    EDGE = 21
    REACH = 22
    EMIT = 23
    MODE = 24
    TURN = 25
    ORIENTATION = 26
    ATOM = 27
    CONCAT = 28
    REPEAT = 29
    REVERSE = 30
    INDEX = 31
    MOVE = 32
    TAKE = 33
    DROP = 34
    WHERE = 35


N_ADDR, N_VALUES, WIDTH = 16, 64, 4
VALUES = range(8, N_VALUES)
# Same integer codes for payloads and addresses are harmless: field embeddings
# are role-specific. All fields fit a shared embedding table.
VOCAB = N_VALUES
TASKS = (
    "flip_flop", "associative_recall", "register_swap", "register_copy",
    "pointer_walk", "indirect_write", "stack_peek", "queue_front",
    "scoped_binding", "rollback", "reachability", "frequency", "group_state",
    "composed_command", "entity_location",
)
PERMS = tuple(itertools.permutations(range(3)))
GENS = (PERMS[1], PERMS[3])


def event(op, a=0, b=0, v=0):
    return int(op), a, b, v


def interpret(program):
    """Execute only observed events. Returns (query index, label)."""
    mem, links = {}, {}
    stack, queue = [], []
    scopes, snapshots = [{}], []
    edges, counts = {}, {}
    orient, expression = PERMS[0], []
    location, carrying = {}, {}
    for t, (op, a, b, v) in enumerate(program):
        if op == Op.WRITE:
            mem[a] = v
        elif op == Op.SWAP:
            mem[a], mem[b] = mem[b], mem[a]
        elif op == Op.COPY:
            mem[a] = mem[b]
        elif op == Op.READ:
            return t, mem[a]
        elif op == Op.LINK:
            links[a] = b
        elif op == Op.FOLLOW:
            for _ in range(b):
                a = links[a]
            return t, mem[a]
        elif op == Op.IWRITE:
            mem[links[a]] = v
        elif op == Op.PUSH:
            stack.append(v)
        elif op == Op.POP:
            stack.pop()
        elif op == Op.PEEK:
            return t, stack[-1]
        elif op == Op.ENQUEUE:
            queue.append(v)
        elif op == Op.DEQUEUE:
            queue.pop(0)
        elif op == Op.FRONT:
            return t, queue[0]
        elif op == Op.OPEN:
            scopes.append({})
        elif op == Op.CLOSE:
            scopes.pop()
        elif op == Op.BIND:
            scopes[-1][a] = v
        elif op == Op.LOOKUP:
            return t, next(s[a] for s in reversed(scopes) if a in s)
        elif op == Op.SNAPSHOT:
            snapshots.append(mem.copy())
        elif op == Op.RESTORE:
            mem = snapshots.pop()
        elif op == Op.COMMIT:
            snapshots.pop()
        elif op == Op.EDGE:
            edges.setdefault(a, set()).add(b)
        elif op == Op.REACH:
            todo, seen = [a], {a}
            while todo:
                u = todo.pop()
                for w in edges.get(u, ()) - seen if u in edges else ():
                    todo.append(w)
                    seen.add(w)
            return t, int(b in seen)
        elif op == Op.EMIT:
            counts[v] = counts.get(v, 0) + 1
        elif op == Op.MODE:
            return t, max(counts, key=lambda x: (counts[x], -x))
        elif op == Op.TURN:
            orient = tuple(orient[GENS[a][i]] for i in range(3))
        elif op == Op.ORIENTATION:
            return t, PERMS.index(orient)
        elif op == Op.ATOM:
            expression = [v]
        elif op == Op.CONCAT:
            expression.append(v)
        elif op == Op.REPEAT:
            expression *= a
        elif op == Op.REVERSE:
            expression.reverse()
        elif op == Op.INDEX:
            return t, expression[a]
        elif op == Op.MOVE:
            location[a] = v
        elif op == Op.TAKE:
            carrying[b] = a
        elif op == Op.DROP:
            location[b] = location[carrying.pop(b)]
        elif op == Op.WHERE:
            return t, location[carrying.get(a, a)]
        elif op != Op.NOISE:
            raise ValueError(f"unknown opcode {op}")
    raise ValueError("no query")


@dataclass(frozen=True)
class Example:
    original: list[tuple[int, int, int, int]]
    edited: list[tuple[int, int, int, int]]
    target: int
    edited_target: int


def make_example(task: str, rng: random.Random, length: int, difficulty: int) -> Example:
    """Construct a paired, same-length program with one causal edit.

    `difficulty` grows the count of real operations/slots (not just padding).
    Evaluation includes noise AND label-changing edits that preserve the query.
    """
    if not 1 <= difficulty <= 8 or length < 24 + 6 * difficulty:
        raise ValueError("need difficulty 1..8 and length >= 24+6*difficulty")
    slots = min(12, difficulty + 3)
    ids = rng.sample(range(N_ADDR), slots)
    a, b, c, d = ids[:4]
    contents = rng.sample(VALUES, slots + 4)
    vals, rest = contents[:slots], contents[slots:]
    x, y, z, w = rest
    base = [event(Op.WRITE, i, v=v) for i, v in zip(ids, vals)]
    pl, pr = [], []
    changed = None

    def emit(e, alt=None):
        nonlocal changed
        pl.append(e)
        pr.append(e if alt is None else alt)
        if alt is not None:
            if changed is not None:
                raise ValueError("more than one edit")
            changed = len(pl) - 1

    def many(events):
        for e in events:
            emit(e)

    if task == "flip_flop":
        many(event(Op.WRITE, a, v=rng.choice(VALUES)) for _ in range(2 + difficulty))
        emit(event(Op.WRITE, a, v=x), event(Op.WRITE, a, v=y))
        query = event(Op.READ, a)
    elif task == "associative_recall":
        for addr, value in zip(ids, vals):
            emit(event(Op.WRITE, addr, v=value),
                 event(Op.WRITE, addr, v=x) if addr == a else None)
        many(event(Op.WRITE, rng.choice(ids[1:]), v=rng.choice(VALUES)) for _ in range(difficulty))
        query = event(Op.READ, a)
    elif task in ("register_swap", "register_copy"):
        many(base)
        emit(event(Op.SWAP, a, b), event(Op.SWAP, a, c))
        # Additional transport permutes the observed address; copies follow the
        # same route in both members. No labels reveal the source address.
        at = a
        for _ in range(difficulty):
            dest = rng.choice([i for i in ids if i != at])
            emit(event(Op.SWAP if task == "register_swap" else Op.COPY,
                       dest, at))
            at = dest
        query = event(Op.READ, at)
    elif task == "pointer_walk":
        many(base)
        path = ids[3:]
        many(event(Op.LINK, u, v) for u, v in zip(path, path[1:]))
        emit(event(Op.LINK, path[-1], b), event(Op.LINK, path[-1], c))
        query = event(Op.FOLLOW, path[0], len(path))
    elif task == "indirect_write":
        many(base)
        emit(event(Op.LINK, a, b), event(Op.LINK, a, c))
        many(event(Op.IWRITE, a, v=rng.choice(VALUES)) for _ in range(difficulty))
        emit(event(Op.IWRITE, a, v=x))
        query = event(Op.READ, b)
    elif task in ("stack_peek", "queue_front"):
        is_stack = task == "stack_peek"
        push, pop = ((Op.PUSH, Op.POP) if is_stack else (Op.ENQUEUE, Op.DEQUEUE))
        above = [event(push, v=u) for u in vals[:difficulty + 1]]
        if not is_stack:
            many(above)
        emit(event(push, v=x), event(push, v=y))
        if is_stack:
            many(above)
        else:
            many(event(push, v=u) for u in rest[2:])
        many(event(pop) for _ in above)
        query = event(Op.PEEK if is_stack else Op.FRONT)
    elif task == "scoped_binding":
        emit(event(Op.BIND, a, v=x), event(Op.BIND, a, v=y))
        for _ in range(difficulty):
            many([event(Op.OPEN), event(Op.BIND, a, v=rng.choice(VALUES)),
                  event(Op.BIND, rng.choice(ids[1:]), v=rng.choice(VALUES))])
        many(event(Op.CLOSE) for _ in range(difficulty))
        query = event(Op.LOOKUP, a)
    elif task == "rollback":
        many(base)
        emit(event(Op.WRITE, a, v=x), event(Op.WRITE, a, v=y))
        for _ in range(difficulty):
            many([event(Op.SNAPSHOT), event(Op.WRITE, a, v=rng.choice(VALUES))])
        many(event(Op.COMMIT) for _ in range(difficulty - 1))
        emit(event(Op.RESTORE))
        query = event(Op.READ, a)
    elif task == "reachability":
        path = ids[2:]
        emit(event(Op.EDGE, a, path[0]), event(Op.EDGE, b, path[0]))
        many(event(Op.EDGE, u, v) for u, v in zip(path, path[1:]))
        many(event(Op.EDGE, u, rng.choice(path)) for u in path)
        query = event(Op.REACH, a, path[-1])
    elif task == "frequency":
        emit(event(Op.EMIT, v=x), event(Op.EMIT, v=y))
        values = [x, y, z] * (difficulty + 1) + list(vals)
        rng.shuffle(values)
        many(event(Op.EMIT, v=value) for value in values)
        query = event(Op.MODE)
    elif task == "group_state":
        g = rng.randrange(2)
        emit(event(Op.TURN, g), event(Op.TURN, 1 - g))
        many(event(Op.TURN, rng.randrange(2)) for _ in range(3 * difficulty))
        query = event(Op.ORIENTATION)
    elif task == "composed_command":
        emit(event(Op.ATOM, v=x), event(Op.ATOM, v=y))
        # Mirror the boolean 'is this the original atom?' independently of its value.
        markers = [True]
        for i in range(difficulty):
            emit(event(Op.CONCAT, v=rng.choice(vals)))
            markers.append(False)
            if rng.random() < 0.5:
                emit(event(Op.REVERSE)); markers.reverse()
            if i % 2 == 0:
                emit(event(Op.REPEAT, 2)); markers *= 2
        query = event(Op.INDEX, rng.choice([i for i, marker in enumerate(markers) if marker]))
    elif task == "entity_location":
        emit(event(Op.MOVE, a, v=x), event(Op.MOVE, a, v=y))
        emit(event(Op.TAKE, a, b))
        for _ in range(difficulty):
            many([event(Op.MOVE, c, v=rng.choice(VALUES)), event(Op.TAKE, c, d),
                  event(Op.DROP, b=d)])
        emit(event(Op.DROP, b=b))
        query = event(Op.WHERE, b)
    else:
        raise ValueError(task)

    # Blind the query to the branch identity and the random answer. A suffix of
    # at least 7 decoys keeps final-item retrieval from being immediate.
    spare = length - 1 - len(pl)
    assert changed is not None and spare >= 7
    indices = sorted(rng.sample(range(length - 8), len(pl)))
    p = [event(Op.NOISE, rng.randrange(N_ADDR), rng.randrange(N_ADDR), rng.randrange(N_VALUES))
         for _ in range(length - 1)] + [query]
    q = p.copy()
    for i, left, right in zip(indices, pl, pr):
        p[i], q[i] = left, right
    _, label = interpret(p)
    _, opposite = interpret(q)
    assert label != opposite, (task, label)
    if rng.random() < 0.5:
        return Example(q, p, opposite, label)
    return Example(p, q, label, opposite)


def dataset(samples: int, length: int, difficulty: int, seed: int,
            tasks=TASKS, paired: bool = False):
    """Return integer events/labels/task indexes, and optionally paired data."""
    names = tuple(tasks)
    if not names or samples < len(names):
        raise ValueError("need at least one sample per requested task")
    rng = random.Random(seed)
    examples = [(i, make_example(name, rng, length, difficulty))
                for j in range(samples) for i, name in [(j % len(names), names[j % len(names)])]]
    rng.shuffle(examples)
    left = torch.tensor([e.original for _, e in examples], dtype=torch.uint8)
    labels = torch.tensor([e.target for _, e in examples])
    task = torch.tensor([i for i, _ in examples])
    if not paired:
        return left, labels, task
    right = torch.tensor([e.edited for _, e in examples], dtype=torch.uint8)
    edited_labels = torch.tensor([e.edited_target for _, e in examples])
    return left, labels, task, right, edited_labels
