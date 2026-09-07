"""The paper's mechanism is about CREATING stacks, not holding them.
Measure, for each stacking mode, the fraction of decisions whose chosen
allocation creates a new friendly stack."""
from tab_look import LookAgent
from tab_runner import play_one_game
import tab_ai


def creates_stack(before, after, player):
    for i in range(len(before)):
        was = abs(before[i]) if before[i] * player > 0 else 0
        now = abs(after[i]) if after[i] * player > 0 else 0
        if now > 1 and now > was:
            return True
    return False


def probe(mode, label, n_games=8, seed0=7000):
    st = {'n': 0, 'create': 0, 'hold': 0}
    orig = LookAgent.choose_allocation

    def wrapped(self, board, player, throw_values, movable_soldiers, rules):
        before = list(board.cells)
        res = orig(self, board, player, throw_values, movable_soldiers, rules)
        st['n'] += 1
        if any(v * player > 1 for v in before):
            st['hold'] += 1
        if res is not None:
            # TurnAllocation -> per-soldier value lists aligned to movable_soldiers
            lists = [[] for _ in movable_soldiers]
            for sm in getattr(res, 'soldier_moves', []):
                try:
                    idx = movable_soldiers.index(sm.from_pos)
                except ValueError:
                    continue
                lists[idx].extend(sm.values_used)
            sim = rules.simulate_allocation(board, lists, movable_soldiers, player)
            after = list(sim.cells)
            if creates_stack(before, after, player):
                st['create'] += 1
        return res

    LookAgent.choose_allocation = wrapped
    try:
        for g in range(n_games):
            a = LookAgent(columns=8, stacking_mode=mode)
            b = tab_ai.GAAgent('beginner', fitness_type='original')
            if g % 2 == 0:
                play_one_game(a, b, seed=seed0 + g)
            else:
                play_one_game(b, a, seed=seed0 + g)
    finally:
        LookAgent.choose_allocation = orig
    n = st['n'] or 1
    print('  %-26s creates %5.1f%%   holds %5.1f%%   (%d decisions)'
          % (label, 100.0 * st['create'] / n, 100.0 * st['hold'] / n, st['n']))


print('PYTHON engine, stacking behaviour by evaluator mode:')
probe('simple', 'TLA-S  (flat, all stacks)')
probe('quadratic', 'TLE    (quadratic, new only)')
probe('off', 'no stacking term')
print()
print('note: "simple" penalises EVERY stack of size>1 unconditionally;')
print('      "quadratic" penalises only NEWLY CREATED stacks, with a rescue')
print('      exemption. They differ in conditionality as well as magnitude.')
