"""
verify_distil.py - is the distilled agent's 19.0% real, or an artefact?

Three identical win rates against three different opponents is implausible, so
this checks the agent rather than trusting the number:
  1. win rate vs Random and vs TLA-S on DIFFERENT seed bases
  2. how often it agrees with its teacher DURING ACTUAL PLAY (not on the
     training distribution) - the distribution-shift check
  3. whether its choices are actually varying (not stuck on index 0)
"""
import argparse
import random
from collections import Counter

import numpy as np

import tab_env
from tab_game import Board, TabRules, StickDice, PLAYER_1
from tab_look import get_candidate_allocations
from tab_mcts import apply_allocation, resolve_freeing
from tab_az import make_az_net, AZAgent, MAX_CAND, STATE_DIM
from tab_distil import teacher, _match
from tab_runner import play_one_game


def load(path):
    import torch
    net = make_az_net()
    net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    net.eval()
    return net


def winrate(net, opp_factory, n, seed0):
    w = 0
    for g in range(n):
        me = AZAgent(net, temperature=0.0)
        opp = opp_factory()
        if g % 2 == 0:
            r = play_one_game(me, opp, seed=seed0 + g); w += (r['winner'] == 1)
        else:
            r = play_one_game(opp, me, seed=seed0 + g); w += (r['winner'] == -1)
    return w / n


def agreement_in_play(net, n_games, seed0):
    """Play the DISTILLED agent's own games; at each of its decisions ask what
    the teacher would have done. Measures agreement under the student's own
    state distribution, which is what actually matters."""
    rules = TabRules(8)
    tea = teacher()
    agree = tot = 0
    picks = Counter()
    for g in range(n_games):
        random.seed(seed0 + g)
        board = Board(8)
        turn = 0
        from tab_ai import GAAgent
        opp = GAAgent('beginner', fitness_type='original')
        while turn < 300:
            player = PLAYER_1 if turn % 2 == 0 else -PLAYER_1
            seq = StickDice.throw_turn()
            rem = resolve_freeing(board, player, seq, rules)
            mov = rules.get_movable_soldiers(board, player, rem)
            if mov and rem:
                if player == PLAYER_1:
                    acts = get_candidate_allocations(rem, mov, player, board,
                                                       rules, enum_threshold=2000,
                                                       sample_n=80)
                    if len(acts) > MAX_CAND:
                        acts = random.sample(acts, MAX_CAND)
                    if len(acts) > 1:
                        feats = np.empty((len(acts), STATE_DIM), dtype=np.float32)
                        for i, a in enumerate(acts):
                            post = board.clone()
                            apply_allocation(post, a, mov, player, rules)
                            feats[i] = tab_env.encode_state(board, rem, post, 0,
                                                              player, rules)
                        import torch
                        with torch.no_grad():
                            lg, _ = net(torch.from_numpy(feats))
                        mine = int(lg.numpy().argmax())
                        picks[mine] += 1
                        talloc = tea.choose_allocation(board, player, rem, mov, rules)
                        tpick = _match(talloc, acts, mov)
                        if tpick >= 0:
                            tot += 1
                            agree += (mine == tpick)
                        chosen = acts[mine]
                    else:
                        chosen = acts[0] if acts else None
                    if chosen is not None:
                        apply_allocation(board, chosen, mov, player, rules)
                else:
                    al = opp.choose_allocation(board, player, rem, mov, rules)
                    lists = [[] for _ in mov]
                    idx = {p: i for i, p in enumerate(mov)}
                    for sm in al.soldier_moves:
                        if sm.from_pos in idx:
                            lists[idx[sm.from_pos]] = list(sm.values_used)
                    apply_allocation(board, lists, mov, player, rules)
            if board.is_game_over() is not None:
                break
            turn += 1
    return (agree / max(tot, 1)), tot, picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='../03_Results/checkpoints_distil/distil.pt')
    ap.add_argument('--games', type=int, default=60)
    args = ap.parse_args()
    net = load(args.ckpt)
    from tab_ai import GAAgent, RandomAgent

    print('win rates on DISTINCT seed bases:')
    for nm, f, s0 in [('Random', RandomAgent, 100000),
                        ('GA-Original', lambda: GAAgent('beginner', fitness_type='original'), 200000),
                        ('GA-Expert', lambda: GAAgent('beginner', fitness_type='expert'), 300000),
                        ('TLA-S', teacher, 400000)]:
        wr = winrate(net, f, args.games, s0)
        print(f'  vs {nm:<14} {wr*100:5.1f}%   (n={args.games}, seed0={s0})', flush=True)

    print('\nagreement with teacher UNDER THE STUDENT\'S OWN STATE DISTRIBUTION:')
    ag, tot, picks = agreement_in_play(net, 25, 500000)
    print(f'  {ag*100:.1f}%  over {tot} decisions')
    print(f'  training-distribution val accuracy was 77.8%')
    print(f'\n  chosen candidate index histogram (top 8): {picks.most_common(8)}')
    print(f'  distinct indices chosen: {len(picks)}')


if __name__ == '__main__':
    main()
