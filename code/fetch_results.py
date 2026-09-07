"""Download the revision's result folders from the server into 03_Results."""
import os
import posixpath
import sys

import srv

REMOTE = 'D:/Ahmad Tarawneh/Tab/HassanatTab/03_Results'
LOCAL = r'E:\Work\Mutah\Research\HassanatTab\03_Results'

FOLDERS = [
    'alignment_with_tle', 'autopsy_full',
    'board_4x7', 'board_4x7_p', 'board_4x8', 'board_4x8_p',
    'board_4x9', 'board_4x9_p', 'board_4x11', 'board_4x11_p',
    'budget_sweep', 'budget_sweep_p',
    'final_roster', 'mcts_ladder', 'mcts_tree', 'mcts_tree_p',
    'tlann', 'tlann2', 'checkpoints_td2',
]


def main():
    c = srv.client()
    sf = c.open_sftp()
    total = 0
    try:
        # result folders
        for d in FOLDERS:
            rd = posixpath.join(REMOTE, d)
            try:
                names = sf.listdir(rd)
            except IOError:
                print('  !! missing on server: %s' % d)
                continue
            ld = os.path.join(LOCAL, d)
            os.makedirs(ld, exist_ok=True)
            for n in names:
                rp = posixpath.join(rd, n)
                lp = os.path.join(ld, n)
                try:
                    size = sf.stat(rp).st_size
                except IOError:
                    continue
                if os.path.exists(lp) and os.path.getsize(lp) == size:
                    print('  = %-34s %10d (already)' % (d + '/' + n, size))
                    total += size
                    continue
                sf.get(rp, lp)
                total += size
                print('  + %-34s %10d' % (d + '/' + n, size))
                sys.stdout.flush()

        # run logs at the results root
        for n in sf.listdir(REMOTE):
            if not n.lower().endswith('.log'):
                continue
            rp = posixpath.join(REMOTE, n)
            lp = os.path.join(LOCAL, 'run_logs', n)
            os.makedirs(os.path.dirname(lp), exist_ok=True)
            sf.get(rp, lp)
            total += sf.stat(rp).st_size
            print('  + run_logs/%-25s %10d' % (n, sf.stat(rp).st_size))
    finally:
        sf.close()
        c.close()
    print()
    print('DONE  %.2f MB' % (total / 1048576.0))


if __name__ == '__main__':
    main()
