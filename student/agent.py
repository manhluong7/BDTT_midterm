"""Agent 2048: Expectimax trên bitboard (mỗi ô 4 bit = số mũ của ô, 0 = trống)."""
import random, time
import numpy as np
from game2048.observation import Observation
from game2048.rules import move_board
from game2048.types import Move

# Trọng số heuristic cho mỗi hàng / cột
base_score       = 200000.0
empty_weight     = 270.0
merge_weight     = 700.0
monotonic_power  = 4.0
monotonic_weight = 47.0
sum_power        = 3.5
sum_weight       = 11.0
row_mask         = 0xFFFF


# ---- Bảng tra cứu cho mọi hàng 16 bit (65536 khả năng) ----------------------
def slide_left(row):
    """Trượt hàng sang trái và gộp các ô bằng nhau."""
    tiles, result, i = [tile for tile in row if tile], [], 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            result.append(min(tiles[i] + 1, 15)); i += 2
        else:
            result.append(tiles[i]); i += 1
    return result + [0] * (4 - len(result))

def pack_row(row):          # [a,b,c,d] -> số 16 bit
    return row[0] | row[1] << 4 | row[2] << 8 | row[3] << 12

def spread_bits(bits):      # 4 nibble của hàng -> nibble đầu của 4 hàng (để ghép cột)
    return (bits & 15) | (bits >> 4 & 15) << 16 | (bits >> 8 & 15) << 32 | (bits >> 12 & 15) << 48

def score_row(row):
    """Điểm heuristic của một hàng: nhiều ô trống, nhiều ô gộp được, đơn điệu."""
    total_power = empty_count = merge_count = merge_streak = prev_tile = 0
    for tile in row:
        total_power += tile ** sum_power
        if tile == 0: empty_count += 1
        elif tile == prev_tile: merge_streak += 1
        else:
            if merge_streak: merge_count += 1 + merge_streak; merge_streak = 0
            prev_tile = tile
    if merge_streak: merge_count += 1 + merge_streak
    down_penalty = sum(row[i-1] ** monotonic_power - row[i] ** monotonic_power for i in (1, 2, 3) if row[i-1] > row[i])
    up_penalty   = sum(row[i] ** monotonic_power - row[i-1] ** monotonic_power for i in (1, 2, 3) if row[i-1] <= row[i])
    return (base_score + empty_weight * empty_count + merge_weight * merge_count
            - monotonic_weight * min(down_penalty, up_penalty) - sum_weight * total_power)

slide_left_table, slide_right_table, row_score_table = [0] * 65536, [0] * 65536, [0.0] * 65536
for bits in range(65536):
    row = [bits >> (4 * k) & 15 for k in range(4)]
    slide_left_table[bits]  = pack_row(slide_left(row))
    slide_right_table[bits] = pack_row(slide_left(row[::-1])[::-1])
    row_score_table[bits]   = score_row(row)
spread_table       = [spread_bits(bits) for bits in range(65536)]
col_up_table       = [spread_table[row] for row in slide_left_table]   # kết quả trượt lên của một cột
col_down_table     = [spread_table[row] for row in slide_right_table]  # kết quả trượt xuống của một cột
empty_shifts_table = [tuple(4 * k for k in range(4) if not bits >> (4 * k) & 15) for bits in range(65536)]


# ---- Thao tác trên bitboard --------------------------------------------------
def split_into_rows(board):
    return board & row_mask, board >> 16 & row_mask, board >> 32 & row_mask, board >> 48

def transpose_board(board):
    """Đổi hàng <-> cột. Ô (hàng i, cột j) nằm ở bit 16*i + 4*j, cần chuyển tới 16*j + 4*i.
    spread_table[hàng] đưa ô thứ j của hàng tới bit 16*j; dịch thêm 4*i bit là đúng vị trí mới."""
    return (spread_table[board & row_mask] | spread_table[board >> 16 & row_mask] << 4
            | spread_table[board >> 32 & row_mask] << 8 | spread_table[board >> 48] << 12)

def score_board(board):      # điểm = tổng điểm 4 hàng + 4 cột
    scores, cols = row_score_table, transpose_board(board)
    return (scores[board & row_mask] + scores[board >> 16 & row_mask]
            + scores[board >> 32 & row_mask] + scores[board >> 48]
            + scores[cols & row_mask]  + scores[cols >> 16 & row_mask]
            + scores[cols >> 32 & row_mask] + scores[cols >> 48])

def all_move_results(board):
    """(trái, phải, lên, xuống); nước đi không hợp lệ cho ra đúng board."""
    cols = transpose_board(board)
    row0, row1, row2, row3 = board & row_mask, board >> 16 & row_mask, board >> 32 & row_mask, board >> 48
    col0, col1, col2, col3 = cols & row_mask, cols >> 16 & row_mask, cols >> 32 & row_mask, cols >> 48
    slide_l, slide_r, slide_u, slide_d = slide_left_table, slide_right_table, col_up_table, col_down_table
    return (slide_l[row0] | slide_l[row1] << 16 | slide_l[row2] << 32 | slide_l[row3] << 48,
            slide_r[row0] | slide_r[row1] << 16 | slide_r[row2] << 32 | slide_r[row3] << 48,
            slide_u[col0] | slide_u[col1] << 4 | slide_u[col2] << 8 | slide_u[col3] << 12,
            slide_d[col0] | slide_d[col1] << 4 | slide_d[col2] << 8 | slide_d[col3] << 12)

def grid_to_bitboard(grid):
    return sum(min(int(cell).bit_length() - 1, 15) << 4 * i
               for i, cell in enumerate(grid.reshape(-1)) if cell > 0)


class Timeout(Exception):
    pass


class Agent:
    name          = "Team_4_Optimized"
    max_depth     = 12
    prob_cut      = 0.0008   # bỏ nhánh có xác suất nhỏ hơn mức này
    time_fraction = 0.85     # chỉ dùng 85% thời gian cho phép

    def __init__(self):
        self.cache, self._deadline, self._nodes = {}, 0.0, 0

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed); np.random.seed(seed)
        self.cache.clear()

    @staticmethod
    def evaluate(board, merge_score=0):
        return float(score_board(grid_to_bitboard(board)))

    def choose_move(self, obs: Observation, time_limit_ms: float) -> Move:
        if not obs.legal_moves:
            raise ValueError("no legal moves")
        start = time.perf_counter()
        try:
            time_budget = float(time_limit_ms) / 1000.0 * self.time_fraction
        except (TypeError, ValueError):
            time_budget = 0.1

        # Bàn cờ càng chật thì tìm càng sâu và được nghĩ lâu hơn
        empty_count = f"{grid_to_bitboard(obs.board):016x}".count("0")
        target_depth, depth_time_cap = 3, 0.006
        if empty_count <= 2: target_depth, depth_time_cap = 4, 0.02
        if empty_count <= 1: target_depth, depth_time_cap = 5, 0.04

        self._deadline = start + max(min(time_budget, depth_time_cap), 0.005)
        self._nodes = 0
        self.cache.clear()

        options = []            # (nước đi, bàn cờ sau khi đi) của các nước hợp lệ
        for move in obs.legal_moves:
            next_grid, _, changed = move_board(obs.board, move)
            if changed:
                options.append((move, grid_to_bitboard(next_grid)))
        if not options: return obs.legal_moves[0]
        if len(options) == 1: return options[0][0]

        best = options[0][0]
        for depth in range(1, min(target_depth, self.max_depth) + 1):   # tìm sâu dần
            try:
                best = max(options, key=lambda item: self._chance(item[1], depth - 1, 1.0))[0]
            except Timeout:
                break           # hết giờ: giữ kết quả của độ sâu trước
            if time.perf_counter() > self._deadline:
                break
        return best

    def _max(self, board, depth, prob):
        """Nút MAX: người chơi chọn nước đi tốt nhất."""
        self._nodes += 1
        if self._nodes & 255 == 0 and time.perf_counter() > self._deadline:
            raise Timeout
        if depth <= 0:
            return score_board(board)
        key = board << 4 | depth
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        at_leaf = depth == 1 or prob < self.prob_cut
        best = 0.0
        for nxt in all_move_results(board):
            if nxt != board:
                value = score_board(nxt) if at_leaf else self._chance(nxt, depth - 1, prob)
                if value > best:
                    best = value
        self.cache[key] = best
        return best

    def _chance(self, board, depth, prob):
        """Nút CHANCE: ô mới (2 với xác suất 0.9, 4 với 0.1) rơi vào ô trống ngẫu nhiên."""
        if depth <= 0 or prob < self.prob_cut:
            return score_board(board)
        empty_cells = [16 * i + shift for i, row in enumerate(split_into_rows(board))
                       for shift in empty_shifts_table[row]]
        count = len(empty_cells)
        if count == 0:
            return score_board(board)
        share = prob / count
        total = 0.0
        for shift in empty_cells:
            total += 0.9 * self._max(board | 1 << shift, depth, share * 0.9)
            total += 0.1 * self._max(board | 2 << shift, depth, share * 0.1)
        return total / count
