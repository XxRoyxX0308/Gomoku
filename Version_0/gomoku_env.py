import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GomokuEnv(gym.Env):
    """
    五子棋環境 (15x15)，符合 Gymnasium 介面。棋盤以 numpy 2D 陣列儲存。
    動作：單一整數 0~224；觀測：15x15 陣列 (0=空, 1=玩家1, 2=玩家2)。
    贏的條件：任一玩家連成 5 子即勝。
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, board_size=15, win_length=5):
        self.board_size = board_size
        self.win_length = win_length
        self.board = np.zeros((2, board_size, board_size), dtype=np.int8)

        self.current_player = 0
        self.done = False

    def reset(self):
        """
        重置環境，清空棋盤，當前玩家設為 1，返回初始觀測。
        """
        self.board.fill(0)
        self.current_player = 0
        self.done = False
        return self.board.copy(), None

    def render(self):
        """
        簡單列印棋盤，0 為空、1 為玩家1棋子、2 為玩家2棋子。
        """
        # 可以使用更豐富的視覺化，例如 matplotlib，但這裡採簡單文字列印
        for i in range(len(self.board[0])):
            row = self.board[0][i] + self.board[1][i] * 2
            print(' '.join("-" if x == 0 else ("O" if x == 1 else "X") for x in row))
        print()

    def check_five(self, r, c, player, win=5):
        """
        檢查最近在 (r, c) 位置放置 player 的棋子後，是否連成5個子。 
        檢測四個方向：水平、垂直、兩條對角線。
        """
        directions = [(0,1), (1,0), (1,1), (-1,1)]
        for dr, dc in directions:
            count = 1
            rr, cc = r + dr, c + dc
            while 0 <= rr < self.board_size and 0 <= cc < self.board_size \
                  and self.board[player, rr, cc] == 1:
                count += 1
                rr += dr
                cc += dc
            # 向相反方向後退計數
            rr, cc = r - dr, c - dc
            while 0 <= rr < self.board_size and 0 <= cc < self.board_size \
                  and self.board[player, rr, cc] == 1:
                count += 1
                rr -= dr
                cc -= dc
            # 如連續數 >= win_length(5)，則勝利
            if count >= win:
                return True
        return False

    def step(self, action):
        """
        執行一步棋。action 為 0~224 的整數，對應放置棋子的座標。
        返回觀測 obs, reward, done, info:contentReference[oaicite:5]{index=5}。
        Reward: 玩家1(智能體)勝=+1，玩家2(對手)勝=-1，平局=0。非法步驟也視為 -1 並結束。
        """
        if self.done:
            raise RuntimeError("Game is done. Please reset the environment.")
        
        row, col = action
        reward = 0

        # 檢查動作是否在範圍內
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            raise ValueError(f"Invalid action {action}")
        # 如果該位置已有棋子，視為非法
        if (self.board[:, row, col] != 0).any():
            self.done = True
            reward = -1
            print("Why", row, col)
            self.render()
            return None, reward, self.done, {}
        
        # 放置當前玩家的棋子
        self.board[self.current_player, row, col] = 1
        # 檢查是否構成五子連珠
        if self.check_five(row, col, self.current_player, self.win_length):
            self.done = True
            # 當前玩家獲勝，若為玩家1(智能體)則 reward=+1，否則-1
            reward = 1 # if self.current_player == 1 else -1
            self.render()
            return None, reward, self.done, {}
        # elif self.check_five(row, col, self.current_player, 4):
        #     reward += 0.5
        # elif self.check_five(row, col, self.current_player, 3):
        #     reward += 0.3
        # elif self.check_five(row, col, self.current_player, 2):
        #     reward += 0.2
        
        # 檢查是否和局（棋盤已滿）
        if not (self.board[0] + self.board[1] == 0).any():
            self.done = True
            return None, 0, self.done, {}
        
        # # 切換到另一位玩家 (環境) 下棋
        self.current_player = int(not self.current_player)
        # 未分出勝負，回傳 reward=0
        # reward = -0.01
        reward = 0

        if self.current_player:
            return self.board.copy(), reward, self.done, {}
        else:
            flip = np.flip(self.board, 0)
            return flip.copy(), reward, self.done, {}
