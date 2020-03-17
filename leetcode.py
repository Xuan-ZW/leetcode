from typing import List
import numpy as np 
class Solution:
    def maxProfit(self, prices:List[int]) ->int:
        if not prices:
            return 0
        min_tmp = 1000000
        max_tmp = -1000000
        for price in prices:
            if price <= min_tmp:
                min_tmp = price
            if price -min_tmp >= max_tmp:
                max_tmp = price-min_tmp
        return max_tmp
    
    def majorityElement(self, nums:List[int]) -> int:
        """

            给定一个大小为 n 的数组，找到其中的多数元素。
            多数元素是指在数组中出现次数大于 ⌊ n/2 ⌋ 的元素。
            你可以假设数组是非空的，并且给定的数组总是存在多数元素。
        """ 
        tmp_dict = {}
        for num in nums:
            try:
                tmp_dict[num] += 1
            except:
                tmp_dict[num] = 1
        print(tmp_dict)
        length = len(nums) / 2
        for tmp in tmp_dict:
            if tmp_dict[tmp] > length:
                return tmp
            # print(tmp_dict[tmp])
        # return 0


    def lengthOfLIS(self, nums:List[int])->int:
        """
        给定一个无序的整数数组，找到其中最长上升子序列的长度。
        示例:
        输入: [10,9,2,5,3,7,101,18]
        输出: 4 
        解释: 最长的上升子序列是 [2,3,7,101]，它的长度是 4。
        说明:
        可能会有多种最长上升子序列的组合，你只需要输出对应的长度即可。
        你算法的时间复杂度应该为 O(n2) 。
        进阶: 你能将算法的时间复杂度降低到 O(n log n) 吗?
        """

        if not nums:
            return 0
        if len(nums) == 1:
            return 1
        length = len(nums)
        dp = []
        # dp.append(1)
        for i in range(length):
            
            dp.append(1) 

            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)


        return max(dp)


    def compressString(self, S: str) -> str:
        """
        字符串压缩。利用字符重复出现的次数，编写一种方法，实现基本的字符串压缩功能。
        比如，字符串aabcccccaaa会变为a2b1c5a3。若“压缩”后的字符串没有变短，则返回原先的字符串。你可以假设字符串中只包含大小写英文字母（a至z）。
        示例1:

        输入："aabcccccaaa"
        输出："a2b1c5a3"
        示例2:

        输入："abbccd"
        输出："abbccd"
        解释："abbccd"压缩后为"a1b2c2d1"，比原字符串长度更长。
        提示：
        字符串长度在[0, 50000]范围内。

        """
        if not S:
            return S
        count = 1
        result = ''

        first = S[0]
        for s in S[1:]:
            if s==first:
                count += 1
            else:
                tmp = first + str(count)
                result += tmp
                first = s
                count = 1
        result = result + first + str(count)
        if len(result) >= len(S):
            return S
            
    
    def dfs(self, grid, cur_i, cur_j):
        if cur_i < 0 or cur_j < 0 or cur_i == len(grid) or cur_j ==len(grid[0]) or grid[cur_i][cur_j] == 0:
            return 0
        grid[cur_i][cur_j] = 0
        result = 1
        for i, j in [[-1, 0], [1, 0], [0, -1], [0,1]]:
            result += self.dfs(grid, cur_i + i, cur_j + j)
        return result

    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        """
        给定一个包含了一些 0 和 1的非空二维数组 grid , 一个 岛屿 
        是由四个方向 (水平或垂直) 的 1 (代表土地) 构成的组合。你可以假设二维矩阵的四个边缘都被水包围着。
        找到给定的二维数组中最大的岛屿面积。(如果没有岛屿，则返回面积为0。)
        示例 1:

        [[0,0,1,0,0,0,0,1,0,0,0,0,0],
         [0,0,0,0,0,0,0,1,1,1,0,0,0],
         [0,1,1,0,1,0,0,0,0,0,0,0,0],
         [0,1,0,0,1,1,0,0,1,0,1,0,0],
         [0,1,0,0,1,1,0,0,1,1,1,0,0],
         [0,0,0,0,0,0,0,0,0,0,1,0,0],
         [0,0,0,0,0,0,0,1,1,1,0,0,0],
         [0,0,0,0,0,0,0,1,1,0,0,0,0]]
        对于上面这个给定矩阵应返回 6。注意答案不应该是11，因为岛屿只能包含水平或垂直的四个方向的‘1’。
        示例 2:

        [[0,0,0,0,0,0,0,0]]
        对于上面这个给定的矩阵, 返回 0。
        注意: 给定的矩阵grid 的长度和宽度都不超过 50。
        """
        ans = 0
        for i, row in enumerate(grid):
            for j, col in enumerate(row):
                ans = max(self.dfs(grid, i, j), ans)
        return ans

    def countCharacters(self, words: List[str], chars: str) -> int:
        """
        给你一份『词汇表』（字符串数组） words 和一张『字母表』（字符串） chars。
        假如你可以用 chars 中的『字母』（字符）拼写出 words 中的某个『单词』（字符串），那么我们就认为你掌握了这个单词。
        注意：每次拼写时，chars 中的每个字母都只能用一次。
        返回词汇表 words 中你掌握的所有单词的 长度之和。

        示例 1：

        输入：words = ["cat","bt","hat","tree"], chars = "atach"
        输出：6
        解释： 
        可以形成字符串 "cat" 和 "hat"，所以答案是 3 + 3 = 6。
        示例 2：

        输入：words = ["hello","world","leetcode"], chars = "welldonehoneyr"
        输出：10
        解释：
        可以形成字符串 "hello" 和 "world"，所以答案是 5 + 5 = 10。
        """
        dict_chars = {}
        flag = True
        for c in chars:
            try:
                dict_chars[c] += 1
            except:
                dict_chars[c] = 1
        length = 0
        for word in words:
            dict_word = {}
            for c in word:
                try:
                    dict_word[c] += 1
                except:
                    dict_word[c] = 1
            for c in set(word):
                try:
                    if dict_chars[c] < dict_word[c]:
                        flag = False
                        break
                except:
                    flag = False
                    break
            if flag == True:
                length += len(word)
            else:
                flag = True
        return length

            




        






if __name__ == "__main__":
    solu = Solution()
    # result = solu.maxProfit([7, 1, 5, 3, 6, 4])
    # result = solu.maxProfit([7, 6, 4, 3, 1])
    # result = solu.majorityElement([3,2,3])
    # result = solu.lengthOfLIS([4,10,4,3,8,9])
    # result = solu.compressString("aabcccccaaa")
    # result = solu.maxAreaOfIsland([[0,0,0,0,0,0,0,0]])
    # grid = [[0,0,1,0,0,0,0,1,0,0,0,0,0],
    #      [0,0,0,0,0,0,0,1,1,1,0,0,0],
    #      [0,1,1,0,1,0,0,0,0,0,0,0,0],
    #      [0,1,0,0,1,1,0,0,1,0,1,0,0],
    #      [0,1,0,0,1,1,0,0,1,1,1,0,0],
    #      [0,0,0,0,0,0,0,0,0,0,1,0,0],
    #      [0,0,0,0,0,0,0,1,1,1,0,0,0],
    #      [0,0,0,0,0,0,0,1,1,0,0,0,0]]
    # bool_matrix = [[False] *2]*3
    # result = solu.maxAreaOfIsland(grid)

    words = ["hello","world","leetcode"]
    chars = "welldonehoneyr"
    result = solu.countCharacters(words, chars)
    print(result)
    
