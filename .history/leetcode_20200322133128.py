from typing import List
import numpy as np 
import heapq
import math
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

    def isRectangleOverlap(self, rec1:List[int], rec2:List[int]) ->bool:
        """
        矩形以列表 [x1, y1, x2, y2] 的形式表示，其中 (x1, y1) 为左下角的坐标，(x2, y2) 是右上角的坐标。

        如果相交的面积为正，则称两矩形重叠。需要明确的是，只在角或边接触的两个矩形不构成重叠。

        给出两个矩形，判断它们是否重叠并返回结果。

        示例 1：

        输入：rec1 = [0,0,2,2], rec2 = [1,1,3,3]
        输出：true
        示例 2：

        输入：rec1 = [0,0,1,1], rec2 = [1,0,2,1]
        输出：false
        

        提示：

        两个矩形 rec1 和 rec2 都以含有四个整数的列表的形式给出。
        矩形中的所有坐标都处于 -10^9 和 10^9 之间。
        x 轴默认指向右，y 轴默认指向上。
        你可以仅考虑矩形是正放的情况。
        """

        if rec1[2] <=rec2[0] or rec2[2] <= rec1[0]:
            return False
        if rec1[3] <= rec2[1] or rec2[3] <= rec1[1]:
            return False
        return True  
            
    def longestPalindrome(self, s:str) -> int:
        """
        给定一个包含大写字母和小写字母的字符串，找到通过这些字母构造成的最长的回文串。

        在构造过程中，请注意区分大小写。比如 "Aa" 不能当做一个回文字符串。

        注意:
        假设字符串的长度不会超过 1010。

        示例 1:

        输入:
        "abccccdd"

        输出:
        7

        解释:
        我们可以构造的最长的回文串是"dccaccd", 它的长度是 7。
        """
        dict_s = {}
        flag = False
        length = 0
        for s1 in s:
            if s1 not in dict_s:
                dict_s[s1] = 1
            else:
                dict_s[s1] += 1
        
        for key in dict_s:
            value = dict_s[key]
            if value % 2 == 0:
                length += value
            else:
                flag = True
                length += value-1
        if flag == True:
            length += 1
        return length 

    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        """
        输入整数数组 arr ，找出其中最小的 k 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。
        示例 1：

        输入：arr = [3,2,1], k = 2
        输出：[1,2] 或者 [2,1]
        示例 2：

        输入：arr = [0,1,2,1], k = 1
        输出：[0]

        限制：

        0 <= k <= arr.length <= 10000
        0 <= arr[i] <= 10000
        """
        if k == 0 or len(arr)==0:
            return []
        hp = [-x for x in arr[0:k]]
        heapq.heapify(hp)
        for i in range(k, len(arr)):
            if -hp[0] > arr[i]:
                heapq.heappop(hp)
                heapq.heappush(hp, -arr[i])
        ans = [-x for x in hp]
        return ans

    def canMeasureWater(self, x:int, y:int, z:int) ->bool:
        """
        有两个容量分别为 x升 和 y升 的水壶以及无限多的水。
        请判断能否通过使用这两个水壶，从而可以得到恰好 z升 的水？
        如果可以，最后请用以上水壶中的一或两个来盛放取得的 z升 水。
        你允许：
        装满任意一个水壶
        清空任意一个水壶
        从一个水壶向另外一个水壶倒水，直到装满或者倒空
        示例 1: (From the famous "Die Hard" example)
        输入: x = 3, y = 5, z = 4
        输出: True
        示例 2:
        输入: x = 2, y = 6, z = 5
        输出: False
        """

        
        """
        # 法一：
        if x + y < z:
            return False
        if x == 0 or y == 0:
            return z == 0 or x + y == 0
        return z % math.gcd(x, y) == 0
        """

        # 法二
        stack = [(0, 0)]
        self.seen = set()
        while stack:
            remain_x, remain_y = stack.pop()
            if remain_x == z or remain_y == z or remain_x + remain_y == z:
                return True
            if (remain_x, remain_y) in self.seen:
                continue
            self.seen.add((remain_x, remain_y))
            # 把X壶灌满
            stack.append((x, remain_y))
            # 把Y壶灌满
            stack.append((remain_x, y))
            # 把X 壶倒空
            stack.append((0, remain_y))
            # 把Y 壶倒空
            stack.append((remain_x, 0))
            # 把X壶中的水倒入Y壶，直至灌满或倒空
            stack.append((remain_x - min(remain_x, y-remain_y), remain_y + min(remain_x, y-remain_y)))
            # 把Y壶中的水倒入X壶，直至灌满或倒空
            stack.append((remain_x + min(remain_y, x-remain_x), remain_y - min(remain_y, x-remain_x)))

        return False


    def minIncrementForUnique(self, A: List[int]) -> int:
        """
        给定整数数组 A，每次 move 操作将会选择任意 A[i]，并将其递增 1。
        返回使 A 中的每个值都是唯一的最少操作次数。
        示例 1:
        输入：[1,2,2]
        输出：1
        解释：经过一次 move 操作，数组将变为 [1, 2, 3]。
        示例 2:
        输入：[3,2,1,2,1,7]
        输出：6
        解释：经过 6 次 move 操作，数组将变为 [3, 4, 1, 2, 5, 7]。
        可以看出 5 次或 5 次以下的 move 操作是不能让数组的每个值唯一的。
        提示：
        0 <= A.length <= 40000
        0 <= A[i] < 40000
        """
        temp = []
        res = 0
        counter = [0] * 80000
        for x in A:
            counter[x] += 1
        for i in range(8):
            if counter[i] >= 2:

                temp.extend([i] * (counter[i] -1))

            else:
                if counter[i] == 0 and temp:
                    res += i - temp.pop(0)
        return res












        






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

    # words = ["hello","world","leetcode"]
    # chars = "welldonehoneyr"
    # result = solu.countCharacters(words, chars)

    # rec1 = [0,0,1,1]
    # rec2 = [1,0,2,1]
    # result = solu.isRectangleOverlap(rec1, rec2)


    # s = "A"
    # result = solu.longestPalindrome(s)

    # arr = [3,2,1]
    # k = 2
    # result = solu.getLeastNumbers(arr, k)

    # result = solu.canMeasureWater(3, 5, 4)

    result = solu.minIncrementForUnique([1,2,2])

    print(result)
    
