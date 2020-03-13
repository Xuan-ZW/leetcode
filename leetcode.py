from typing import List
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


            



if __name__ == "__main__":
    solu = Solution()
    # result = solu.maxProfit([7, 1, 5, 3, 6, 4])
    result = solu.maxProfit([7, 6, 4, 3, 1])
    print(result)
    
