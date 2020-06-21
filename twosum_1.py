class Solution:

    from typing import List
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dict1 = {}
        for index, num in enumerate(nums):
            # print(index, num)
            
            if target - num not in dict1.keys():
                dict1[num] = index
            else:
                return[dict1[target - num], index]
            
        # for num in nums:
        #     try:
        #         if dict1[target - num]:
        #             return [dict1[num], dict1[target - num]]
        #     except:
        #         continue
if __name__ == "__main__":
    solu = Solution()
    nums = [2, 7, 11, 15]
    target = 22
    print(solu.twoSum(nums, target))