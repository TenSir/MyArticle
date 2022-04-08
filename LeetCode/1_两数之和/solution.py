# -*- coding: utf-8 -*-
# @Author  : TenSir
# @FileName: solution.py
# @Time    : 2020/9/18 21:26
# @Cnblogs ：Python知识学堂


# -*- codeing: utf-8 -*-


# 超出时间限制
'''
class Solution(object):
    def twoSum(self, nums, target):
        """
        :param nums:
        :param target:
        :return:
        """
        for i in range(0, len(nums)):
            for j in range(0, len(nums) - 1):
                if i == j:
                    pass
                else:
                    if nums[i] + nums[j] == target:
                        return [i, j]

#####################################################
'''

'''
#执行用时：6096 ms, 在所有 Python3 提交中击败了15.08%的用户
#内存消耗：13.9 MB, 在所有 Python3 提交中击败了96.64%的用户
class Solution(object):
    def twoSum(self, nums, target):
        """
        :param nums:
        :param target:
        :return:
        """
        for i in range(0, len(nums)):
            for j in range(i + 1, len(nums)):
                    if nums[i] + nums[j] == target:
                        return [i, j]
'''

#####################################################

'''
执行用时：1068 ms, 在所有 Python3 提交中击败了34.59%的用户
内存消耗：14 MB, 在所有 Python3 提交中击败了92.70%的用户

class Solution:
    def twoSum(self, nums, target):
        """
        :param nums:
        :param target:
        :return:
        """
        global i,j
        for i in range(len(nums)):
            # 如果差值在原nums[]中
            if (target - nums[i]) in nums:
                # 找到一个后，判断是不是自身，即target =nums[i] + nums[i]的情况
                if (nums.count(target - nums[i]) == 1) and (target - nums[i] == nums[i]):
                    # continue
                    pass
                else:
                    # 获取target - nums[i]在nums[]中的下表
                    j = nums.index(target - nums[i], i + 1)
                    break
        if j > 0:
            return [i,j]
        else:
            return []


nums = [2, 7, 11, 15]
target = 9
solution = Solution()
print(solution.twoSum(nums,target))

'''
#####################################################


# 哈希表方法
'''
class Solution(object):
    def twoSum(self, nums, target):
        """
        :param nums:list[]
        :param target:
        :return:
        """
        hashmap = {}
        for i in range(len(nums)):
            if target - nums[i] in hashmap:
                return [hashmap[target-nums[i]], i]
            else:
                hashmap[nums[i]] = i


nums = [2, 7, 11, 15]
target = 9
solution = Solution()
print(solution.twoSum(nums,target))
'''


# 双指针法

'''
执行用时：48 ms, 在所有 Python3 提交中击败了93.84%的用户
内存消耗：13.8 MB, 在所有 Python3 提交中击败了99.56%的用户
class Solution(object):
    def twoSum(self, nums, target):
        """
        :param nums:
        :param target:
        :return:
        """
        res = []
        nums_temp = nums[:]
        nums_temp.sort()
        left = 0
        right = len(nums_temp) - 1

        while left < right:
            if nums_temp[left] + nums_temp[right] == target:
                for i in range(0, len(nums)):
                    if nums[i] == nums_temp[left]:
                        res.append(i)
                        break
                for i in range(len(nums) - 1, -1, -1):
                    if nums[i] == nums_temp[right]:
                        res.append(i)
                        break
                res.sort()
                break

            elif nums_temp[left] + nums_temp[right] < target:
                left = left + 1
            elif nums_temp[left] + nums_temp[right] > target:
                right = right - 1
        return res


nums = [7, 11, 15, 2]
target = 9
solution = Solution()
print(solution.twoSum(nums, target))
'''