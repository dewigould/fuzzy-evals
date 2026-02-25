# LiveCodeBench v5 — Example Problems

## Example 1: Stdin (atcoder)

**Title:** Separated Lunch  
**Difficulty:** medium  
**Platform:** atcoder  
**Test cases:** 30  
**Starter code:** None  

### Prompt sent to model

~~~
You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.

As KEYENCE headquarters have more and more workers, they decided to divide the departments in the headquarters into two groups and stagger their lunch breaks.
KEYENCE headquarters have N departments, and the number of people in the i-th department (1\leq i\leq N) is K_i.
When assigning each department to Group A or Group B, having each group take lunch breaks at the same time, and ensuring that the lunch break times of Group A and Group B do not overlap, find the minimum possible value of the maximum number of people taking a lunch break at the same time.
In other words, find the minimum possible value of the larger of the following: the total number of people in departments assigned to Group A, and the total number of people in departments assigned to Group B.

Input

The input is given from Standard Input in the following format:
N
K_1 K_2 \ldots K_N

Output

Print the minimum possible value of the maximum number of people taking a lunch break at the same time.

Constraints


- 2 \leq N \leq 20
- 1 \leq K_i \leq 10^8
- All input values are integers.

Sample Input 1

5
2 3 5 10 12

Sample Output 1

17

When assigning departments 1, 2, and 5 to Group A, and departments 3 and 4 to Group B, Group A has a total of 2+3+12=17 people, and Group B has a total of 5+10=15 people. Thus, the maximum number of people taking a lunch break at the same time is 17.
It is impossible to assign the departments so that both groups have 16 or fewer people, so print 17.

Sample Input 2

2
1 1

Sample Output 2

1

Multiple departments may have the same number of people.

Sample Input 3

6
22 25 26 45 22 31

Sample Output 3

89

For example, when assigning departments 1, 4, and 5 to Group A, and departments 2, 3, and 6 to Group B, the maximum number of people taking a lunch break at the same time is 89.### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.
```python
# YOUR CODE HERE
```

### Answer: (use the provided format with backticks)


~~~

### Example test case

**Input:**
```
5
2 3 5 10 12
```

**Expected output:**
```
17
```

---

## Example 2: Functional (leetcode)

**Title:** find-x-sum-of-all-k-long-subarrays-i  
**Difficulty:** easy  
**Platform:** leetcode  
**Test cases:** 33  
**Starter code:**
```python
class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
```

### Prompt sent to model

~~~
You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.

You are given an array nums of n integers and two integers k and x.
The x-sum of an array is calculated by the following procedure:

Count the occurrences of all elements in the array.
Keep only the occurrences of the top x most frequent elements. If two elements have the same number of occurrences, the element with the bigger value is considered more frequent.
Calculate the sum of the resulting array.

Note that if an array has less than x distinct elements, its x-sum is the sum of the array.
Return an integer array answer of length n - k + 1 where answer[i] is the x-sum of the subarray nums[i..i + k - 1].
 
Example 1:

Input: nums = [1,1,2,2,3,4,2,3], k = 6, x = 2
Output: [6,10,12]
Explanation:

For subarray [1, 1, 2, 2, 3, 4], only elements 1 and 2 will be kept in the resulting array. Hence, answer[0] = 1 + 1 + 2 + 2.
For subarray [1, 2, 2, 3, 4, 2], only elements 2 and 4 will be kept in the resulting array. Hence, answer[1] = 2 + 2 + 2 + 4. Note that 4 is kept in the array since it is bigger than 3 and 1 which occur the same number of times.
For subarray [2, 2, 3, 4, 2, 3], only elements 2 and 3 are kept in the resulting array. Hence, answer[2] = 2 + 2 + 2 + 3 + 3.


Example 2:

Input: nums = [3,8,7,8,7,5], k = 2, x = 2
Output: [11,15,15,15,12]
Explanation:
Since k == x, answer[i] is equal to the sum of the subarray nums[i..i + k - 1].

 
Constraints:

1 <= n == nums.length <= 50
1 <= nums[i] <= 50
1 <= x <= k <= nums.length### Format: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.
```python
class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
```

### Answer: (use the provided format with backticks)


~~~

### Example test case

**Input:**
```
[1, 1, 2, 2, 3, 4, 2, 3]
6
2
```

**Expected output:**
```
[6, 10, 12]
```

