#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <stack> 
#include <map>
#include <queue>
#include <unordered_set>
using namespace std;
//19. 删除链表的倒数第N个节点
struct ListNode
{
    int val;
    ListNode* next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode* next) : val(x), next(next) {}
    void putVector(vector<int> data) {
        val = data[0];
        ListNode* cur = this;
        for (int i = 1; i < data.size(); i++) {
            ListNode* temp = new ListNode(data[i]);
            cur->next = temp;
            cur = temp;
        }
    }
    friend ostream& operator << (ostream& out, ListNode* l) {
        l->printList();
        return out;
    }
    friend ostream& operator << (ostream& out, ListNode& l) {
        l.printList();
        return out;
    }
    void printList() {
        ListNode* p = this;
        while (p != NULL) {
            cout << p->val << "\t";
            p = p->next;
        }
    }

};
struct TreeNode {
	int val;
	TreeNode* left;
	TreeNode* right;
	TreeNode() : val(0), left(nullptr), right(nullptr) {}
	TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
	TreeNode(int x, TreeNode* left, TreeNode* right) : val(x), left(left), right(right) {}
};
pair<ListNode*, ListNode*> reverseK(ListNode* head, ListNode* tail);
vector<int> samPreSuf(string s);
void printVector(vector<int> v);
void printVector2(vector<vector<int>> dp);
//68 Solution
string pushnum(vector<string>& words, int start, int end, int maxWidth, int len, bool isEnd);


// Definition for a Node.
class Node {
public:
    int val;
    vector<Node*> children;

    Node() {}

    Node(int _val) {
        val = _val;
    }

    Node(int _val, vector<Node*> _children) {
        val = _val;
        children = _children;
    }
};



class Solution
{
public:
    //4.寻找两个正序数组的中位数  
    static double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2);
    //5.最长回文子串  
    static string longestPalindrome(string s);
    //6.Z字型变换
    static string convertZ(string s, int numRows);
    //7.整数反转
    static int Myreverse(int x);
    //8.字符串转换整数 (atoi)
    static int myAtoi(string str);
    //9.回文数
    static bool isPalindrome(int x);
    //11.盛最多水的容器
    static int maxArea(vector<int>& height);
    //12.整数转罗马数字
    static string intToRoman(int num);
    //13.罗马转整数
    static int romanToInt(string s);
    //14.最长公共前缀
    static string longestCommonPrefix(vector<string>& strs);
    //15.三数之和
    static vector<vector<int>> threeSum(vector<int>& nums);
    //16.最接近的三数之和  
    static int threeSumClosest(vector<int>& nums, int target);
    //17. 电话号码的字母组合
    static vector<string> letterCombinations(string digits);
    //18.四数之和
    static vector<vector<int>> fourSum(vector<int>& nums, int target);
    //19. 删除链表的倒数第N个节点
    static ListNode* removeNthFromEnd(ListNode* head, int n);
    //20. 有效的括号
    static bool isValid(string s);
    //21.合并两个有序链表  
    static ListNode* mergeTwoLists(ListNode* l1, ListNode* l2);
    //22.括号生成
    static vector<string> generateParenthesis(int n);
    //23.合并K个升序链表
    static ListNode* mergeKLists(vector<ListNode*>& lists);
    //24. 两两交换链表中的节点
    static ListNode* swapPairs(ListNode* head);
    //25. K 个一组翻转链表
    static ListNode* reverseKGroup(ListNode* head, int k);
    //26. 删除排序数组中的重复项
    static int removeDuplicates(vector<int>& nums);
    //27. 移除元素
    static int removeElement(vector<int>& nums, int val);
    //28. 实现 strStr()
    static int strStr(string haystack, string needle);
    //29. 两数相除
    static int divide(int dividend, int divisor);
    //30. 串联所有单词的子串
    static vector<int> findSubstring(string s, vector<string>& words);
    //31. 下一个排列
    static void nextPermutation(vector<int>& nums);
    //32. 最长有效括号
    static int longestValidParentheses(string s);
    //33. 搜索旋转排序数组
    static int search(vector<int>& nums, int target);
    //34. 在排序数组中查找元素的第一个和最后一个位置
    static vector<int> searchRange(vector<int>& nums, int target);
    //35. 搜索插入位置
    static int searchInsert(vector<int>& nums, int target);
    //36. 有效的数独
    static bool isValidSudoku(vector<vector<char>>& board);
    //37. 解数独
    static void solveSudoku();
    //38.外观数列
    static string countAndSay(int n);
    //39. 组合总和
    static vector<vector<int>> combinationSum(vector<int>& candidates, int target);
    //40. 组合总和 II
    static vector<vector<int>> combinationSum2(vector<int>& candidates, int target);
    //41. 缺失的第一个正数
    static  int firstMissingPositive(vector<int>& nums);
    //42. 接雨水
    static int trap(vector<int>& height);
    //43. 字符串相乘
    static string multiply(string num1, string num2);
    //44.通配符匹配
    static bool isMatch(string s, string p);
    //45. 跳跃游戏 II
    static int jump(vector<int>& nums);
    //46. 全排列
    static vector<vector<int>> permute(vector<int>& nums);
    //47. 全排列 II
    static vector<vector<int>> permuteUnique(vector<int>& nums);
    //48. 旋转图像
    static void rotate(vector<vector<int>>& matrix);
    //49. 字母异位词分组
    static vector<vector<string>> groupAnagrams(vector<string>& strs);
    //50. Pow(x, n)
    static double myPow(double x, int n);
    //51. N 皇后
    static vector<vector<string>> solveNQueens(int n);
    //53. 最大子序和
    static int maxSubArray(vector<int>& nums);
    static int maxSubArray2(vector<int>& nums);
    static int maxSubArray3(vector<int>& nums);
    //54. 螺旋矩阵
    static vector<int> spiralOrder(vector<vector<int>>& matrix);
    //55. 跳跃游戏
    static bool canJump(vector<int>& nums);
    //56. 合并区间
    static vector<vector<int>> merge(vector<vector<int>>& intervals);
    //57. 插入区间
    static vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval);
    //58. 最后一个单词的长度
    static int lengthOfLastWord(string s);
    //59. 螺旋矩阵 II
    static vector<vector<int>> generateMatrix(int n);
    //60. 排列序列
    static string getPermutation(int n, int k);
    //61. 旋转链表
    static ListNode* rotateRight(ListNode* head, int k);
    //62. 不同路径
    static int uniquePaths(int m, int n);
    static int uniquePaths1(int m, int n);
    //63. 不同路径 II
    static int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid);
    static int uniquePathsWithObstacles2(vector<vector<int>>& obstacleGrid);
    static int uniquePathsWithObstacles3(vector<vector<int>>& obstacleGrid);
    //64. 最小路径和
    static int minPathSum(vector<vector<int>>& grid);
    //65.耗子尾汁

    //66. 加一
    static vector<int> plusOne(vector<int>& digits);
    //67.二进制求和
    static string addBinary1(string a, string b);
    static string addBinary2(string a, string b);
    //68. 文本左右对齐
    static vector<string> fullJustify(vector<string>& words, int maxWidth);
    static vector<string> fullJustify2(vector<string>& words, int maxWidth);
    //69. x 的平方根
    static int mySqrt(int x);
    //70. 爬楼梯
    static int climbStairs(int n);
    //71. 简化路径
    static string simplifyPath(string path);
    static string simplifyPath2(string path);
    //72. 编辑距离
    static int minDistance(string word1, string word2);
    static int minDistance2(string word1, string word2);
    //73. 矩阵置零
    static void setZeroes(vector<vector<int>>& matrix);
    //74. 搜索二维矩阵
    static bool searchMatrix(vector<vector<int>>& matrix, int target);
    //75. 颜色分类
    static void sortColors(vector<int>& nums);
};
//76. 最小覆盖子串
class Solution76 {
public:
    unordered_map<char, int> ori;
    unordered_map<char, int> cnt;
    int sleft = 0, sright = 0;
    int len = INT_MAX;
    void shrinkIndex(string& s, int& pleft, const int& pright);
    string minWindow(string s, string t);
};

class Solution84 {
public:
    //从左到右遍历，记录每个柱子的左边界
    vector<int> findLeftBorder(vector<int>& heights) {
        stack<int> borderStack;
        int border = 0;
        vector<int> ret(heights.size(), 0);
        for (int i = 0; i < heights.size(); i++) {
            if (i == 0 || heights[i] > heights[i - 1]) {
                //上升的
                borderStack.push(i);
                border = i;
            }
            else {
                //下降了，找它的左边界
                while (!borderStack.empty() && borderStack.top() >= heights[i]) {
                    border = borderStack.top();
                    borderStack.pop();
                }
            }
            ret[i] = border;
        }
        return ret;
    }
    //从右到左遍历，记录每个柱子的右边界
    //[ 3,2,4,3,4 ]
    vector<int> findRightBorder(vector<int>& heights) {
        stack<int> borderStack;
        int n = heights.size();
        int border = n - 1;
        vector<int> ret(n, n - 1);
        for (int i = heights.size() - 1; i >= 0; i--) {
            if (i == n - 1 || heights[i] > heights[i + 1]) {
                //上升的
                borderStack.push(i);
                border = i;
            }
            else {
                //下降了，找它的右边界
                while (!borderStack.empty() && borderStack.top() >= heights[i]) {
                    border = borderStack.top();
                    borderStack.pop();
                }
            }
            ret[i] = border;
        }
        return ret;
    }

    int largestRectangleArea1(vector<int>& heights) {
        if (heights.size() == 1) {
            return heights[0];
        }

        vector<int> left = findLeftBorder(heights);
        vector<int> right = findRightBorder(heights);
        int ret = 0;
        for (int i = 0; i < heights.size(); i++) {
            ret = max(ret, (right[i] - left[i] + 1) * heights[i]);
        }
        return ret;
    }
    /*
思路：
    如果暴力解法，对于某一个柱子，我们让它往两边找，直到找到坑为止就算到头了，然后算面积就是了

    但是这样也发现，如果我们从头往后遍历，要是下降了，那后面的肯定是前面那个的右边界，对应的他的左边界可以自己往前去搜索一下
    可以用while()一直找到比它小的heights,然后就是它的左边界
    但是这样的话，很容易出现重复的遍历


*/
//从左到右遍历，记录每个柱子的左边界
/*
    对于某个柱子
    如果它后面是 严格下降的话，那它的右边界就是他本身，到头了
    如果后面是 上升的话，那么它的右边界是不能确定的，需要往后找了，每一个都可能是它的右边界，
    所以每一个都要和当前没确定右边界的柱子比较一下，如果自己比没确定边界的小，那这个位置的就找到自己属于的右边界了
    但是如果比较的话，我们应该和最大的比，比完了如果比它小，那它的右边界就确定了，然后再看次小的，看看能不能确定它的右边界看样子才合理

    因此我们需要一个能存储未确定右边界的数据结构，并且还能按照顺序排列，而且我们用的时候拿出来是最大值，很容易想到，就是栈
    当然我们应用的过程也肯定是有序的，因为我们每次拿数的时候先看看自己是不是比没确定右边界的最大的值 严格小，
    如果小的话，我就把它拿出来了，因为它的右边界已经确定了
    如果大的话，我自己就塞进去了，因为我确定不了了

    这样一想，好像可以再优化
    我们把第一次和后面的比较的直接也算进这个循环呢,不就是把自己压入栈么？

    还有一点我没有发现，他的左边界怎么确定，我一开始以为可以通过反向的方法来确定它的左边界
    但是看到一个题解我突然醒悟，栈里面既然是单调增的，那么对应的这个柱子左边的不就是它的左边界么？
    因为中间的被弹出去的实际上是被我和前面那个小矮子一起把它给挤出去了
*/
    int largestRectangleArea(vector<int>& heights) {
        stack<int> st;
        st.push(-1);
        int maxArea = 0;
        int n = heights.size();
        for (int i = 0; i < heights.size(); i++) {
            while (st.top() != -1 && heights[st.top()] > heights[i]) {
                //可以夹出去当前的了，栈顶的最大面积已经确定了
                int cur = st.top();//可以确定的号
                st.pop();
                int left = st.top() + 1;
                int right = i;//左闭又开
                maxArea = max(maxArea, (right - left) * heights[cur]);
            }
            st.push(i);
        }
        while (st.top() != -1) {
            int cur = st.top();//可以确定的号
            st.pop();
            int left = st.top() + 1;
            int right = n;//一直到最后都没人消掉它，说明它可以延续到最后
            maxArea = max(maxArea, (right - left) * heights[cur]);
        }
        return maxArea;
    }
};

class Solution85 {
public:
    int largestRectangleArea(vector<int>& heights) {
        stack<int> st;
        st.push(-1);
        int maxArea = 0;
        int n = heights.size();
        for (int i = 0; i < heights.size(); i++) {
            while (st.top() != -1 && heights[st.top()] > heights[i]) {
                //可以夹出去当前的了，栈顶的最大面积已经确定了
                int cur = st.top();//可以确定的号
                st.pop();
                int left = st.top() + 1;
                int right = i;//左闭又开
                maxArea = max(maxArea, (right - left) * heights[cur]);
            }
            st.push(i);
        }
        while (st.top() != -1) {
            int cur = st.top();//可以确定的号
            st.pop();
            int left = st.top() + 1;
            int right = n;//一直到最后都没人消掉它，说明它可以延续到最后
            maxArea = max(maxArea, (right - left) * heights[cur]);
        }
        return maxArea;
    }
    int maximalRectangle(vector<vector<char>>& matrix) {
        if (matrix.empty()) {
            return 0;
        }
        int m = matrix.size();
        int n = matrix[0].size();
        vector<vector<int>> heights(m, vector<int>(n, 1));
        for (int j = 0; j < n; j++) {
            for (int i = m - 1; i >= 0; i--) {
                if (i == m - 1) {
                    heights[i][j] = matrix[i][j] - '0';
                }
                else if (matrix[i][j] == '1') {
                    heights[i][j] = heights[i + 1][j] + 1;
                }
                else {
                    heights[i][j] = 0;
                }
            }
        }
        int maxArea = 0;
        for (int i = 0; i < m; i++) {
            maxArea = max(maxArea, largestRectangleArea(heights[i]));
        }
        return maxArea;
    }
};
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution86 {
public:
    //1,4,3,2,5,2
    ListNode* partition(ListNode* head, int x) {
        ListNode smallHead = ListNode(0);
        ListNode* small = &smallHead;
        ListNode bigHead = ListNode(0);
        ListNode* big = &bigHead;
        ListNode* p = head;
        while (p != NULL) {
            if (p->val < x) {
                small->next = p;
                small = small->next;
            }
            else {
                big->next = p;
                big = big->next;
            }
            p = p->next;
        }
        small->next = bigHead.next;
        big->next = NULL;
        return smallHead.next;

    }
};

class Solution90 {
public:
    vector<int> nums;
    int n;
    vector<vector<int>> result;
    //每一层的任务就是看看能不能往下一层放东西
    void dfs(int depth, int start, vector<int> cur) {
        for (int i = start; i < n; i++) {
            if (i > start && nums[i] == nums[i - 1]) {
                continue;
            }
            //符合要求
            cur.push_back(nums[i]);
            result.push_back(cur);
            dfs(depth + 1, i + 1, cur);
            cur.pop_back();
        }


    }
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        this->nums = nums;
        this->n = nums.size();
        sort(nums.begin(), nums.end());
        vector<int> cur;
        result.push_back(cur);
        dfs(1, 0, cur);
        return result;
    }
};
/*
思路：回溯，一个个的查一遍
树:
每一层要做的，是拿到当前的字符,
看这个位置是
1.放数还是
2.点小数点

*/
/*
class Solution93 {
public:

    vector<string> result;

    void dfs(int index, int part, int partnum, string& curs, string& s) {
        if (index == s.size()) {
            result.push_back(curs);
            return;
        }
        char c = s[index];
        if (partnum == -1) {
            //没数,直接塞进去这个数看下一个了
            curs += c;
            partnum = c - '0';
            dfs(index + 1, part, partnum, curs, s);
        }
        else {
            //看看放数行不行
            if (partnum == 0 || (partnum * 10 + c - '0') > 255) {
                //放数不行，放点行不行
                if (part < 4) {
                    curs.push_back('.');
                    dfs(index + 1, part + 1, -1, curs, s);
                }
                else {
                    return;
                }
            }
            else {
                //放数了
                partnum = partnum * 10 + c - '0';
                curs.push_back(c);
                dfs(index + 1, part + 1, -1, curs, s);
            }

        }

        while (curs.back() == '.') curs.pop_back();
    }

    vector<string> restoreIpAddresses(string s) {


    }
};
*/
/*******************数组专题*****************************/

class Solution867 {
public:
    vector<vector<int>> transpose(vector<vector<int>>& matrix) {
        vector<vector<int>> ret(matrix[0].size(), vector<int>(matrix.size()));
        for (int i = 0; i < matrix.size(); i++) {
            for (int j = 0; j < matrix[0].size(); j++) {
                ret[j][i] = matrix[i][j];
            }
        }
        return ret;
    }
};
 


//88. 合并两个有序数组
class Solution88 {
public:
	void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
		int index1 = m - 1;
		int index2 = n - 1;
		int cur = m + n - 1;
		while (index1 >= 0 && index2 >= 0) {
			if (nums1[index1] > nums2[index2]) {
				nums1[cur] = nums1[index1];
				index1--;
			}
			else {
				nums1[cur] = nums2[index2];
				index2--;
			}
			cur--;
		}
		if (index1 < 0) {
			while (index2 >= 0) {
				nums1[cur] = nums2[index2];
				index2--;
				cur--;
			}
		}

	}
}; 
//90. 子集 II
/*
思路：
	组合排列
	去除同一层中重复的情况
	对于每一层的情况，都算作结果,
	对于去重的话，同一层，去重的原则应该是，我和前面的重复了的话就要去除，而对于不同层的不应该去重，具体
*/
class Solution90duplicate {
public:
	vector<int> nums;
	int n;
	vector<vector<int>> result;
	//每一层的任务就是看看能不能往下一层放东西
	void dfs(int depth,int start,vector<int> cur) {
		if (depth == n + 1) {
			//最后一层
			return;
		}
		for (int i = start; i < n; i++) {
			if (i > start && nums[i] == nums[i - 1]) {
				continue;
			}
			//符合要求
			cur.push_back(nums[i]);
			result.push_back(cur);
			dfs(depth + 1, i + 1, cur);
			cur.pop_back();
		}


	}
	vector<vector<int>> subsetsWithDup(vector<int>& nums) {
		this->nums = nums;
		this->n=nums.size();	
		vector<int> cur;
		result.push_back(cur);
		dfs(1, 0, cur);
		return result;
	}
};

class Solution91 {
public:
	int numDecodings(string s) {
		string s1 = "99"+s;
		int n = s1.size();
		vector<int> dp(n);
		dp[0] = 1;
		dp[1] = 1;				
		for (int i = 2; i < n; i++)
		{
			int pre = s1[i - 1] - '0';
			int cur = s1[i] - '0';
			int num = pre * 10 + cur;
			//当前数字可以和前面的组成有效数字
			if (pre != 0 && num <= 26) {
				//	1.当前数字是0			dp[i] =dp[i - 2]
				if (s1[i] == '0') {
					dp[i] = dp[i - 2];
				}
				//	2.当前数字不是0			dp[i] = dp[i - 1] + dp[i - 2];
				else
				{
					dp[i] = dp[i - 1] + dp[i - 2];
				}
			}
			else
			{
				//当前数字不能和前面数字组成有效数字
				//	1.当前数字是0			return 0
				if (s1[i] == '0') {
					return 0;
				}
				//	2.当前数字不是0			dp[i]=dp[i-1]
				else
				{
					dp[i] = dp[i - 1];
				}				
			}		
		}
		for (int num : dp) {
			cout << num << "\t";
		}
		return dp[n-1];
	}
};

/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution92 {
	/*
	思路：拿头拿尾，剩下的倒插

	*/
public:
	
	ListNode* reverseBetween(ListNode* head, int m, int n) {
		ListNode* preHead=new ListNode(0, head);
		ListNode* preLeft = preHead;
		for (int i = 0; i < m - 1; i++) {
			preLeft = preLeft->next;
		}
		ListNode* leftOrigin = preLeft->next;
		//开始倒插
		ListNode* cur = preLeft->next;
		preLeft->next = NULL;
		ListNode* temp = preLeft->next;
		for (int i = m; i <= n; i++) {
			preLeft->next = cur;
			cur = cur->next;
			preLeft->next->next = temp;
			temp = preLeft->next;
		}
		leftOrigin->next = cur;
		return preHead->next;
	}
};

//数组专题
class Solution238 {
public:
	vector<int> productExceptSelf(vector<int>& nums) {
		int n = nums.size();
		vector<int> left(n, 1);//表示第i个数前面i-1个数的乘积，left[0]=1，left[1]=nums[0],left[2]=left[1]*nums[1]
		vector<int> ret(n);
		for (int i = 1; i < n; i++) {
			left[i] = nums[i-1] * left[i - 1];
		}
		int right = 1;
		for (int i = n - 1; i >= 0; i--) {
			ret[i] = right * left[i];
			right = right * nums[i];
		}
		return ret;
	}
};
class Solution179 {
public:
	string largestNumber(vector<int>& nums) {
		sort(nums.begin(), nums.end(), compare2nums);

		string ret;
		if (nums[0] == 0) {
			return "0";
		}
		else {
			for (int data : nums) {
				ret += to_string(data);
			}
			return ret;
		}
	}
	static bool compare2nums(int& a, int& b) {
		string comp1 = to_string(a) + to_string(b);
		string comp2 = to_string(b) + to_string(a);
		return comp1 > comp2;//a+b>b+a 那么就认为a>b为true
	}
};
/*
 * @lc app=leetcode.cn id=498 lang=cpp
 *
 * [498] 对角线遍历
 */

 // @lc code=start
class Solution498 {
public:
    vector<int> findDiagonalOrder(vector<vector<int>>& matrix) {
        int m, n;
        if ((m = matrix.size()) == 0 || (n = matrix[0].size()) == 0)
        {
            return {};
        }
        vector<int> ret(m * n);
        enum {
            UP = 0,
            DOWN
        };
        int dir = UP;//true为往上，false为往下
        int x = -1, y = 1;
        for (int i = 0; i < m * n; i++)
        {
            if (dir == UP) {
                x += 1;
                y -= 1;
                if (y == -1 && x == n) {
                    //右上角
                    x -= 1;
                    y += 2;
                    dir = DOWN;
                }
                else {
                    if (y == -1) {
                        //上边界

                        y = 0;
                        dir = DOWN;
                    }
                    if (x == n) {
                        //右边界
                        x = n - 1;
                        y += 2;
                        dir = DOWN;
                    }
                }

            }
            else {
                x -= 1;
                y += 1;
                if (y == m && x == -1) {
                    //左下角
                    x += 2;
                    y = m - 1;
                    dir = UP;
                }
                else
                {
                    if (y == m) {
                        //下边界
                        x += 2;
                        y = m - 1;
                        dir = UP;
                    }
                    if (x == -1) {
                        //左边界
                        x = 0;
                        dir = UP;
                    }
                }

            }
            ret[i] = matrix[y][x];
        }
        return ret;
    }
};

class Solution334 {
public:
	bool increasingTriplet(vector<int>& nums) {
		int n = nums.size();
		if (n < 3) {
			return false;
		}
		int min1 = INT_MAX;
		int min2 = INT_MAX;
		for (int i = 0; i < n; i++) {
			if (nums[i] > min2) {
				return true;
			}
			if (nums[i] < min1) {
				min1 = nums[i];
			}
			else if (nums[i] > min1 && nums[i] < min2) {
				min2 = nums[i];
			}
		}
		return false;
	}
};
class Solution442 {
public:
	vector<vector<int>> ret;
	vector <vector<int>> mpeople;
	static bool compareHeight(vector<int>& p1, vector<int>& p2) {
		//从小到大排个
		if (p1[0] < p2[0]) {
			return true;
		}
		else if (p1[0] > p2[0]) {
			return false;
		}
		else { /*(p1[0] == p2[0])*/
			//同号的第二位从大到小排，这样减去的时候符合逻辑
			if (p1[1] > p2[1]) {
				return true;
			}
			else {
				return false;
			}
		}
	}
	void findHead(vector<vector<int>>& people, vector<bool>& isInQueen) {
		for (int i = 0; i < mpeople.size(); i++) {
			//找排头            
			if (isInQueen[i] == false) {
				if (mpeople[i][1] != 0) {
					mpeople[i][1] -= 1;
				}
				else if (mpeople[i][1] == 0) {
					ret.push_back(people[i]);
					int index = i;
					isInQueen[i] = true;
					findHead(people, isInQueen);
					break;
				}
			}
		}
	}
	vector<vector<int>> reconstructQueue1(vector<vector<int>>& people) {
		sort(people.begin(), people.end(), compareHeight);
		mpeople = people;
		vector<bool> isInQueen(people.size(), false);
		findHead(people, isInQueen);
		return ret;
	}

	vector<vector<int>> reconstructQueue(vector<vector<int>>& people) {
		//身高从低到高，排位从高到低
		sort(people.begin(), people.end(), [](vector<int> p1, vector<int> p2)
			{
				return p1[0] < p2[0] || (p1[0] == p2[0] && p1[1] > p2[1]);
			}
		);
		vector<vector<int>> ret(people.size());
		for (auto person : people) {
			int n = person[1];
			//找n个空位留给比他个高的
			for (int i = 0; i < people.size(); i++) {
				if (n == 0 && ret[i].empty()) {
					ret[i] = person;
					break;
				}
				if (ret[i].empty()) {
					n--;
				}
			}
		}
		return ret;
	}
};
class Solution105 {
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        return getHead(0, preorder.size() - 1, 0, inorder.size() - 1, preorder, inorder);
    }
    /*
        先序遍历的第一个为根节点，左子树的范围为中序遍历中找到的根节点的位置
        左闭右开
    */
    TreeNode* getHead(int pstart, int pend, int istart, int iend, vector<int>& preorder, vector<int>& inorder) {
        if (pstart > pend) {
            return {};
        }
        else if (pstart == pend) {
            return new TreeNode(preorder[pstart]);
        }
        TreeNode* head = new TreeNode(preorder[pstart]);
        int i;
        //中序遍历寻找根节点的位置，找到值为val的序号
        for (i = istart; i <= iend; i++) {
            if (inorder[i] == head->val) {
                break;
            }
        }

        int leftNum = i - istart;
        int rightNum = iend - i;
        //左子树的范围[pstart,i-1]
        head->left = getHead(pstart + 1, pstart + leftNum, istart, istart + leftNum - 1, preorder, inorder);
        //右子树的范围[i+1,pend]
        head->right = getHead(1 + pstart + leftNum, pend, istart + leftNum + 1, iend, preorder, inorder);
        return head;
    }
};
/*
 * @lc app=leetcode.cn id=106 lang=cpp
 *
 * [106] 从中序与后序遍历序列构造二叉树
 * [9,3,15,20,7]
 * [9,15,7,20,3]
 * [3,9,20,null,null,15,7]
 */

class Solution106 {
public:
    unordered_map<int, int> map;
    TreeNode* buildTree1(vector<int>& inorder, vector<int>& postorder) {
        if (inorder.size() != postorder.size()) {
            //错误输入
            return {};
        }
        for (int i = 0; i < inorder.size(); i++) {
            map[inorder[i]] = i;
        }
        int n = inorder.size();
        return findHead(0, n - 1, 0, n - 1, inorder, postorder);
    }
    TreeNode* findHead(int iStart, int iEnd, int pStart, int pEnd, vector<int>& inorder, vector<int>& postorder) {
        if (iStart > iEnd) {
            return {};
        }
        else if (iStart == iEnd) {
            return new TreeNode(inorder[iStart]);
        }
        int i = map[postorder[pEnd]] - iStart;
        // 中序遍历的istart+i即为头结点
        TreeNode* head = new TreeNode(inorder[iStart + i]);
        head->left = findHead(iStart, iStart + i - 1, pStart, pStart + i - 1, inorder, postorder);
        head->right = findHead(iStart + i + 1, iEnd, pStart + i, pEnd - 1, inorder, postorder);
        return head;
    }
    //规定好循环不变量后尝试，顺序后切割重组
};
/*
输入: 5
输出:
[
row
0	 [1],
1	[1,1],
2   [1,2,1],
3  [1,3,3,1],
4 [1,4,6,4,1]
]
*/
class Solution118 {
public:
	vector<vector<int>> generate(int numRows) {
		if (numRows <= 0) {
			return {};
		}

		vector<vector<int>> ret(numRows);
		for (int row = 0; row < numRows; row++) {
			ret[row] = vector<int>(row + 1, 1);
			for (int i = 1; i <= row - 1; i++) {
				ret[row][i] = ret[row - 1][i - 1] + ret[row - 1][i];
			}
		}
		return ret;
	}
};
class Solution119 {
public:
	vector<int> getRow(int rowIndex) {
		if (rowIndex < 1) {
			return {};
		}
		vector<int> ret(rowIndex + 1, 1);
		for (int row = 0; row <= rowIndex; row++) {
			for (int i = row - 1; i > 0; i--) {
				ret[i] = ret[i - 1] + ret[i];
			}
		}
		return ret;
	}
};
/*
输入：triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
输出：11
*/
class Solution120 {
public:
	int minimumTotal(vector<vector<int>>& triangle) {
		//只有一行或者0行
		int n = triangle.size();
		if (n == 1) {
			return triangle[0][0];
		}

		vector<vector<int>> dp(n);
		dp[n - 1] = triangle[n - 1];
		for (int row = n - 2; row >= 0; row--) {
			dp[row].resize(triangle[row].size());
			for (int i = 0; i < dp[row].size(); i++) {
				dp[row][i] = triangle[row][i] + min(dp[row + 1][i], dp[row + 1][i + 1]);
			}
		}
		return dp[0][0];
	}
};
class Solution121 {
public:
	int maxProfit(vector<int>& prices) {
		int maxPro = 0;
		int minIncome = INT_MAX;
		for (int i = 0; i < prices.size(); i++) {
			minIncome = min(minIncome, prices[i]);
			maxPro = max(maxPro, prices[i] - minIncome);
		}
		return maxPro;
	}
};
class Solution122 {
public:
	int maxProfit(vector<int>& prices) {
		int maxPro = 0;
		for (int i = 1; i < prices.size(); i++) {
			if (prices[i] > prices[i - 1]) {
				//交易
				maxPro += prices[i] - prices[i - 1];
			}
		}
		return maxPro;
	}
};
class Solution123 {
public:
	//[3,3,5,0,0,3,1,4]  6
	// 4 6  
	//7  8
	int maxProfit(vector<int>& prices) {
		int n = prices.size();
		vector<int> leftMaxPro(n);
		vector<int> rightMaxPro(n);
		int maxPro = 0;
		int minIncome = INT_MAX;
		for (int i = 0; i < prices.size(); i++) {
			minIncome = min(minIncome, prices[i]);
			maxPro = max(maxPro, prices[i] - minIncome);
			leftMaxPro[i] = maxPro;
		}
		int maxIncome = 0;
		maxPro = 0;
		for (int i = n - 1; i >= 0; i--) {
			maxIncome = max(prices[i], maxIncome);
			maxPro = max(maxPro, maxIncome - prices[i]);
			rightMaxPro[i] = maxPro;
		}
		maxPro = 0;
		for (int i = 0; i < n; i++) {
			maxPro = max(maxPro, leftMaxPro[i] + rightMaxPro[i]);
		}
		return maxPro;
	}
};
//单词接龙
/*
class Solution126 {
public:
	vector<vector<string>> findLadders(string beginWord, string endWord, vector<string>& wordList) {

	}
};
*/

class Solution908 {
public:
	int smallestRangeI(vector<int>& A, int K) {
		int minVal = INT_MAX;
		int maxVal = INT_MIN;
		for (int data : A) {
			minVal = min(data, minVal);
			maxVal = max(data, maxVal);
		}
		if ((maxVal - minVal) > 2 * K) return (maxVal - minVal - 2 * K);
		else return 0;
	}
};
class Solution910 {
public:
	// 前面的肯定要+K，后面的肯定要-K，如果相对考虑的话就是前面的不变，后面的-2K
	int smallestRangeII(vector<int>& A, int K) {
		if (A.size() == 1) {
			return 0;
		}
		sort(A.begin(), A.end());
		int n = A.size();
		int ret = INT_MAX;
		int maxVal = INT_MIN;
		int minVal = INT_MAX;
		for (int i = 0; i < n; i++) {
			maxVal = max(maxVal, A[i]);
			minVal = min(minVal, A[i]);
		}
		ret = maxVal - minVal;
		for (int i = 0; i < n - 1; i++) {
			maxVal = max(A[i], A[n - 1] - 2 * K);
			minVal = min(A[i + 1] - 2 * K, A[0]);
			ret = min(ret, maxVal - minVal);
		}
		return ret;
	}
};
class Solution153 {
public:
	//最右边的肯定小于等于最左边的
	int findMin1(vector<int>& nums) {
		int i = 0;
		for (i = 1; i < nums.size(); i++) {
			if (nums[i - 1] > nums[i]) {
				break;
			}
		}
		if (i != nums.size())	return nums[i];
		else return nums[0];
	}
	//找坑
	int findMin(vector<int>& nums) {
		int left = 0;
		int right = nums.size() - 1;
		int mid = 0;
		while (left < right) {
			mid = (left + right) / 2;
			if (nums[mid] >= nums[left]) {
				//左边升
				if (mid + 1 <= right && nums[mid] > nums[mid + 1]) {
					return nums[mid + 1];
				}
				else {
					left = mid + 1;
				}
			}
			else if (nums[mid] < nums[right])
			{
				//右边升
				if (mid - 1 >= left && nums[mid - 1] > nums[mid]) {
					return nums[mid];
				}
				else {
					right = mid - 1;
				}
			}
			else {
				//两边等
				left++;
			}

		}
		return nums[0];
	}
};
/********************开学刷专题************************/
/*
思路：二分查找
    边界条件练习
*/
class Solution35 {
public:
    //如果我认为target在[left,right]里，那么我的判断条件应该定为left<=right，如果出了我的判断，那么只可能会是 right在left的左边,这时候最佳的插入位置为left或者right+1;
    int searchInsert1(vector<int>& nums, int target) {
        int left = 0;
        int right = nums.size() - 1;
        int mid = 0;
        while (left <= right)
        {
            mid = (left + right) / 2;
            if (target < nums[mid]) {
                right = mid - 1;
            }
            else if (target > nums[mid]) {
                left = mid + 1;
            }
            else{
                return mid;
            }
        }
        return right + 1;
    }
	//同理如果我认为target在[left,right)里,那么判断条件应该为left<right,这样的话如果出了我的判断结束条件，只可能会是left=right，这时候最佳的插入位置为left或者right都可以
	int searchInsert2(vector<int>& nums, int target) {
		int left = 0;
		int right = nums.size();
		int mid = 0;
        while (left < right) {
            mid = (left + right) / 2;
            if (target < nums[mid]) {
                right = mid;
            }
            else if (target > nums[mid]) {
                left = mid + 1;
            }
            else {
                return mid;
            }
        }
        return right;
	}

};

class Solution206 {
public:
	ListNode* reverseList(ListNode* head) {
		if (head == NULL) {
			return head;
		}
		ListNode* pre = NULL;
		ListNode* cur = head;
		ListNode* later = head->next;
		while (cur != NULL) {
			later = cur->next;
			cur->next = pre;
			pre = cur;
			cur = later;
		}
		return pre;
	}
};
class Solution209 {
public:
	int minSubArrayLen(int target, vector<int>& nums) {
        int left = 0;
        int right = -1;
        int sum = 0;
        int len = INT_MAX;
        int n = nums.size();
        
        while (right < n && left < n) {
            if (sum < target) {
				right++;
                if (right < n)	sum += nums[right];
			}
			else {
                len = min(len, right - left + 1);
				sum -= nums[left];
				left++;
			}
        }
        return len == INT_MAX ? 0:len;
	}
};
/*
class Solution59 {
public:
	vector<vector<int>> generateMatrix(int n) {
	vector<vector<int>> result(n,vector<int>(n));
	int left = 0, right = n - 1, up = 0, down = n - 1;
	enum {RIGHT=0,DOWN,LEFT,UP};
	int state = RIGHT;
	int index = 1;
	while (left <= right && up <= down) {
		switch (state % 4) {
		case RIGHT:
			for (int i = up, j = left; j <= right; j++) {
				result[i][j] = index++;
			}
			up++;
			break;
		case DOWN:
			for (int i = up, j = right; i <= down; i++) {
				result[i][j] = index++;
			}
			right--;
			break;
		case LEFT:
			for (int i = down, j = right; j >= left; j--) {
				result[i][j] = index++;
			}
			down--;
			break;
		case UP:
			for (int i = down, j = left; i >= up; i--){
				result[i][j] = index++;
			}
			left++;
			break;
		}
		state++;
	}
	return result;
	}
};
*/
class Solution59 {
public:
	vector<vector<int>> generateMatrix(int n) {
        vector<vector<int>> matrix(n, vector<int>(n));
        int x = 0;//横向
        int y = 0;//纵向
        int len = n - 1;
        int num = 1;

        for (int cir = 0; cir < (n + 1) / 2; cir++) {//层数            
			for (int dir = 0; dir < 4; dir++) {//方向
				for (int i = 0; i < len; i++) {
                    matrix[x][y++] = num++;
				}
                for (int i = 0; i < len; i++) {
                    matrix[x++][y] = num++;
                }
				for (int i = 0; i < len; i++) {
					matrix[x][y--] = num++;
				}
				for (int i = 0; i < len; i++) {
					matrix[x--][y] = num++;
				}
                len -= 2;
			}
        }
        return matrix;
	}
};
//链表
class Solution203 {
public:
	ListNode* removeElements(ListNode* head, int val) {
        ListNode* preHead = new ListNode(0,head);
        ListNode* p = preHead;
        while (p->next != NULL) {
            if (p->next->val == val) {
                ListNode* temp = p->next;
                p->next = temp->next;
                delete temp;
            }
        }
        ListNode*  ret = preHead->next;
        delete preHead;
        return ret;
	}
};
//Solution707
class MyLinkedList {
public:
    struct LinkedNode {
        int val;
        LinkedNode* next;
        LinkedNode(int val):val(val), next(nullptr){}
    };
    /** Initialize your data structure here. */
    MyLinkedList() {
        preHead = new LinkedNode(0);
        m_size = 0;
    }

    /** Get the value of the index-th node in the linked list. If the index is invalid, return -1. */
    int get(int index) {
        LinkedNode* cur = preHead;
        if (index<0 || index>=m_size) {
            return -1;
        }
        while (index--)
        {
            cur = cur->next;
        }
        return cur->next->val;

    }

    /** Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list. */
    void addAtHead(int val) {
        m_size++;
        LinkedNode* temp = preHead->next;
        preHead->next = new LinkedNode(val);
        preHead->next->next = temp;

    }

    /** Append a node of value val to the last element of the linked list. */
    void addAtTail(int val) {
        m_size++;
        LinkedNode* node = new LinkedNode(val);
        LinkedNode* cur = preHead;
        while (cur->next != NULL)
        {
            cur = cur->next;
        }
        cur->next = node;
    }

    /** Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted. */
    void addAtIndex(int index, int val) {
        if (index > m_size) {
            return;
        }
        else if (index == m_size) {
            addAtTail(val);
        }
        else if (index <= 0) {
            addAtHead(val);
        }
        else{
            LinkedNode* cur = preHead;
            while (index--) {
                cur = cur->next;
            }
            LinkedNode* temp = cur->next;
            LinkedNode* newNode = new LinkedNode(val);
            cur->next = newNode;
            newNode->next = temp;
            m_size++;
        }
       
    }

    /** Delete the index-th node in the linked list, if the index is valid. */
    void deleteAtIndex(int index) {
        if (index < 0 || index >= m_size) {
            return;
        }
        LinkedNode* cur = preHead;
        while (index--) {
            cur = cur->next;
        }
        LinkedNode* delNode = cur->next;
        cur->next = delNode->next;
        delete delNode;
        m_size--;
    }
    void printList() {
        LinkedNode* cur = preHead->next;
        cout << m_size << endl;
        while (cur!=NULL)
        {
            cout << cur->val << "\t";
            cur = cur->next;
        }

    }

private:
    LinkedNode* preHead;
    int m_size;
};

/**
 * Your MyLinkedList object will be instantiated and called as such:
 * MyLinkedList* obj = new MyLinkedList();
 * int param_1 = obj->get(index);
 * obj->addAtHead(val);
 * obj->addAtTail(val);
 * obj->addAtIndex(index,val);
 * obj->deleteAtIndex(index);
 */
class Solution142 {
public:
	ListNode* detectCycle(ListNode* head) {
        ListNode* fast = head;
        ListNode* slow = head;

        while (fast != NULL && fast->next != NULL) {
            fast = fast->next->next;
            slow = slow->next;
            if (fast == slow) {
                ListNode* index1 = head;
                ListNode* index2 = slow;
                while (index1 != index2) {
                    index1 = index1->next;
                    index2 = index2->next;
                }
                return index1;
            }
        }
        return NULL;
	}
};

//map操作
class Solution242 {
public:
	bool isAnagram(string s, string t) {
        if (s.size() != t.size()) {
            return false;
        }

        unordered_map<char, int> smap;
        for (char c : s) {
            smap[c]++;
        }
        for (char c : t) {
            if (smap.find(c) == smap.end()||smap[c]==0) {
                return false;
            }
            else {
                smap[c]--;
            }
        }
        return true;
	}
};

class Solution1 {
public:
	vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> mmap;
        for (int i = 0; i < nums.size(); i++) {
            auto iter = mmap.find(target - nums[i]);
            if (iter != mmap.end()) {
                return { iter->second,i };
            }
            mmap.insert({ nums[i],i });
        }
        return {};
	}
};
//set操作
class Solution349 {
public:
	vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
        unordered_set<int> result_set;
        unordered_set<int> nums_set(nums1.begin(), nums1.end());
        for (int i : nums2) {
            if (nums_set.find(i) != nums_set.end()) {
                result_set.insert(i);
            }
        }
        return vector<int>(result_set.begin(), result_set.end());
	}
};

class Solution202 {
public:
    int getnum(int num) {
        int sum = 0;
        while (num != 0) {
            sum += (num % 10) * (num % 10);
            num /= 10;
        }
        return sum;
    }

	bool isHappy(int n) {
        unordered_set<int> num_set;
        num_set.insert(n);
        while (true) {
            n = getnum(n);
            if (n == 1) {
                return true;
            }
            if (num_set.find(n) != num_set.end()) {
                return false;
            }
            num_set.insert(n);
        }
	}
};

class Solution454 {
public:
	int fourSumCount(vector<int>& A, vector<int>& B, vector<int>& C, vector<int>& D) {
        unordered_map<int,int> num_map;
        int cnt = 0;
        for (int i = 0; i < A.size(); i++) {
            for (int j = 0; j < B.size(); j++) {
                num_map[A[i] + B[j]]++;
            }
        }
        for (int i = 0; i < C.size(); i++) {
            for (int j = 0; j < D.size(); j++) {
                if (num_map.find(-(C[i] + D[j]))!= num_map.end()) {
                    cnt += num_map[-(C[i] + D[j])];
                }
            }
        }
        return cnt;
	}
};

class Solution383 {
public:
    //哈希方法，
	bool canConstruct1(string ransomNote, string magazine) {
        unordered_map<char, int> mag_map;
        for (char c : magazine) {
            mag_map[c]++;
        }
        for (char c : ransomNote) {
            if (mag_map.find(c) != mag_map.end()) {
                if (mag_map[c] != 0) {
                    mag_map[c]--;
                }
                else {
                    return false;
                }
            }
            else {
                return false;
            }
        }
        return true;;

	}
    //数组方法，哈希需要计算哈希函数占用额外空间，因此用数组
	bool canConstruct(string ransomNote, string magazine) {
        vector<int> record(26, 0);        
        for (char c : magazine) {
            record[c - 'a']++;
        }
        for (char c : ransomNote) {
            if (record[c - 'a'] == 0) {
                return false;
            }
            else {
                record[c - 'a']--;
            }
        }
        return true;
	}
};

class Solution15 {
public:
    //去重非常繁琐,尝试失败了
    /*
	vector<vector<int>> threeSum1(vector<int>& nums) {
        unordered_map<int, int> num_map;
        vector<vector<int>> result;
        for (int i : nums) {
            num_map[i]++;
        }
        for (auto it = num_map.begin(); it != num_map.end(); it++) {
            //拿出一个数
            it->second--;
            for (auto iit = it; iit != num_map.end(); iit++) {
                if (iit->second > 0) {
                    //拿出第二个数
                    iit->second--;
                    //找第三个数 
                    int num1 = it->first;
                    int num2 = iit->first;
                    auto num3_iter = num_map.find(-(num1 + num2));
                    if (num3_iter != num_map.end() && num3_iter->second > 0) {
                        result.push_back({ num1,num2,num3_iter->first });
                    }
                    //还回第二个数
                    iit->second++;
                }
            }
            //放回
            it->second++;
        }
        return result;
	}
    */
	vector<vector<int>> threeSum(vector<int>& nums) {
		if (nums.empty()) {
			return {};
		}
        sort(nums.begin(), nums.end());
        int left = 0;
        int right = nums.size() - 1;
		vector<vector<int>> result;
        for (int i = 0; i < nums.size() - 2; i++) {
			left = i + 1;
			right = nums.size() - 1;
			if ((nums[i] > 0 && nums[right] > 0)
				|| (nums[i] < 0 && nums[right] < 0)
				)
			{
				return result;
			}
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }

            while (left < right) {
                if (nums[i] + nums[left] + nums[right] > 0) {
                    right--;
                }
                else if (nums[i] + nums[left] + nums[right] < 0) {
                    left++;
				}else {
					result.push_back({ nums[i],nums[left],nums[right] });
                    while (left < right && nums[left] == nums[left + 1]) left++;
                    while (left < right && nums[right] == nums[right - 1])right--;
                    left++;
                    right--;
				}
            }

		}
        return result;
	}

};
class Solution344 {
public:
	void reverseString(vector<char>& s) {
        int n = s.size();
        for (int i = 0; i < n / 2; i++) {
            swap(s[i], s[n - 1 - i]);
        }
	}
};

class Solution541 {
public:
	string reverseStr1(string s, int k) {
        int n = s.size();
        int start = 0;
        for (int i = 0; i < n / (2 * k); i++) {
            start = i * (2 * k);
			for (int j = 0; j < k / 2; j++) {
                swap(s[start + j], s[start + k - 1 - j]);
			}
        }
        int last_len = n % (2 * k);
        start = (n / (2 * k)) * 2 * k;
        if (last_len >= k) {
			for (int j = 0; j < k / 2; j++) {
				swap(s[start + j], s[start + k - 1 - j]);
			}
        }
        else {
			for (int j = 0; j < last_len / 2; j++) {
                swap(s[start + j], s[start + last_len - 1 - j]);
			}
        }
        return s;

	}
	string reverseStr(string s, int k) {
        int n = s.size();
        for (int i = 0; i < s.size(); i += 2 * k) {
            if (i + k <= s.size() - 1) {
                reverse(s.begin() + i, s.begin() + i + k);
            }
            else {
                reverse(s.begin() + i, s.end());
            }
        }
        return s;
	}
};
class Solution151 {
public:
	string reverseWords(string s) {
        //去除后面的空格和前面的空格
		while (s.back() == ' ') {
			s.erase(s.end() - 1);
		}
		while (s[0] == ' ') {
			s.erase(s.begin());
		}
		for (int i = 0; i < s.size(); i++) {
			//清除掉多余的空格
			if (s[i] == ' ') {
				while (i + 1 < s.size() && s[i + 1] == ' ') {
					s.erase(s.begin() + i + 1);
				}
			}
		}
        reverse(s.begin(), s.end());
        int lastSpace = -1;
        for (int i = 0; i < s.size(); i++) {
            if (s[i] == ' ') {
                reverse(s.begin() + lastSpace + 1, s.begin() + i);
                lastSpace = i;
            }
        }
        reverse(s.begin() + lastSpace + 1, s.end());
        return s;
	}

};

class Solution27 {
public:
	int removeElement(vector<int>& nums, int val) {
        int start = 0;
        for (start = 0; start < nums.size(); start++) {
            if (nums[start] == val) {
                break;
            }
        }
        int slowIndex = start;
        for (int fastIndex = start; fastIndex < nums.size(); fastIndex++) {
            if (nums[fastIndex] != val) {
                nums[slowIndex++] = nums[fastIndex];
            }
        }
        return slowIndex;
	}
};

//232solution
//个人理解成两个对着屁股的口就挺好理解 [  ]这种，塞进去的话往左边塞，弹出的话先都弄到右边来，然后从右边口出就可以
class MyQueue {
public:
    /** Initialize your data structure here. */
	stack<int> stIn;
	stack<int> stOut;
    MyQueue() {

    }

    /** Push element x to the back of queue. */
    void push(int x) {
        stIn.push(x);
    }

    /** Removes the element from in front of queue and returns that element. */
    int pop() {
        if (stOut.empty()) {
            while (!stIn.empty()) {
                stOut.push(stIn.top());
                stIn.pop();
            }
        }
        if (!stOut.empty()) {
            int ret = stOut.top();
            stOut.pop();
            return ret;
        }
        else {
            return -1;
        }
    }

    /** Get the front element. */
    int peek() {
        int ret = pop();
        stOut.push(ret);
        return ret;
    }

    /** Returns whether the queue is empty. */
    bool empty() {
        return stIn.empty() && stOut.empty();
    }
};

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue* obj = new MyQueue();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->peek();
 * bool param_4 = obj->empty();
 */
//Solution225
class MyStack {
public:
	queue<int> q1;
	queue<int> q2; // 辅助队列，用来备份
    /** Initialize your data structure here. */
    MyStack() {
        
    }

    /** Push element x onto stack. */
    void push(int x) {
        if (q1.empty()) {
            q2.push(x);
        }
        else {
            q1.push(x);
        }
    }

    /** Removes the element on top of the stack and returns that element. */
    int pop() {
        int ret = -1;
        if (q1.empty()) {
            while (q2.size()>1)
            {
                q1.push(q2.front());
                q2.pop();
            }
            ret = q2.front();
            q2.pop();
        }
        else {
			while (q1.size() > 1)
			{
				q2.push(q1.front());
				q1.pop();
			}
			ret = q1.front();
			q1.pop();
        }
        return ret;
    }

    /** Get the top element. */
    int top() {
        int ret = pop();
        push(ret);
        return ret;
    }

    /** Returns whether the stack is empty. */
    bool empty() {
        return q1.empty() && q2.empty();
    }
};

/**
 * Your MyStack object will be instantiated and called as such:
 * MyStack* obj = new MyStack();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->top();
 * bool param_4 = obj->empty();
 */

class Solution20 {
public:
	bool isValid(string s) {
        stack<char> st;
        unordered_map<char, char> m_map =
        {
            {'{','}'},
            {'(',')'},
            {'[',']'}
        };

        for (char c : s) {
            if (c == '(' || c == '{' || c == '[') {
                st.push(m_map[c]);
            }
            else {
                if (st.empty() || st.top() != c) return false;
                st.pop();
            }
        }
        if (st.empty())  return true;
        else return false;
	}
};
class Solution1047 {
public:
	string removeDuplicates(string S) {
        stack<char> st;
        for (char c : S) {
            if (st.empty() || st.top() != c) {
                st.push(c);
            }
            else {
                st.pop();
            }
        }
        string result = "";
        while (!st.empty()) {
            result += st.top();
            st.pop();
        }
        reverse(result.begin(), result.end());
        return result;

	}
};
class Solution150 {
public:
	int evalRPN(vector<string>& tokens) {
        stack<int> num_st;
        for (string s : tokens) {
            if (s == "+" || s == "-" || s == "*" || s == "/") {
                int num2 = num_st.top();
                num_st.pop();
                int num1 = num_st.top();
                num_st.pop();
                int new_num;
                if (s == "+") new_num = num1 + num2;
                else if(s=="-") new_num = num1 - num2;
                else if (s == "*") new_num = num1 * num2;
                else if (s == "/") new_num = num1 / num2;
                num_st.push(new_num);
            }
            else {
                num_st.push(stoi(s));
            }
        }
        return num_st.top();

	}
};

class Solution239 {
public:
    //要点：新的大数的到来可以直接影响到前面的小数，让他们的离开变得无关紧要。毕竟窗口里面已经有了比他们大的数。
    class MyQueue {
    public: deque<int> que;
          //新的数进来,会使得比它小的数变得无关紧要，因为比它小的在它的前面比它先走，所以不可能是这个队列里的最大值了
          void push(int val) {
              while (!que.empty() && que.back() < val) {
                  que.pop_back();
              }
              que.push_back(val);
          }
          //左边窗口滑出新的数据
          //走的时候只有两种情况，要么是最大值，要么不是最大值。
          //如果是最大值的话，那应该弹出里面最大的，如果不是最大值的话，那后面的最大值肯定会早就把他弹出去了，因此他对最终的滑动窗口的最大值是没有影响的。所以不用管就可以。
          void pop(int val) {
              if (!que.empty() && que.front() == val) {
                  que.pop_front();
              }
          }
          int front() {
              return que.front();
          }

    };


	vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> result(nums.size() - k + 1);
        MyQueue mq;
        for (int i = 0; i < k; i++) {
            mq.push(nums[i]);
        }
        result[0] = mq.front();
        for (int i = k; i < nums.size(); i++) {
            mq.pop(nums[i - k]);
            mq.push(nums[i]);
            result[i - k + 1] = mq.front();
        }
        return result;
	}
};

//优先级队列
class Solution347 {
public:
	class myCompare {
    public:
		//第一个操作数我理解为子节点，第二个为父亲节点
		bool operator()(pair<int, int> lhs, pair<int, int> rhs) {
			return lhs.second > rhs.second;
		}
	};
	vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> numsMap;
        for (int i : nums) {
            numsMap[i]++;
        }

        priority_queue<pair<int, int>, vector<pair<int, int>>, myCompare> pri_que;
        for (auto it = numsMap.begin(); it != numsMap.end(); it++) {
            pri_que.push(*it);
            if (pri_que.size() > k) {
                pri_que.pop();
            }
        }
        vector<int> result(k);
        for (int i = k - 1; i >= 0; i--) {
            result[i] = pri_que.top().first;
            pri_que.pop();
        }
        return result;
	}
};

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
//先序遍历
class Solution144 {
public:
    void preOrder(TreeNode* root, vector<int>& result) 
    {
        if (root == NULL) {
            return;
        }
        result.push_back(root->val);
        preOrder(root->left, result);
        preOrder(root->right, result);
    }
    //递归写法
    vector<int> preorderTraversal1(TreeNode* root) 
    {
        vector<int> result;
        preOrder(root, result);
        return result;
    }
    //栈写法
	vector<int> preorderTraversal(TreeNode* root)
	{
        stack<TreeNode*> stk;
        vector<int> result;
        if(root!=NULL) stk.push(root);
        while (!stk.empty()) {
            TreeNode* node = stk.top();
            stk.pop();            
			if (node != NULL) {
                result.push_back(node->val);//中
				stk.push(node->right);//右
				stk.push(node->left);//左
            }
        }
        return result;
	}

};
//145,给定一个二叉树，返回它的 后序 遍历。
class Solution145 {
public:
    void postOrder(TreeNode* root, vector<int>& result) {
        if (root == NULL) {
            return;
        }
        postOrder(root->left, result);
        postOrder(root->right, result);
        result.push_back(root->val);
    }
    //递归写法
	vector<int> postorderTraversal1(TreeNode* root) {
		vector<int> result;
        postOrder(root, result);
		return result;
	}
	//栈写法
	vector<int> postorderTraversal(TreeNode* root) {
		stack<TreeNode*> stk;
		if (root) stk.push(root);
		vector<int> result;
		while (!stk.empty()) {
			TreeNode* node = stk.top();
			stk.pop();
			if (node != NULL) {
				if (node->right) stk.push(node->right);
				if (node->left) stk.push(node->left);
				stk.push(node);
				stk.push(NULL);
			}
			else {
				TreeNode* node = stk.top();
				stk.pop();
				result.push_back(node->val);
			}
		}
		return result;
	}

};
//94. 二叉树的中序遍历
class Solution94 {
public:
    void inOrder(TreeNode* root,vector<int>& result) {
        if (root == NULL) {
            return;
        }
        inOrder(root->left, result);
        result.push_back(root->val);
        inOrder(root->right, result);
    }
	vector<int> inorderTraversal1(TreeNode* root) {
		vector<int> result;
        inOrder(root, result);
		return result;
	}
    //栈写法
	vector<int> inorderTraversal(TreeNode* root) {
        stack<TreeNode*> stk;
        if (root) stk.push(root);
        vector<int> result;
        while (!stk.empty()) {
            TreeNode* node = stk.top();
            stk.pop();
            if (node != NULL) {
                if (node->right) stk.push(node->right);
                stk.push(node);
                stk.push(NULL);
                if (node->left) stk.push(node->left);
            }
            else{
                TreeNode* node= stk.top();
                stk.pop();
                result.push_back(node->val);
            }
        }
        return result;
	}
};

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution102 {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        queue<TreeNode*> que;
        if (root != NULL) que.push(root);
        vector<vector<int>> result;
        while (!que.empty())
        {
            int size = que.size();
            vector<int> vec;
            for (int i = 0; i < size; i++) {
                TreeNode* node=que.front();
                que.pop();
                vec.push_back(node->val);
                if (node->left) que.push(node->left);
                if (node->right) que.push(node->right);
            }
            result.push_back(vec);
        }
        return result;
    }
};

class Solution107 {
public:
	vector<vector<int>> levelOrderBottom(TreeNode* root) {
        queue<TreeNode*> que;
        vector<vector<int>> result;
        if (root != NULL) que.push(root);
        while (!que.empty())
        {
            int size = que.size();
            vector<int> vec;
            for (int i = 0; i < size; i++) {
                TreeNode* node = que.front();
                que.pop();
                vec.push_back(node->val);
                if (node->left) que.push(node->left);
                if (node->right) que.push(node->right);
            }
            result.push_back(vec);
        }
        reverse(result.begin(), result.end());
        return result;
	}
};

class Solution199 {
public:
	vector<int> rightSideView(TreeNode* root) {
        queue<TreeNode*> que;
        vector<int> result;
        if (root != NULL) que.push(root);
        while (!que.empty()) {
            int size= que.size();
            int num = 0;
            TreeNode* node;
            for (int i = 0; i < size; i++) {
                node = que.front();
				que.pop();                
                if(node->left) que.push(node->left);
                if(node->right) que.push(node->right);
            }
            result.push_back(node->val);
        }
        return result;
	}
};

class Solution637 {
public:
	vector<double> averageOfLevels(TreeNode* root) {
        queue<TreeNode*> que;
        vector<double> result;
        if(root!=NULL) que.push(root);
        while (!que.empty()) {
            int size = que.size();
            double ave = 0;
            for (int i = 0; i < size; i++) {
                TreeNode* node = que.front();
                que.pop();
                ave += node->val;
                if(node->left) que.push(node->left);
                if(node->right) que.push(node->right);
            }
            ave /= size;
            result.push_back(ave);
        }
        return result;

	}
};



class Solution429 {
public:
    // Definition for a Node.
	class Node {
	public:
		int val;
		vector<Node*> children;

		Node() {}

		Node(int _val) {
			val = _val;
		}

		Node(int _val, vector<Node*> _children) {
			val = _val;
			children = _children;
		}
	};

    vector<vector<int>> levelOrder(Node* root) 
    {
        queue<Node*> que;
        vector<vector<int>> result;
        if (root != NULL) que.push(root);
        while (!que.empty()) 
        {
            int size = que.size();
            vector<int> vec;
            for (int i = 0; i < size; i++) 
            {
				Node* node = que.front();
				que.pop();
                vec.push_back(node->val);
                for (int i = 0; i < node->children.size(); i++) 
                {
                    if (node->children[i]) que.push(node->children[i]);
                }
            }
            result.push_back(vec);
        }
        return result;

    }
};

class Solution226 {
public:
    //前序遍历
	TreeNode* invertTree(TreeNode* root) {
        if (root == NULL) return root;
        swap(root->left, root->right);
        invertTree(root->left);
        invertTree(root->right);
        return root;
	}
};

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution101 {
public:
    //使用递归方法
    bool isSymmetric1(TreeNode* root) {
        if (root == NULL) {
            return true;
        }
        return compare(root->left, root->right);

    }
    //比较左右子树内外是不是相同，相同就继续比较下一层，直到到了null
    //1.确定参数和返回值
    //2.确定终止条件，,就是不管怎么操作，你总会到了这一层
    //3.确定每一层的逻辑
    bool compare(TreeNode* left, TreeNode* right) {
        //终止条件
        if (left == NULL && right == NULL)  return true;
        else if (left == NULL || right == NULL) return false;
        else if (left->val != right->val) return false;
        //左右相等且不为空的情况
        //接着比较下一层了       
        return compare(left->left, right->right) && compare(left->right, right->left);
    }
    //使用迭代法
	bool isSymmetric(TreeNode* root) {
		if (root == NULL) {
			return true;
		}
        queue<TreeNode*> que;
        que.push(root->left);
        que.push(root->right);
        while (!que.empty())
        {
            TreeNode* lnode = que.front();
            que.pop();
            TreeNode* rnode = que.front();
            que.pop();
            //将要比较的节点按顺序依次放进去,一个比较机一样
            if(lnode==NULL&&rnode==NULL) continue;
            else if (lnode == NULL || rnode == NULL) {
                return false;
            }
            else if (lnode->val != rnode->val) {
                return false;
            }
            else{
                que.push(lnode->left);
                que.push(rnode->right);
                que.push(lnode->right);
                que.push(rnode->left);
            }

        }
        return true;
	}

};

class Solution104 {
public:
    int maxDepth(TreeNode* root) {
        if (root == NULL) {
            return 0;
        }
        return max(maxDepth(root->left), maxDepth(root->right)) + 1;
    }
    //迭代法
	int maxDepth1(TreeNode* root) {
        queue<TreeNode*> que;
        int depth = 0;
        if(root!=NULL) que.push(root);
        while (!que.empty()) {
            int size = que.size();
            for (int i = 0; i < size; i++) {
				TreeNode* node = que.front();
				que.pop();
                if (node->left) que.push(node->left);
                if (node->right) que.push(node->right);
            }
            depth++;
        }
        return depth;
	}

};


class Solution559 {
public:
    //递归法
    int maxDepth(Node* root) {
        if (root == NULL) {
            return 0;
        }
        int depth = 0;
        for (int i = 0; i < root->children.size(); i++) {
            depth = max(maxDepth(root->children[i]), depth);
        }
        return depth + 1;
    }


    //迭代法
    int maxDepth2(Node* root) {
        queue<Node*> que;
        int depth = 0;
        if(root) que.push(root);
        while (!que.empty()) {
            int size = que.size();
            for (int i = 0; i < size; i++) {
				Node* node = que.front();
				que.pop();
                if (node != NULL) {
                    for (int i = 0; i < node->children.size(); i++) {
                        que.push(node->children[i]);
                    }
                }
            }
            depth++;
        }
        return depth;
    }
};
class Solution111 {
public:
    int minDepth(TreeNode* root) {    
        if (root == NULL) return 0;
        int leftDepth = minDepth(root->left);
        int rightDepth = minDepth(root->right);
		if (root->left == NULL && root->right != NULL) {
			return rightDepth + 1;
		}
		if (root->left != NULL && root->right == NULL) {
			return leftDepth + 1;
		}
        return min(leftDepth, rightDepth) + 1;
    }
    //迭代法
	int minDepth1(TreeNode* root) {
        queue<TreeNode*> que;
        if (root != NULL)  que.push(root);
        int depth = 0;
        while (!que.empty()) {
            int size = que.size();  
            depth++; 
			for (int i = 0; i < size; i++) {
				TreeNode* node = que.front();
				que.pop();
				if (node->left) que.push(node->left);
				if (node->right) que.push(node->right);
                if (node->left==NULL && node->right==NULL) return depth;
			}           
        }
        return depth;
	}
};

class Solution222 {
public:
    //递归法
	int countNodes1(TreeNode* root) {
        if (root == NULL) return 0;
        int leftNodes = countNodes(root->left);
        int rightNodes = countNodes(root->right);
        return leftNodes + rightNodes + 1;
	}
    //迭代法
	int countNodes(TreeNode* root) {
        queue<TreeNode*> que;
        int nodeCnt = 0;
        if (root != NULL) que.push(root);
        while (!que.empty()) {
            int size = que.size();
            nodeCnt+=size;
            for (int i = 0; i < size; i++) {
                TreeNode* node = que.front();
                que.pop();
                if (node->left) que.push(node->left);
                if (node->right) que.push(node->right);
            }
        }
        return nodeCnt;

	}
};
class Solution110 {
public:
    int getdepth(TreeNode* root) {
        //结束条件
        if (root == NULL) {
            return 0;
        }
        //内部逻辑
        int leftDepth = getdepth(root->left);
        int rightDepth = getdepth(root->right);
        if (leftDepth == -1 || rightDepth == -1) return -1;
        if (abs(leftDepth - rightDepth) > 1) return -1;
        return max(leftDepth, rightDepth) + 1;

    }
    //递归方法
	bool isBalanced1(TreeNode* root) {
        return getdepth(root) == -1 ? false: true;
	}
};
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution257 {
public:
    vector<string> result;
    void trans2Res(vector<int>& path) {
        string str;
        for (int i = 0; i < path.size() - 1; i++) {
            str = str + to_string(path[i]) + "->";
        }
        str += (to_string(path.back()));
        result.push_back(str);
    }
    void traversal(TreeNode* root, vector<int>& path) {
        //前序遍历
        path.push_back(root->val);//中        
        if (root->left == NULL && root->right == NULL) {
            trans2Res(path);
            return;
        }
        if (root->left) {
            traversal(root->left, path);//左,如果有的探索那么肯定会往里面加入一个元素，溯回来的时候要将其弹出
            path.pop_back();
        }
        if (root->right) {
            traversal(root->right, path);//右
            path.pop_back();
        }      
        
        return;
    }

    vector<string> binaryTreePaths1
    (TreeNode* root) {
		if (root == NULL) {
			return {};
		}
        vector<int> path;
        traversal(root, path);
        return result;
    }
    /*
    //非递归方法，尝试失败
	vector<string> binaryTreePaths(TreeNode* root) {
        stack<TreeNode*> stk;
        vector<int> path;
        if (root) stk.push(root);
        while (!stk.empty()) {
            //拿出当前节点
            TreeNode* node = stk.top();
            stk.pop();
            //中
            path.push_back(node->val);
            if (node->left == NULL && node->right == NULL) {
                trans2Res(path);                
                continue;
            }
            //右
            if (node->right) {
                stk.push(node->right);
            }
            //左
            if (node->right) {
                stk.push(node->right);
            }


        }
	}
    */
};
class Solution100 {
public:
	bool isSameTree(TreeNode* p, TreeNode* q) {
        //终止条件
        if (p == NULL && q == NULL) {
            return true;
        }
        if (p == NULL || q == NULL) {
            return false;
        }
        //中
        if (p->val != q->val) {
            return false;
        }
		//左
        //右
        return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);       

	}
};
class Solution572 {
public:
    //比较两个树是不是相同的
    bool compare2Tree(TreeNode* s, TreeNode* t) {
        if (s == NULL && t == NULL) {
            return true;
        }
        if (s == NULL || t == NULL) {
            return false;
        }
        if (s->val != t->val) {
            return false;
        }
        return compare2Tree(s->left, t->left) && compare2Tree(s->right, t->right);

    }
    /*
    //节点值不相同的话
	TreeNode* findSubtree(TreeNode* s,int num) {
		if (s == NULL) {
            return {};
		}
        //中
        if (s->val == num) {
            return s;
        }
        //左
		TreeNode* lnode = findSubtree(s->left, num);
		TreeNode* rnode = findSubtree(s->right, num);
        return lnode == NULL ? rnode : lnode;
	}
    */
    bool dfs(TreeNode* s, TreeNode* t) {
        if (s == NULL) {
            return false;
        }
        return compare2Tree(s, t) || dfs(s->left, t) || dfs(s->right, t);;
    }

	bool isSubtree(TreeNode* s, TreeNode* t) {        
        return dfs(s, t);;
	}
};
class Solution404 {
public:
	int sumOfLeftLeaves1(TreeNode* root) {
        //终止条件
        if (root == NULL) {
            return 0;
        }
        //遍历方式：后序遍历,因为是到最后汇总
        int lsum = sumOfLeftLeaves(root->left);//左
        int rsum = sumOfLeftLeaves(root->right);//右
        //中,对于单个节点来说，如果是左叶子点，那么就讲它的值加进去
        if (root->left != NULL && root->left->left == NULL && root->left->right == NULL) {
            return lsum + rsum + root->left->val;
        }
        return lsum + rsum;
	}
    int sumOfLeftLeaves(TreeNode* root) {
        stack<TreeNode*> stk;
        if (root) stk.push(root);
        int sum = 0;
        while (!stk.empty()) {
            TreeNode* node = stk.top();
            stk.pop();
            if (node != NULL) {
				//右
				if (node->right) stk.push(node->right);
				//左
				if (node->left) stk.push(node->left);
                //中
                stk.push(node);
                stk.push(NULL);
            }
            else {
				//中
				TreeNode* node = stk.top();
				stk.pop();
				if (node->left != NULL && node->left->left == NULL && node->left->right == NULL) {
					sum += node->left->val;
				}
            }
        }
        return sum;
    }
};
class Solution513 {
public:
	int findBottomLeftValue(TreeNode* root) {
        queue<TreeNode*> que;
        if (root) que.push(root);
        TreeNode* lastNode;
        while (!que.empty()) {
            int size = que.size();
            lastNode = que.front();
            for (int i = 0; i < size; i++) {
				TreeNode* node = que.front();
				que.pop();
                if (node->left) que.push(node->left);
                if (node->right) que.push(node->right);
            }
        }
        return lastNode->val;
	}
};

class Solution112 {
public:
	bool hasPathSum(TreeNode* root, int targetSum) {
        //终止条件
        if (root == NULL) {
            return false;
        }
        if (root != NULL && root->left == NULL && root->right == NULL && root->val == targetSum) {
            return true;
        }
        if (root != NULL && root->left == NULL && root->right == NULL && root->val != targetSum) {
            return false;
        }
        //后序遍历
        if (hasPathSum(root->left, targetSum - root->val)) {
            return true;
        }
        if (hasPathSum(root->right, targetSum - root->val)) {
            return true;
        }
        //中间节点处理，好像没什么需要处理的，就是汇总一下两边的结果,但是前面都已经返回了
        return false;
	}
    bool hasPathSum1(TreeNode* root, int targetSum) {
        stack<pair<TreeNode*,int>> stk;
        if (root) stk.push(make_pair(root,targetSum));
        while (!stk.empty()) {
            //拿出个节点
            pair<TreeNode*, int> cur = stk.top();
            stk.pop();
            TreeNode* node = cur.first;
            int num = cur.second;
            if (node != NULL) {
                //中序遍历
                if (node->right) stk.push(make_pair(node->right,num-node->val));
                if (node->left) stk.push(make_pair(node->left, num - node->val));
                stk.push(cur);
                stk.push(make_pair((TreeNode*)NULL, 0));
            }
            else {
                //再拿出节点
				pair<TreeNode*, int> cur = stk.top();
				stk.pop();
				TreeNode* node = cur.first;
				int num = cur.second;
                //是叶子节点
                if (node->left == NULL && node->right == NULL && node->val == num) {
                    return true;
                }
            }
        }
        return false;
    }
};
class Solution113 {
public:
    vector<vector<int>> result;
    void traversal(TreeNode* root, int targetSum,vector<int>& path) {
        //结束条件
        if (root == NULL) {
            return;
        }        
        //先序遍历 
        if (root->left == NULL && root->right == NULL && root->val == targetSum) {
            result.push_back(path);
        }
        if (root->left == NULL && root->right == NULL && root->val != targetSum) {
            return;
        }
		if (root->left) {
			path.push_back(root->left->val);
			traversal(root->left, targetSum - root->val, path);
            path.pop_back();
		}
		if (root->right) {
			path.push_back(root->right->val);
			traversal(root->right, targetSum - root->val, path);
            path.pop_back();
		}


    }
	vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
        vector<int> path;
        if (root) path.push_back(root->val);
        traversal(root, targetSum, path);
        return result;
	}
};
class Solution654 {
public:
    //左闭右闭
    TreeNode* traversal(vector<int>& nums,int left,int right) {
        //结束条件 
        if (left > right) {
            return NULL;
        }
        if (left == right) {
            return new TreeNode(nums[left]);
        }
        //先序遍历
        //中  ，逻辑处理,找到最大值作为新的root节点返回
        TreeNode* root=new TreeNode;
        int maxVal=INT_MIN;
        int maxValIndex = 0;
        for (int i = left; i <= right; i++) {
            if (maxVal < nums[i]) {
                maxVal = nums[i];
                maxValIndex = i;
            }
        }
        root->val = maxVal;
        //左
        root->left=traversal(nums, left, maxValIndex - 1);
        //右
        root->right = traversal(nums, maxValIndex + 1, right);
        return root;
    }


	TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
        return traversal(nums, 0, nums.size() - 1);
	}
};

class Solution617 {
public:
	TreeNode* mergeTrees(TreeNode* root1, TreeNode* root2) {
        //终止条件
		if (root1==NULL)  return root2;
		if (root2==NULL)  return root1;
        //先序遍历,中       
        root1->val += root2->val;
        //左
        root1->left = mergeTrees(root1->left, root2->left);
        //右
        root1->right = mergeTrees(root1->right, root2->right);
        return root1;
	}
	
};

class Solution700 {
public:
	TreeNode* searchBST1(TreeNode* root, int val) {
        queue<TreeNode*> que;
        if (root) que.push(root);
        while (!que.empty()) {
            int size = que.size();
            for (int i = 0; i < size; i++) {
                TreeNode* node = que.front();
                que.pop();
                //处理本节点
                if (node->val == val) return node;
                if (node->left) que.push(node->left);
                if (node->right) que.push(node->right);
            }
        }
        return NULL;
	}
    TreeNode* searchBST2(TreeNode* root, int val) {
        if (root == NULL || root->val == val) return root;

        while (root != NULL) {
			if (root->val == val) return root;
			else if (root->val < val) root = root->right;
			else if (root->val > val) root = root->left;
        }
        return NULL;

    }

	TreeNode* searchBST(TreeNode* root, int val) {
		if (root == NULL) {
			return NULL;
		}
		//中
		if (root->val == val) {
			return root;
		}
		if (root->val < val) return searchBST(root->right, val);
		if (root->val > val) return searchBST(root->left, val);
		return NULL;
	}
};

class Solution98 {
public:
    TreeNode* preNode=NULL;

	bool isValidBST(TreeNode* root) {
		if (root == NULL) return true;
		//中序遍历
		//左
		bool left_check = true;
		bool right_check = true;
		if (root->left) {
			left_check = isValidBST(root->left);
		}
		//中
		if (preNode != NULL && root->val < preNode->val) return false;
        preNode = root;
		//右
		if (root->right) {
			right_check = isValidBST(root->right);
		}
		return left_check && right_check;

	}
    //迭代法
    bool isValidBST2(TreeNode* root) {
        stack<TreeNode*> stk;
        if (root) stk.push(root);
        while (!stk.empty()) {
            TreeNode* node = stk.top();
            stk.pop();
            if (node != NULL) {
                //右
                if (node->right) stk.push(node->right);
                //中
                stk.push(node);
                stk.push(NULL);
                //左
                if (node->left) stk.push(node->left);
            }
            else {
				TreeNode* node = stk.top();
				stk.pop();
                if (preNode != NULL && node->val <= preNode->val) {
                    return false;
                }
                preNode = node;
            }

        }
        return true;
    }
};

class Solution530 {
public:
    TreeNode* preNode = NULL;
    int minVal=INT_MAX;
	int getMinimumDifference(TreeNode* root) {
        //结束条件
        if (root == NULL) {
            return -1;
        }

        //中序遍历
        getMinimumDifference(root->left);
        if (preNode != NULL) {
            minVal = min(minVal, root->val - preNode->val);
        }
        preNode = root;
        getMinimumDifference(root->right);
        return minVal;
	}

	int getMinimumDifference1(TreeNode* root) {
        stack<TreeNode*> stk;
        if (root) stk.push(root);
        TreeNode* preNode = NULL;
        int minVal=INT_MAX;
        while (!stk.empty()) {
            TreeNode* node = stk.top();
            stk.pop();
            if (node != NULL) {
                //中序遍历
                //右
				if (node->right) {
					stk.push(node->right);
				}
                stk.push(node);
                stk.push(NULL);
                //左
				if (node->left) {
					stk.push(node->left);
				}
            }
            else {
				TreeNode* node = stk.top();
				stk.pop();
				//中,节点的逻辑处理
				if (preNode != NULL) {
					minVal = min(minVal, node->val - preNode->val);
				}
				preNode = node;
            }
        }
        return minVal;
	}
};

class Solution501 {
public:
    TreeNode* preNode = NULL;
    int maxFre = 1;//历史最大频率
    int curFre = 1;//当前节点的频率
    vector<int> result;
    void traversal(TreeNode* root) {
        //停止条件
        if (root == NULL) {
            return;
        }
        //中序遍历
        traversal(root->left);//左
        //中
		if (preNode != NULL) {
			//从第二个开始看
			if (preNode->val == root->val) {    //和前面的相同了，连杀继续
				curFre++;
			}
			else {
				curFre = 1;     //归一
			}
		}
		if (curFre == maxFre) { //前面兄弟的超神故事结束
			result.push_back(root->val);
		}
		else if (curFre > maxFre) {
			result.clear();
			result.push_back(root->val);
            maxFre = curFre;
		}
        preNode = root;
        traversal(root->right);

    }

	vector<int> findMode(TreeNode* root) {
        traversal(root);
        return result;

	}
	vector<int> findMode2(TreeNode* root) {
        stack<TreeNode*> stk;
        if (root) stk.push(root);
		TreeNode* preNode = NULL;
		int maxFre = 1;//历史最大频率
		int curFre = 1;//当前节点的频率
		vector<int> result;
        while (!stk.empty()) {
            TreeNode* node = stk.top();
            stk.pop();
            if (node != NULL) {
                //右
                if (node->right) stk.push(node->right);
                stk.push(node);
                stk.push(NULL);
                if (node->left) stk.push(node->left);
            }
            else {
				TreeNode* node = stk.top();
				stk.pop();
				if (preNode != NULL) {
					//从第二个开始看
					if (preNode->val == node->val) {    //和前面的相同了，连杀继续
						curFre++;
					}
					else {
						curFre = 1;     //归一
					}
				}
				if (curFre == maxFre) { //前面兄弟的超神故事结束
					result.push_back(node->val);
				}
				else if (curFre > maxFre) {
					result.clear();
					result.push_back(node->val);
					maxFre = curFre;
				}
				preNode = node;
            }

        }
        return result;

	}
};

class Solution236 {
public:
    //找到了返回找到的节点，没找到返回NULL
	TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (root == NULL || p == NULL || q == NULL) {
            return NULL;
        }
        TreeNode* left = lowestCommonAncestor(root->left, p, q);
        TreeNode* right = lowestCommonAncestor(root->right, p, q);
        //中间节点的判断逻辑，拿到左右两边的结果，进行逻辑判断
        //左右两边汇合处
        if (left != NULL && right != NULL) {
            return root;
        }
        if (left != NULL && right == NULL) {
            if (root == p || root == q) return root;
            else return left;
        }
        if (left == NULL && right != NULL) {
			if (root == p || root == q) return root;
			else return right;
        }
        if (root == p || root == q) return root;
        else return NULL;
	}
};

class Solution235 {
public:
	TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        //先序遍历
        //结束条件
        int minval = min(p->val, q->val);
        int maxval = max(p->val, q->val);
        if(root == NULL||(root->val>=minval&&root->val<=maxval)) return root;//中节点逻辑
        if (root->val > maxval) {
            TreeNode* left = lowestCommonAncestor(root->left, p, q);
            if (left != NULL) return left;
        }
		TreeNode* right = lowestCommonAncestor(root->right, p, q);
		return right;
	}

    TreeNode* lowestCommonAncestor2(TreeNode* root, TreeNode* p, TreeNode* q) {
        int minval = min(p->val, q->val);
        int maxval = max(p->val, q->val);
        while (root) {
            if (root->val > maxval) {
                root = root->left;
            }
            else if (root->val < minval) {
                root = root->right;
            }
            else {
                return root;
            }
        }
        return root;
    }
};
class Solution701 {
public:
    TreeNode* insertIntoBST(TreeNode* root, int val) {
        //后序遍历

    }
	TreeNode* insertIntoBST1(TreeNode* root, int val) {
        if (root == NULL) return new TreeNode(val);
        TreeNode* cur = root;
        while (cur)
        {
            if (val < cur->val) {
                if (cur->left)  cur = cur->left;
                else {
                    cur->left = new TreeNode(val);
                    break;
                }
            }
            else {
				if (cur->right)  cur = cur->right;
                else {
                    cur->right = new TreeNode(val);
                    break;
                }    
            }
        }
        return root;

	}
};

class Solution450 {
public:
    //处理被删除节点的后事
    TreeNode* handleDelNode(TreeNode* delNode) {
		//将它的左节点放到右子树适当的位置
        TreeNode* left = delNode->left;
        TreeNode* right = delNode->right;
        if (left == NULL) {
            return right;
        }
        if (right == NULL) {
            return left;
        }
        else {
            TreeNode* cur=right;
            while (cur->left!=NULL) {
                cur = cur->left;
            }
            cur->left = left;
        }
        TreeNode* ret = delNode->right;
        delete delNode;
        return ret;
    }

	TreeNode* deleteNode(TreeNode* root, int key) {
        TreeNode* cur = root;
        TreeNode* delNode = NULL;
        TreeNode* preNode=NULL;
        while (cur!=NULL) {
            if (key == cur->val) {
                delNode = cur;
                break;
            }
            else if (key < cur->val) {
                preNode = cur;
                cur = cur->left;
            }
            else {
                preNode = cur;
                cur = cur->right;
            }      
        }
        if (delNode == NULL) return root;
        if (preNode == NULL) return handleDelNode(root);
        else if (preNode->left == delNode) {
            preNode->left = handleDelNode(delNode);
        }
        else {
            preNode->right = handleDelNode(delNode);
        }

        return root;
	}
};

class Solution669 {
public:
    //思路：递归，只返回符合条件的节点
    //当前节点只有两种情况;符合条件被不符合条件
	TreeNode* trimBST(TreeNode* root, int low, int high) {
       //停止条件
        if(root==NULL){
            return root;
        }
        //中序遍历
        //当前节点不符合条件，那么递归找到符合条件的节点返回
        if (high < root->val) {
            return trimBST(root->left, low, high);//不用右边递归了，直接往左边找就ok了
        }
        if (root->val < low) {
            return trimBST(root->right, low, high);//不用左边递归了，直接往左边找就ok了
        }
        //当前节点符合条件，继续遍历，完事返回本节点
        root->left = trimBST(root->left, low, high);
        root->right = trimBST(root->right, low, high);
        return root;
	}
    TreeNode* trimBST1(TreeNode* root, int low, int high) {
        if (root == NULL) return root;
        //先到点上合适的位置去
		while (root != NULL && (root->val<low || root->val>high)) {
            if (root->val < low) {
                root = root->right;
            }
            else {
                root = root->left;
            }
        }
        if (root == NULL) return NULL;
        //将它的左右节点中不合适的全部删除掉
        TreeNode* cur = root;
        //处理左边节点,当左边不符合条件的时候，它的左分支也就全都不符合条件了，这样我们只需要它的右分支就可以了
        while (cur != NULL){
            while (cur->left != NULL && cur->left->val < low) {
                cur->left = cur->left->right;
            }     
            cur = cur->left;
        }
        //处理右分支
        cur = root;
        while (cur != NULL) {
            while (cur->right && cur->right->val > high) {
                cur->right = cur->right->left;
            }
            cur = cur->right;
        }
        return root;

    }
};
class Solution108 {
public:
	// 左闭右闭区间[left, right]
    TreeNode* traversal(vector<int>& nums, int left, int right)
    {
        //停止条件
        if (left > right) {
            return NULL;
        }
        //中序遍历
        TreeNode* node = new TreeNode();
        int mid = left + (right - left + 1) / 2;
        node->val = nums[mid];
        //左
        node->left = traversal(nums, left, mid - 1);
        node->right = traversal(nums, mid + 1, right);
        return node;
    }
	TreeNode* sortedArrayToBST(vector<int>& nums) {
        return traversal(nums, 0, nums.size() - 1);
	}
};

class Solution538 {
public:
    //遍历方式，右中左 
    TreeNode* preNode = NULL;
    void traversal(TreeNode* root) {
		//停止条件
		if (root == NULL) {
            return;
		}
		//右
		convertBST(root->right);
		//中
		if (preNode) {
			root->val += preNode->val;
		}
        preNode = root;
		//左
		convertBST(root->left);
    }

	TreeNode* convertBST(TreeNode* root) {
        traversal(root);
        return root;
	}
};

class Solution77_copy {
public:
    vector<vector<int>> result;
    vector<int> path; // 用来存放符合条件结果
    void backtracking(int n, int k, int startIndex) {
        if (path.size() == k) {
            result.push_back(path);
            return;
        }
        for (int i = startIndex; i <= n - (k - path.size()) + 1; i++) {
            path.push_back(i);//处理节点
            backtracking(n, k, i + 1);
            path.pop_back();

        }

    }

	vector<vector<int>> combine(int n, int k) {
        backtracking(n, k, 1);
        return result;
	}
};
class Solution216 {
public:
	vector<vector<int>> result; // 存放结果集 
	vector<int> path; // 符合条件的结果
    void backtracking(int targetSum, int k, int sum, int startIndex) {
		if (sum > targetSum) { // 剪枝操作
			return;
		}
        if (path.size() == k) {
            if(sum==targetSum) result.push_back(path);
            return;
        }
        for (int i = startIndex; i <= 9; i++) {
            path.push_back(i);
            backtracking(targetSum, k, sum + i, i + 1);
            path.pop_back();
        }

    }

	vector<vector<int>> combinationSum3(int k, int n) {
        backtracking(n, k, 0, 1);
        return result;
	}
};
class Solution17 {
public:
	const string letterMap[10] = {
	"", // 0
	"", // 1
	"abc", // 2
	"def", // 3
	"ghi", // 4
	"jkl", // 5
	"mno", // 6
	"pqrs", // 7
	"tuv", // 8
	"wxyz", // 9
	};
	vector<string> result;
	string s;
    void backtracing(string& digits,int index) {
		//结束条件
		if (s.size() == digits.size()) {
			result.push_back(s);
            return;
		}
		//遍历
        //思考清楚这棵树的形状，每个
		int num = digits[index] - '0';
		for (int j = 0; j < letterMap[num].size(); j++) {
			s += letterMap[num][j];
			backtracing(digits, index + 1);
			s.pop_back();
		}

    }

	vector<string> letterCombinations(string digits) {
        if (digits == "") {
            return {};
        }
        backtracing(digits, 0);
        return result;
	}
};
class Solution39 {
public:
    vector<vector<int>> result;
    vector<int> path;
    void backtracing(vector<int>& candidates, int target,int startIndex) {
        if (target < 0) {
            return;
        }
        if (target == 0) {
            result.push_back(path);
            return;
        }
        for (int i = startIndex; i < candidates.size()&& target - candidates[i]>=0; i++) {
            path.push_back(candidates[i]);
            backtracing(candidates, target-candidates[i], i);
            path.pop_back();
        }

    }

	vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        sort(candidates.begin(), candidates.end()); // 需要排序
        backtracing(candidates, target, 0);
        return result;
	}
};

class Solution40 {
public:
    vector<vector<int>> result;
    vector<int> path;
    void backtracing(vector<int>& candidates, int target,int startIndex) {
        //停止条件
        if (target < 0) {
            return;
        }
        if (target == 0) {
            result.push_back(path);
            return;
        }
        for (int i = startIndex; i < candidates.size(); i++) {
            if (i > startIndex && candidates[i] == candidates[i - 1]) {
                continue;
            }
            path.push_back(candidates[i]);
            backtracing(candidates, target - candidates[i], i + 1);
            path.pop_back();

        }
    }

	vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        sort(candidates.begin(), candidates.end()); // 需要排序
        backtracing(candidates, target, 0);
        return result;
	}
};

class Solution131 {
public:
	vector<vector<string>> result;
	vector<string> path; // 放已经放过挡板的路径
    bool isPalindrome(string s,int left,int right) {
        while (left < right) {
            if (s[left] == s[right]) {
                left++;
                right--;
            }
            else {
                return false;
            }
        }
        return true;
    }
    //分割问题
    void backtracing(string& s,int startIndex) {
        //停止条件
        if (startIndex>=s.size()) {
            result.push_back(path);
            return;
        }
        for (int i = startIndex; i < s.size(); i++) {
            if (isPalindrome(s, startIndex, i)) {
                string str = s.substr(startIndex, i - startIndex + 1);
                path.push_back(str);
            }
            else {
                continue;
            }
            backtracing(s, i + 1);
            path.pop_back();
        }

    }


	vector<vector<string>> partition(string s) {
        backtracing(s, 0);
        return result;
	}
};
class Solution93 {
public:
    //分割问题
    vector<string> result;
    vector<string> path;
    void backtracing(string& s,int startIndex) {
        //终止条件
        if (startIndex >= s.size()) {
            movePath2Result();
            return;
        }
        //单层逻辑
            for (int i = startIndex; i < s.size() && path.size() < 4; i++) {
            string str = s.substr(startIndex, i - startIndex + 1);
            if ( isValid(str)) {
                path.push_back(str);
				backtracing(s, i + 1);
				path.pop_back();
            }
            else {
               break;
            }

        }

    }
    bool isValid(string& s) {
        if (s.size() > 3) return false;
        if (s.size() > 1 && s[0] == '0') return false;
        int num = 0;
        for (char c : s) {
            if (c > '9' || c < '0') {
                return false;
            }
            num = num * 10 + (c - '0');
        }
        if (num > 255) {
            return false;
        }
        return true;

    }
    void movePath2Result() {
        string str;
        for (int i = 0; i < path.size()-1; i++) {
            str += path[i]+".";
        }
        str += path.back();
        result.push_back(str);
    }


	vector<string> restoreIpAddresses(string s) {
        backtracing(s, 0);
        return result;
	}
};
class Solution78 {
public:
	vector<vector<int>> result;
	vector<int> path;
    void backtracking(vector<int>& nums, int startIndex) {
        if (startIndex > nums.size()) {
            return;
        }
        for (int i = startIndex; i < nums.size(); i++) {
            // 处理本节点
            path.push_back(nums[i]);//本节点的值 
            result.push_back(path);//结果摘取本节点
            backtracking(nums, i + 1);//到下一层
            path.pop_back();//回溯回本层
        }

    }

	vector<vector<int>> subsets(vector<int>& nums) {
        result.push_back({});
        backtracking(nums, 0);
        return result;

	}
};

class Solution90_copy {
public:
	vector<vector<int>> result;
	vector<int> path;
    void backtracing(vector<int>& nums, int startIndex, vector<bool>& used) {
        //结束条件
        if (startIndex > nums.size()) {
            return;
        }
        for (int i = startIndex; i < nums.size(); i++) {
            //单层逻辑
            //判断是否重复
            if (i > 0 && nums[i] == nums[i - 1]&&used[i-1]==false) {               
                continue;
            }
            path.push_back(nums[i]);
            result.push_back(path);
			used[i] = true;//回溯前弄成true，给下一层看
			backtracing(nums, i + 1, used);
			used[i] = false;//回溯完弄成false，给本层看
            path.pop_back();
        }

    }

	vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        result.push_back({});
        vector<bool> used(nums.size());
        backtracing(nums, 0, used);
        return result;
	}
};

class Solution491 {
public:
	vector<vector<int>> result;
	vector<int> path;
    void backtracking(vector<int>& nums, int startIndex)
    {
        //结束条件
        if (startIndex >= nums.size()) {
            return;
        }
        //本层逻辑
        unordered_set<int> uset; // 使用set来对本层元素进行去重
        for (int i = startIndex; i < nums.size(); i++) {
            //本节点逻辑
            if ((!path.empty() && path.back() > nums[i])
                || uset.find(nums[i]) != uset.end())
            {
                continue;
            }
            uset.insert(nums[i]);
            path.push_back(nums[i]);
            if (path.size() >= 2)result.push_back(path);
            backtracking(nums, i + 1);
            path.pop_back();
        }

    }
	vector<vector<int>> findSubsequences(vector<int>& nums) {
        backtracking(nums, 0);
        return result;
	}
};
class Solution46 {
public:
	vector<vector<int>> result;
	vector<int> path;
    void  backtracing(vector<int>& nums,vector<bool>& used) {
        //结束条件
        if (path.size() == nums.size()) {
            result.push_back(path);
            return;
        }
        //每一层的逻辑
        for (int i = 0; i < nums.size(); i++) {
            if(used[i]==true)
                continue;
            path.push_back(nums[i]);
            used[i] = true;
            backtracing(nums, used);
            path.pop_back();
            used[i] = false;

        }

    }

	vector<vector<int>> permute(vector<int>& nums) {
        vector<bool> used(nums.size(), false);
        backtracing(nums, used);
        return result;
	}
};
class Solution47 {
public:
	vector<vector<int>> result;
	vector<int> path;
    void backtracing(vector<int> nums,vector<bool>& used) {
        if (path.size() == nums.size()) {
            result.push_back(path);
            return;
        }
        for (int i = 0; i < nums.size(); i++) {
            if (used[i] == true ||
                (i > 0 && nums[i] == nums[i - 1] && used[i - 1] == false) //同一层这个节点已经把后面的情况算过了，去重
                )
            {
                continue;
            }
            path.push_back(nums[i]);
            used[i] = true;
            backtracing(nums,used);
            path.pop_back();
            used[i] = false;
        }

    }

	vector<vector<int>> permuteUnique(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        vector<bool> used(nums.size(), false);
        backtracing(nums, used);
        return result;
	}
};

class Solution332 {
public:
    unordered_map<string, map<string, int>> targets;

    bool backtracing(int ticketsNum, vector<string>& result) {
        //结束条件
        if (result.size() == ticketsNum + 1) {
            return true;
        }
        //单层逻辑
        for (auto& city : targets[result.back()]) {  //单层树
            //单个节点
            if (city.second > 0) {
                result.push_back(city.first);
                city.second--;
                if (backtracing(ticketsNum, result)) return true;
                city.second++;
                result.pop_back();
            }
        }
        return false;
    }

	vector<string> findItinerary(vector<vector<string>>& tickets) {
        vector<string> result;
        for (vector<string> vec : tickets) {
            targets[vec[0]][vec[1]]++;
        }
        result.push_back("JFK");
        backtracing(tickets.size(), result);
        return result;

	}
};
class Solution51 {
public:
    vector<vector<string>> result;

    bool isValid(int row,int col, const vector<string>& chessBoard) {
        //不能在同一列
        for (int i = 0; i <= row - 1; i++) {
            if (chessBoard[i][col] == 'Q')
                return false;
        }
        //不能是斜着的  ,参考了博主的方法，真机智啊
        //45度检查
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
            if (chessBoard[i][j] == 'Q') {
                return false;
            }
        }
        //135度检查
        for (int i = row - 1, j = col + 1; i >= 0 && j >= 0; i--, j++) {
			if (chessBoard[i][j] == 'Q') {
				return false;
			}
		}
        return true;     

    }

    void backtracking(int row, vector<string>& chessBoard) {
        //结束条件 
        if (row == chessBoard.size()) {
            result.push_back(chessBoard);
            return;
        }
        //单层遍历
        for (int i = 0; i < chessBoard.size(); i++) {
            //单个节点的逻辑
            if (isValid(row, i,chessBoard)) {
                chessBoard[row][i] = 'Q';
                backtracking(row + 1,chessBoard);
                chessBoard[row][i] = '.';
            }
        }

    }

	vector<vector<string>> solveNQueens(int n) {
        vector<string> chessBoard(n, string(n, '.'));
        backtracking(0, chessBoard);
        return result;
	}
};

class Solution37 {
public:
    //认为是81个点的递归？
    bool isValid(int row, int col,char val,const vector<vector<char>>& board) {
        //每一行
        for (int i = 0; i < 9; i++) {
           if (board[row][i] == val) {
                return false;
           }
        }
        //每一列
        for (int i = 0; i < 9; i++) {
            if (board[i][col] == val) {
                return false;
            }
        }
        //每个小方格
        int bottom = row / 3 * 3;
        int left = col / 3 * 3;
        for (int i = bottom; i < bottom + 3; i++)
        {
            for (int j = left; j < left + 3; j++) 
            {
                if (board[i][j] == '.' || (i == row && j == col)) {
                    continue;
                }
                else {
                    if (val == board[i][j]) {
                        return false;
                    }
                }                
            }
        }
        return true;
    }

    
    bool backtracking(vector<vector<char>>& board, int sumNum) {
        //终止条件
        if (sumNum == 81) {
            return true;
        }

        int row = sumNum / 9;
        int col = sumNum % 9;
        if (board[row][col] != '.') {
            sumNum++;
            return backtracking(board, sumNum);
        }
        else {        
            //单层逻辑,看这个点放0-9合不合适
			for (char i = '0'; i <= '9'; i++) {
				if (isValid(row, col, i,board)) {
                    board[row][col] = i;
                    sumNum++;
                    if (backtracking(board, sumNum)) return true;
                    sumNum--;
                    board[row][col] = '.';
				}
			}
            return false;
        }
    }
    
	void solveSudoku(vector<vector<char>>& board) {
        backtracking(board,0);
	}
	void printVector2(vector<vector<char>> dp)
	{
        system("clear");
		for (auto it = dp.begin(); it < dp.end(); it++)
		{
			for (auto vit = it->begin(); vit < it->end(); vit++)
			{
				cout << (*vit) << "\t";
			}
			cout << endl;
		}
		cout << "------------------------------------------------" << endl;
	}
};

class Solution455 {
public:
	int findContentChildren(vector<int>& children, vector<int>& biscuit) {
        sort(children.begin(), children.end());
        sort(biscuit.begin(), biscuit.end());
        int bisIndex = biscuit.size() - 1;
        int num = 0;
        for (int i = children.size() - 1; i >= 0; i--) { //小孩从大到小，最大值满足不了那就拉倒了，下一个
            if (bisIndex >= 0 && biscuit[bisIndex] >= children[i]) {//饼干能满足这个小孩
                num++;
                bisIndex--;
            }
        }
        return num;
	}
};

class Solution376 {
public:
	int wiggleMaxLength(vector<int>& nums) {
        if (nums.size() < 2) {
            return nums.size();
        }
        int curdiff = 0;
        int prediff = 0;
        int result = 1;
        for (int i = 1; i < nums.size(); i++) {
            curdiff = nums[i] - nums[i - 1];
            if ((curdiff > 0 && prediff <= 0) || (curdiff < 0 && prediff >= 0)) {
                result++;
                prediff = curdiff;
            }
        }
        return result;
	}
};

class Solution53 {
public:
	int maxSubArray(vector<int>& nums) {
		int sum = 0;
		int result = INT_MIN;
		for (int num : nums) {
			sum += num;
			result = max(result, sum);
			if (sum < 0) {
				sum = 0;
			}
		}
		return result;
	}
};
class Solution55 {
public:
	bool canJump(vector<int>& nums) {
		if (nums.size() <= 1) return true;
		int maxlen = 0;
		for (int i = 0; i < nums.size() - 1; i++) {
			if (i > maxlen) {
				return false;
			}
			maxlen = max(maxlen, i + nums[i]);
			if (maxlen >= nums.size() - 1) {
				return true;
			}

		}
		return false;
	}
};
class Solution45 {
public:
    //想象一下跑跑卡丁车的过程，跑到一定程度告诉你下一个距离在哪
	int jump(vector<int>& nums) {
		if (nums.size() == 1) return 0;
		int curDistance = 0;
		int nextDistance = 0;
		int curIndex = 0;
		int result = 1;
		for (int i = 0; i < nums.size() - 1; i++) {
            //更新棋子位置
			nextDistance = max(i + nums[i], nextDistance);
			if (nextDistance >= nums.size() - 1) {
				break;
			}
			if (curDistance == i) {
				result++;
				curDistance = nextDistance;
			}
		}
		return result;
	}
};

class Solution1005 {
public:
    static bool cmp(int a, int b) {
        return abs(a) > abs(b);
    }

	int largestSumAfterKNegations(vector<int>& A, int K) {
        sort(A.begin(), A.end(), cmp);
        for (int i = 0; i < A.size(); i++) {
            if (A[i] < 0 && K>0) {
                A[i] = -A[i];
                K--;
           }
        }
		if (K%2) {
            A[A.size() - 1] = -A[A.size() - 1];
		}
		int sum = 0;
		for (int i = 0; i < A.size(); i++) {
			sum += A[i];
		}

        return sum;
	}
};
class Solution134 {
public:
	int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        int start = 0;
        int n = cost.size();
        int curSum = 0;
        for (int i = 0; i < n; i++) {
            curSum += gas[i] - cost[i];

            if (curSum < 0) {
                start = (start + n - 1) % n;
                curSum += gas[start] - cost[start];
                while (curSum < 0 && start != i) {
					start = (start + n - 1) % n;
                    curSum += gas[start] - cost[start];
                }
				if (start == i) {
					return curSum >= 0 ? start : -1;
				}
            }
        }
        return start;
	}
};

class Solution135 {
public:
	int candy(vector<int>& ratings) {
        int n = ratings.size();
        vector<int> result(n, 1);
        for (int i = 1; i < n; i++) {
            if (ratings[i] > ratings[i - 1] && result[i] <= result[i - 1]) {
                result[i] = result[i - 1] + 1;
            }
        }
        for (int i = n - 2; i >= 0; i--) {
            if (ratings[i] > ratings[i + 1] && result[i] <= result[i + 1]) {
                result[i] = result[i + 1] + 1;
            }
        }
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += result[i];
        }

        return sum;

	}
};

class Solution860 {
public:
	bool lemonadeChange(vector<int>& bills) {
        unordered_map<int, int> money;
		for (int cur : bills) {
            if (cur == 10) {
                if (money[5] > 0) money[5]--;
                else return false;
            }
            else if (cur == 20) {
                if (money[10] > 0 && money[5] > 0) {
                    money[10]--;
                    money[5]--;
                }
                else if (money[5] >= 3) {
                    money[5] -= 3;
                }
                else return false;
            }
            money[cur]++;
        }
        return true;
	}
};

class Solution425 {
public:
	int findMinArrowShots(vector<vector<int>>& points) {
        if (points.empty()) return 0;
        sort(points.begin(), points.end(), [](vector<int> a, vector<int> b) {
            return a[0] < b[0];
            });
        int m_end = points[0][1];
        int result = 1;
        for (int i = 1; i < points.size(); i++) {
            if (points[i][0] > m_end) {
                result++;
                m_end = points[i][1];
            }
            m_end = min(m_end, points[i][1]);
        }
        return result;
	}
};
class Solution435 {
public:
	int eraseOverlapIntervals(vector<vector<int>>& intervals) {
        if (intervals.empty()) return 0;
        sort(intervals.begin(), intervals.end(), [](vector<int>& a, vector<int>& b) {
            return a[0] < b[0];
            }
        );
        int m_end = intervals[0][1];
        int result = 0;
        for (int i = 1; i < intervals.size(); i++) {
            if (intervals[i][0] >= m_end) {
                m_end = intervals[i][1];
                continue;
            }
            else {
                result++;
                m_end = min(m_end, intervals[i][1]);
            }      
        }
        return result;
	}
};

class Solution763 {
public:
	vector<int> partitionLabels1(string letters) {
        if (letters.empty()) return {};
        unordered_map<char, pair<int, int>> map;
        for (int i = 0; i < letters.size(); i++) {
            if (map.find(letters[i]) == map.end()) {
                map[letters[i]].first = i;
            }
            map[letters[i]].second = i;
        }
        vector<pair<int, int>> vec;
        for (auto letter : map) {
            vec.push_back(letter.second);
        }
        sort(vec.begin(), vec.end(), [](pair<int, int>& a, pair<int, int>& b) {
            return a.first < b.first;
            }
        );
        int start = 0;
        int end = vec[0].second;
        vector<int> result;
        for (int i = 1; i < vec.size(); i++) {
            if (vec[i].first > end) {
                result.push_back(end - start + 1);
                start = vec[i].first;
            }
            end = max(end, vec[i].second);
        }
        result.push_back(end - start + 1);
        return result;
	}
    vector<int> partitionLabels(string letters) {
        unordered_map<char, int> boarder;//记录最远边界
        for (int i = 0; i < letters.size(); i++) {
            boarder[letters[i]] = i;
        }
        int start = 0;
        int end = boarder[letters[0]];
        vector<int> result;
        for (int i = 0; i < letters.size(); i++) {
            end = max(end, boarder[letters[i]]);
            if (end == i) {
                result.push_back(end - start + 1);
                start = i + 1;
            }
        }
        return result;
    }

};

class Solution56 {
public:
	vector<vector<int>> merge(vector<vector<int>>& intervals) {
        if (intervals.empty()) return {};
        sort(intervals.begin(), intervals.end(), [](vector<int>& a, vector<int>& b){
            return a[0] < b[0];
        }
        );
        int start = intervals[0][0];
        int end = intervals[0][1];
        vector<vector<int>> result;
        for (int i = 1; i < intervals.size(); i++) {
            if (intervals[i][0] > end) {
                result.push_back({ start,end });
                start = intervals[i][0];               
            }
            end = max(end, intervals[i][1]);        
            
        }
		result.push_back({ start,end });
        return result;
	}
};

class Solution738 {
public:
	int monotoneIncreasingDigits1(int N) {
        vector<int> num;
        string str = to_string(N);
        if (str.size() <= 1) return N;
        int spiltIndex = -1;
        for (int i = 1; i < str.size(); i++) {
            if (str[i - 1] > str[i]) {
                spiltIndex = i;
                break;
            }
        }
        if (spiltIndex == -1) return N;
        else {
            // i以后都变为9，i以前第一个为i-1的减去1
            spiltIndex--;
            char temp = str[spiltIndex];
            while (spiltIndex >= 0&&str[spiltIndex]==temp) {                
                spiltIndex--;
            }
            spiltIndex++;
            str[spiltIndex]--;
            for (int i = spiltIndex + 1; i < str.size(); i++) {
                str[i] = '9';
            }
        }
        return stoi(str);

	}
    int monotoneIncreasingDigits(int N) {
		vector<int> num;
		string str = to_string(N);
		if (str.size() <= 1) return N;
        int spiltIndex = str.size() - 1;
        for (int i = str.size() - 2; i >= 0; i--) {
            if (str[i] > str[i + 1]) {
                spiltIndex = i;
                str[i]--;
            }
        }

        for (int i = spiltIndex+1; i < str.size(); i++) {
            str[i] = '9';

        }
        return stoi(str);
    }
};
//----------------------动态规划--------------------------------

class Solution509 {
public:
	int fib(int n) {
        if (n <= 1) return n;
        vector<int> dp(n + 1);//dp含义：斐波那契数列的值
        dp[0] = 0;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
	}
};

class Solution70 {
public:
	int climbStairs(int n) {
        if (n <= 1) return n;
        vector<int> dp(n + 1);//dp代表爬到第i层的方法数
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        printVector(dp);
        return dp[n];
	}
};

class Solution746 {
public:
	int minCostClimbingStairs(vector<int>& cost) {
		int n = cost.size();
		if (n <= 2) return 0;
        vector<int> dp(n + 1);
        for (int i = 2; i <= n; i++) {
            dp[i] = min(dp[i - 2] + cost[i - 2], dp[i - 1] + cost[i - 1]);//从这里走出来需要的能量值
        }
        printVector(dp);
        return dp[n];
	}
};
class Solution62 {
public:
	int uniquePaths1(int m, int n) {
        vector<vector<int>> dp(m, vector<int>(n));
        for (int i = 0; i < m; i++) {
            dp[i][0] = 1;
        }
        for (int j = 0; j < n; j++) {
            dp[0][j] = 1;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        printVector2(dp);
        return dp[m - 1][n - 1];

	}
    /*
    int uniquePaths(int m, int n) {
        long long numerator = m + n - 2;//分子
        long long denominator = m - 1;//分母
        int result1 = 1;
        int result2 = 1;
        int result = 1;
        for (int i = m - 1; i > 0; i--) {
            result1 *= (numerator--);
            result2 *= denominator--;
            if (result2!=0&&result1 % result2 == 0) {
                result *= result1 / result2;
                result1 = 1;
                result2 = 1;
            }

        }
        return re;
    }
    */
};
class Solution343 {
public:
    int integerBreak(int n) {
        //特殊条件处理

        //表示dp
        vector<int> dp(n + 1);//dp表示第n个数的最大乘积
        //初始化dp
        dp[1] = 1;
        dp[2] = 1;
        //寻找递推关系式
        //dp[i]=max
        for (int i = 3; i <= n; i++) {
            for (int j = 2; j < i; j++) {
                //i=  j+  i-j
                dp[i] = max(dp[i],
                    max((i - j) * j,
                        max(dp[i - j] * j,
                            max((i-j)*dp[j],dp[i-j]*dp[j]))));
            }       
        }
        //验证
        printVector(dp);
        return dp[n];

    }
};

//背包问题
class bag_problem {
public:
    void test_2_wei_bag_problem1() {
        vector<int> weight = { 1, 3, 4 };
        vector<int> value = { 15, 20, 30 };
        int bagWeight = 4;
        vector<vector<int>> dp(weight.size(), vector<int>(bagWeight + 1, 0));
        //初始化
        for (int j = bagWeight; j >= weight[0]; j--) {
            dp[0][j] = dp[0][j - weight[0]] + value[0];
        }
        for (int i = 1; i < weight.size(); i++) { //选几个包
            for (int j = 0; j < bagWeight + 1; j++) {
                if (j < weight[i]) {
                    dp[i][j] = dp[i][j];
                }
                else {
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i]);
                }
            }
        }
        cout << dp[weight.size() - 1][bagWeight] << endl;
    }
	void test_2_wei_bag_problem2() {
		vector<int> weight = { 1, 3, 4 };
		vector<int> value = { 15, 20, 30 };
		int bagWeight = 4;
        vector<int> dp(bagWeight + 1, 0);
		//初始化
		for (int i = 0; i < weight.size(); i++) { //选几个包
            for (int j = bagWeight; j >= weight[i]; j--) {
                dp[j] = max(dp[j], dp[j - weight[i]] + value[i]);
			}
		}
		cout << dp[bagWeight] << endl;
	}
};
class Solution416 {
public:
    bool backtracking(vector<int>& nums,vector<bool>& isused, int sum, int target, int start ) {
        //返回条件
        if (sum > target) {
            return false;
        }
        else if(sum==target) {
            return true;
        }
        for (int i = start; i < nums.size(); i++) {
            if (i > 0 && nums[i] == nums[i - 1] && isused[i - 1] == false) {
                continue;
            }
            isused[i] = true;
            if (backtracking(nums, isused, sum + nums[i], target, i + 1)) return true;
            isused[i] = false;
        }
        return false;
    }

    //重复逻辑：装了1，2，5和装 2 5的情况是交叉的，装1，2，5的情况其实就是装（2+5）的条件下+1
	bool canPartition1(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        vector<bool> isused(nums.size(), false);
        int target = 0;
        for (int i = 0; i < nums.size(); i++) {
            target += nums[i];
        }
        if (target % 2 == 1) return false;
        else target /= 2;
        return backtracking(nums, isused, 0, target, 0);
	}

    bool canPartition(vector<int>& nums) {
        if (nums.empty()) return false;
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        if (sum % 2 == 1) return false;
        else sum /= 2;
        vector<int> dp(sum + 1);//定义dp[i][j]为装入前n个数的最大值
        for (int i = 0; i < nums.size(); i++) {
            for (int j = sum; j >= nums[i]; j--) {
                dp[j] = max(dp[j], dp[j - nums[i]] + nums[i]);
            }
        }

        //printVector2(dp);
		if (dp[sum] == sum) {
            return true;
        }
        else {
            return false;
        }
    }

};

/*
class Solution698 {
public:
    vector<vector<int>> group;

    bool calculate(vector<int>& nums,int target) {
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        if (sum == target) {
            return true;
        }
        else {
            return false;
        }
    }
    //分组
    void backtracking() {

    }


	bool canPartitionKSubsets(vector<int>& nums, int k) {
        int sum = 0;
		for (int num : nums) {
			sum += num;
		}
        if (sum % k != 0) {
            return false;
        }
        else {
            sum /= k;
        }
        //分组 Cn k种方法
		vector<vector<int>> group(k);


	}
};
*/
class Solution1049 {
public:
	int lastStoneWeightII(vector<int>& stones) {
        if (stones.size() == 1) return stones[0];
        int sum = 0;
        for (int s : stones) {
            sum += s;
        }
        int package = sum / 2;//背包的大小
        vector<int> dp(package + 1);
        for (int i = 0; i < stones.size(); i++) {
            for (int j = package; j >= stones[i]; j--) {
                dp[j] = max(dp[j], dp[j - stones[i]]+stones[i]);
            }
            printVector(dp);
            cout << endl;
        }
        
        return (sum - dp.back())-dp.back();
	}
};

class Solution494 {
public:
    //加法x 减法sum-x x-(sum-x)=S  x=(S+sum)/2
	int findTargetSumWays1(vector<int>& nums, int S) {
        int sum = 0;
        int target = 0;
        for (int num : nums) {
            sum += num;
        }
        if ((S + sum) % 2 == 1) {
            return 0;
        }
        else {
            target = (S + sum) / 2;
        }
        //存在相同子结构,本问题的解和之前的问题有关系,变化的维度：包的大小，可选的数的数量
        vector<vector<int>> dp(nums.size(),vector<int>(target+1));//dp[i][j]含义：用0-i的数填满容量为j的包包，一共有几种填充方案
        dp[0][0] = 1;
        dp[0][nums[0]] = 1;
        for (int i = 1; i < nums.size(); i++) {
            for (int j = 0; j <= target; j++) {
                if (j < nums[i]) dp[i][j] = dp[i - 1][j];
                else dp[i][j] = dp[i - 1][j] + dp[i - 1][j - nums[i]];
            }
        }
        printVector2(dp);
        return dp[nums.size() - 1].back();

	}
    int findTargetSumWays(vector<int>& nums, int S) {
		int sum = 0;
		int target = 0;
		for (int num : nums) {
			sum += num;
		}
        if (S > sum) return 0;
		if ((S + sum) % 2 == 1) {
			return 0;
		}
		else {
			target = (S + sum) / 2;
		}
        vector<int> dp(target + 1);
        dp[0] = 1;
        for (int i = 0; i < nums.size(); i++) {
            for (int j = target; j >= nums[i]; j--) {
                dp[j] = dp[j] + dp[j - nums[i]];
            }
            printVector(dp);
            cout << endl;
        }
        return dp.back();
        
    }
};
class Solution474 {
public:
	int findMaxForm(vector<string>& strs, int m, int n) {
        vector<vector<int>> map(strs.size(), vector<int>(2,0));
        for (int i = 0; i < strs.size();i++) {
            for (char c : strs[i]) {
                if (c == '0') map[i][0]++;
                else map[i][1]++;
            }
        }
		printVector2(map);

        vector<vector<int>> dp(m + 1, vector<int>(n + 1));//m为0，n为1
        //二维背包
        for (int i = 0; i < strs.size(); i++) { //物品i
            for (int j = m; j >= map[i][0]; j--) {
                for (int k = n; k >= map[i][1]; k--) {
                    dp[j][k] = max(dp[j][k], dp[j - map[i][0]][k - map[i][1]] + 1);
                }
            }
		}
		printVector2(dp);
        return dp[m][n];
	}
};
class Solution518 {
public:
	int change(int amount, vector<int>& coins) {
        vector<int> dp(amount + 1);
        dp[0] = 1;
        
        for (int i = 0; i < coins.size(); i++) {
            for (int j = coins[i]; j <= amount; j++) {
                dp[j] = dp[j] + dp[j - coins[i]];
            }
            printVector(dp);
            cout << endl;
        }   
        cout << "------------------------------------------" << endl;
		for (int j = 0; j <= amount; j++) { // 遍历背包容量
			for (int i = 0; i < coins.size(); i++) { // 遍历物品
				if (j - coins[i] >= 0) {
					dp[j] += dp[j - coins[i]];
                }
			}
            printVector(dp);
            cout << endl;
		}
        return dp.back();
	}
};