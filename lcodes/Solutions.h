#include <vector>
#include <unordered_map>
#include <iostream>
#include <string>
#include<algorithm>

using namespace std;
void printVector2(vector<vector<int>>& v) {	
	int m = v.size();
	int n = v[0].size();
	for (auto it = v.begin(); it != v.end();it++) {
		for (auto iit = it->begin(); iit != it->end();iit++) {
			cout << *iit << "\t";
		}
		cout << endl;
	}
}

  struct ListNode {
      int val;
      ListNode *next;
      ListNode() : val(0), next(nullptr) {}
      ListNode(int x) : val(x), next(nullptr) {}
      ListNode(int x, ListNode *next) : val(x), next(next) {}
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
class Solution90 {
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
class Solution93
{
public:
	vector<string> result;
	string s;
	//点 点，从头到尾依次找三个个点
	void dfs(int start, int IPval, int IPindex)
	{
		for (int i = start; i < s.size(); i++) {			
			char c = s[i];
			IPval=IPval*10+s[i]-'0';
			

		}
	}

	vector<string> restoreIpAddresses(string s)
	{
		this->s = s;
		string cur;
		return result;
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
class Solution498 {
public:
	vector<int> findDiagonalOrder(vector<vector<int>>& matrix) {
		int m, n;

		if ((m = matrix.size())==0 ||( n = matrix[0].size())==0) {
			return {};
		}
		vector<int> ret(m * n);
		enum {
			UP = 0,
			DOWN
		};
		int dir=UP;//true为往上，false为往下
		int x = 0, y = 0;
		for (int i = 0; i < m * n; i++) {
			ret[i] = matrix[x][y];
			if (dir == UP) {
				//只会撞到上面和右边
				if (x==0) {
					//撞了上面					
					x -= 1;
				}

			}
		}


	}
};