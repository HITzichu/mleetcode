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
 


//88. �ϲ�������������
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
//90. �Ӽ� II
/*
˼·��
	�������
	ȥ��ͬһ�����ظ������
	����ÿһ�����������������,
	����ȥ�صĻ���ͬһ�㣬ȥ�ص�ԭ��Ӧ���ǣ��Һ�ǰ����ظ��˵Ļ���Ҫȥ���������ڲ�ͬ��Ĳ�Ӧ��ȥ�أ�����
*/
class Solution90 {
public:
	vector<int> nums;
	int n;
	vector<vector<int>> result;
	//ÿһ���������ǿ����ܲ�������һ��Ŷ���
	void dfs(int depth,int start,vector<int> cur) {
		if (depth == n + 1) {
			//���һ��
			return;
		}
		for (int i = start; i < n; i++) {
			if (i > start && nums[i] == nums[i - 1]) {
				continue;
			}
			//����Ҫ��
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
			//��ǰ���ֿ��Ժ�ǰ��������Ч����
			if (pre != 0 && num <= 26) {
				//	1.��ǰ������0			dp[i] =dp[i - 2]
				if (s1[i] == '0') {
					dp[i] = dp[i - 2];
				}
				//	2.��ǰ���ֲ���0			dp[i] = dp[i - 1] + dp[i - 2];
				else
				{
					dp[i] = dp[i - 1] + dp[i - 2];
				}
			}
			else
			{
				//��ǰ���ֲ��ܺ�ǰ�����������Ч����
				//	1.��ǰ������0			return 0
				if (s1[i] == '0') {
					return 0;
				}
				//	2.��ǰ���ֲ���0			dp[i]=dp[i-1]
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
	˼·����ͷ��β��ʣ�µĵ���

	*/
public:
	
	ListNode* reverseBetween(ListNode* head, int m, int n) {
		ListNode* preHead=new ListNode(0, head);
		ListNode* preLeft = preHead;
		for (int i = 0; i < m - 1; i++) {
			preLeft = preLeft->next;
		}
		ListNode* leftOrigin = preLeft->next;
		//��ʼ����
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
	//�� �㣬��ͷ��β��������������
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
//����ר��
class Solution238 {
public:
	vector<int> productExceptSelf(vector<int>& nums) {
		int n = nums.size();
		vector<int> left(n, 1);//��ʾ��i����ǰ��i-1�����ĳ˻���left[0]=1��left[1]=nums[0],left[2]=left[1]*nums[1]
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
		return comp1 > comp2;//a+b>b+a ��ô����Ϊa>bΪtrue
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
		int dir=UP;//trueΪ���ϣ�falseΪ����
		int x = 0, y = 0;
		for (int i = 0; i < m * n; i++) {
			ret[i] = matrix[x][y];
			if (dir == UP) {
				//ֻ��ײ��������ұ�
				if (x==0) {
					//ײ������					
					x -= 1;
				}

			}
		}


	}
};