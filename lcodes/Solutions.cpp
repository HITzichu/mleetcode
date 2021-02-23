#include "Solutions.h"

//4.Ѱ�����������������λ��
int getKthElement(const vector<int>& nums1, const vector<int>& nums2, int k)
{
	//kΪ��Ҫ�Ƚϵĸ���
	int m = nums1.size();
	int n = nums2.size();
	int index1 = 0, index2 = 0;
	while (true)
	{
		if (index1 == m)
		{
			return nums2[index2 + k - 1];		//Ѱ�ҵ�kС��
		}
		if (index2 == n)
		{
			return nums1[index1 + k - 1];
		}
		if (k == 1)
		{
			return min(nums1[index1], nums2[index2]);
		}

		int newindex1 = min(index1 - 1 + k / 2, m - 1);	//index-1:ǰ������꣬k/2 Ҫ�Ƚϵ�������
		int newindex2 = min(index2 - 1 + k / 2, n - 1);
		//�ĸ�С�޸��ĸ�
		if (nums1[newindex1] <= nums2[newindex2])
		{
			//��ȥ�ų��������ָ�����kΪʣ�µ���Ҫ�Ƚϵĸ���
			k = k - (newindex1 - index1 + 1);
			//�޸�����
			index1 = newindex1 + 1;
		}
		else
		{
			//��ȥ�ų��������ָ�����kΪʣ�µ���Ҫ�Ƚϵĸ���
			k = k - (newindex2 - index2 + 1);
			//�޸�����
			index2 = newindex2 + 1;
		}
	}
}
double Solution::findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
	int m = nums1.size();
	int n = nums2.size();
	//ż��
	if ((m + n) % 2 == 0)
		return (getKthElement(nums1, nums2, (m + n) / 2) + getKthElement(nums1, nums2, (m + n) / 2 + 1)) / 2.0;
	else
		return getKthElement(nums1, nums2, (m + n + 1) / 2);
}
//5.������Ӵ�

void printArr2(vector<vector<bool>> dp)
{
	for (vector<vector<bool>>::iterator it = dp.begin(); it < dp.end(); it++)
	{
		for (vector<bool>::iterator vit = it->begin(); vit < it->end(); vit++)
		{
			cout << (*vit) << " ";
		}
		cout << endl;
	}
}

string Solution::longestPalindrome(string s)
{
	int n = s.size();
	int maxlen = 1, left = 1, right = 1;
	vector<vector<bool>> dp(n, vector<bool>(n));
	if (n == 1)
	{
		return s;
	}

	//��ʼ��
	for (int i = 0; i < n; i++)
	{
		dp[i][i] = 1;
		if (i > 0)
		{
			dp[i][i - 1] = 1;
		}
	}
	printArr2(dp);

	for (int j = 0; j < n; j++)
	{
		for (int i = 0; i < j; i++)
		{
			if (s[i] != s[j])
			{
				dp[i][j] = false;
			}
			else
			{
				dp[i][j] = (dp[i + 1][j - 1] && (s[i] == s[j]));//����һ��
				if (dp[i][j] == true)
				{
					if (j - i + 1 > maxlen)
					{
						maxlen = j - i + 1;
						left = i;
					}
				}
			}
		}
	}
	cout << "left=" << left << "right=" << right << endl;
	cout << "--------------------------------------------" << endl;
	printArr2(dp);

	return s.substr(left, maxlen);
}

//6.Z���ͱ任
string Solution::convertZ(string s, int numRows)
{
	if (s.size() < numRows || numRows == 1)
	{
		return s;
	}

	bool flag = false;
	int curRow = -1;//��¼��ǰ����
	vector<string> rows(numRows);

	for (char c : s)
	{
		if (flag == false)
		{
			rows[++curRow] += c;
			if (curRow == numRows - 1)
			{
				flag = true;
			}
		}
		else
		{
			rows[--curRow] += c;
			if (curRow == 0)
			{
				flag = false;
			}
		}
	}
	string ret;
	for (vector<string>::iterator it = rows.begin(); it < rows.end(); it++)
		ret += (*it);

	return ret;
}

int Solution::Myreverse(int x)
{
	int rev = 0;
	while (x)
	{
		int ge = x % 10;//���λ��

		//Խ��
		if ((rev > INT_MAX / 10)
			|| (rev == INT_MAX / 10 && ge > INT_MAX % 10)
			|| (rev < INT_MIN / 10)
			|| (rev == INT_MIN / 10 && ge < INT_MIN % 10))
		{
			return 0;
		}

		rev = rev * 10 + ge;//��ӵ����
		x = x / 10;
	}

	return rev;
}

/*
	table�б�ʾ��ǰ��״̬
	�б�ʾ�������ַ�
	��Ӧ��Ԫ��Ϊ��Ҫת����״̬
*/
class Automaton
{
	//ʵ��talble�Ķ���
	//table��
	string state = "start";
	unordered_map<string, vector<string>> table =
	{
		{"start",{"start", "signed", "in_number", "end"}},
		{"signed",{"end", "end", "in_number", "end"}},
		{"in_number", {"end", "end", "in_number", "end"}},
		{"end", {"end", "end", "end", "end"}}
	};

	//����������Ӧ�ķ��ŵ�ʱ����Ҫȥ���ҵ��Ķ�Ӧ��vector ����Ԫ��
	int getcol(char c)
	{
		if (c == ' ') return 0;
		else if (c == '+' || c == '-') return 1;
		else if (c - '0' >= 0 && c - '0' <= 9) return 2;
		else return 3;
	};

public:

	int getResult(string str)
	{
		bool sign = 1;
		int result = 0;
		int num = 0;
		//��¼����

		for (char c : str)
		{
			//���µ�ǰ״̬
			state = table[state][getcol(c)];
			//�����˷���״̬���鿴��ǰ����
			if (state == "signed")
			{
				if (c == '+')
				{
					sign = 1;
				}
				else
				{
					sign = 0;
				}
			}
			else if (state == "in_number")
			{
				num = (c - '0');
				//Խ���ж�
				if (sign == 1 &&
					(result > INT_MAX / 10 ||
						(result == INT_MAX / 10 && num > INT_MAX % 10)
						)
					)
					return INT_MAX;
				if (sign == 0 &&
					(-result < INT_MIN / 10 ||
						(-result == INT_MIN / 10 && -num < INT_MIN % 10))
					)
					return INT_MIN;

				result = result * 10 + num;
			}
			else if (state == "end")
			{
				if (sign == 1)	return result;
				else return (-result);
			}
		}
		if (sign == 1)	return result;
		else return (-result);
	}
};

//9.������

bool Solution::isPalindrome(int x)
{
	if (x < 0)
	{
		return false;
	}
	else if (x < 10)
	{
		return true;
	}

	vector<int> v;
	while (x)
	{
		v.push_back(x % 10);
		x = x / 10;
	}
	if (v.size() == 2)
	{
		if (v[0] == v[1])
		{
			return true;
		}
		else
		{
			return false;
		}
	}
	for (int i = 0; i < v.size() / 2; i++)
	{
		if (v[i] != v[(v.size() - 1) - i])
		{
			return false;
		}
	}
	return true;
}

int Solution::maxArea(vector<int>& height) {
	int maxA = 0;
	vector<int>::iterator left = height.begin(), right = height.end() - 1;
	while (left < right)
	{
		//���㵱ǰ�����
		int tempA = (right - left) * min(*left, *right);
		if (tempA > maxA)
		{
			maxA = tempA;
		}
		if (*left < *right)
		{
			left++;
		}
		else
		{
			right--;
		}
	}
	return maxA;
}
/*
I             1
V             5
X             10
L             50
C             100
D             500
M             1000

int[] values = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
String[] symbols = {"M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"};

*/
string Solution::intToRoman(int num)
{
	vector<int> values = { 1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1 };
	vector<string> sysmbols = { "M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I" };
	string result;
	int index = 0;

	while (num > 0)
	{
		while (num >= values[index])
		{
			result += sysmbols[index];
			num -= values[index];
		}
		index++;
	}
	return result;
}

//13.��������ת����

int getValue(char ch) {
	switch (ch) {
	case 'I': return 1;
	case 'V': return 5;
	case 'X': return 10;
	case 'L': return 50;
	case 'C': return 100;
	case 'D': return 500;
	case 'M': return 1000;
	default: return 0;
	}
}

int Solution::romanToInt(string s) {
	int result = 0;
	for (int i = 0; i < s.size() - 1; i++)
	{
		if (getValue(s[i]) >= getValue(s[i + 1]))
		{
			result += getValue(s[i]);
		}
		else
		{
			result -= getValue(s[i]);
		}
	}
	result += getValue(s[s.size() - 1]);
	return result;
}

string Solution::longestCommonPrefix(vector<string>& strs)
{
	if (strs.empty())
	{
		return "";
	}
	string result = strs[0];
	int j = 0, longest = strs[0].size();
	for (vector<string>::iterator it = strs.begin(); it < strs.end(); it++)
	{
		//����ÿ��Ԫ�أ�*it������ÿ���ַ�
		for (j = 0; j < min(longest, (int)(*it).size()); j++)
		{
			if (result[j] != (*it)[j])
			{
				break;
			}
		}

		if (j == 0)
		{
			return "";
		}
		longest = j > longest ? longest : j;
	}
	return result.substr(0, longest);
}

//15.����֮��
vector<vector<int>> Solution::threeSum(vector<int>& nums)
{
	//������
	sort(nums.begin(), nums.end());

	vector<vector<int>> result;
	//�м�����ο������ߵ���������ֵ�ǲ�����
	//С����Ļ���ߵ����ƣ�������Ļ��ұߵ����� ����������ȥ	->���   0.0   <-��С

	if (nums.size() < 3)
	{
		return result;
	}

	//-1,-1,0,1
	for (int i = 0; i < nums.size() - 2; i++)	//�����м����
	{
		if (i > 0 && nums[i] == nums[i - 1])
			continue;

		int left = i + 1, right = nums.size() - 1;	//˫ָ��
		while (left < right)				//Խ������
		{
			//С����Ļ���ߵ����ƣ�������Ļ��ұߵ����� ����������ȥ	->���   0.0   <-��С
			int sum = nums[i] + nums[left] + nums[right];
			if (sum == 0)
			{
				result.push_back({ nums[i] , nums[left] , nums[right] });
				//�����ظ�
				while (left < right && nums[++left] == nums[left - 1]);
				while (left < right && nums[--right] == nums[right + 1]);
			}
			else if (sum > 0)
			{
				//ȥ���ظ���,	�����ε�leftֵ��rightֵ��������һ�ε�ֵ��˵���ظ���
				while (left < right && nums[--right] == nums[right + 1]);
			}
			else
			{
				while (left < right && nums[++left] == nums[left - 1]);
			}
		}
	}
	return result;
}

//��Ϊ�����Ż���Ŀ��Ҳ����ʹ��Ŀ�꺯����ӽ�������þ���ֵ���̻��ǱȽϺ����,����������������0��
//˼·��������һ����t��Ѱ��������m,n��ʹ��Ŀ�꺯��f=|t-(m+n+k)|��С
//����������˫ָ�룬ÿ�α���ֻ��ȥѰ��һ����ԭ����������ֵС����ֵ���������ˣ�˵��ǰһ����Ϊ��Сֵ
//˫ָ������ʹ�õ�ʱ���������ź���ġ���ôһ�������������Ϊ��left+1��m+n���right-1��m+n��С,���ǿ���ÿ�εĽ�������С��0��󣬴������С����ô�����������0.�����Ŀ�꺯������Сֵ��һ�µ�
int Solution::threeSumClosest(vector<int>& nums, int target)
{
	if (nums.size() < 3)
	{
		int sum = 0;
		for (int i = 0; i < nums.size(); i++)
		{
			sum += nums[i];
		}
		return sum;
	}

	sort(nums.begin(), nums.end());
	int result = 100000;
	//1,2,5,10,11
	for (int i = 0; i < nums.size() - 2; i++)
	{
		int left = i + 1, right = nums.size() - 1;
		while (left < right)
		{
			int sum = nums[i] + nums[left] + nums[right];

			if (abs(target - sum) < abs(target - result))
			{
				result = sum;
			}
			//�仯˫ָ��
			if ((target - sum) == 0)
			{
				return  sum;
			}
			else if ((target - sum) < 0)	//���С��0��sumҪ��С������ܱ��
			{
				//ȥ���ظ���,	�����ε�leftֵ��rightֵ��������һ�ε�ֵ��˵���ظ���	--right
				while (left < right && nums[--right] == nums[right + 1]);
			}
			else
			{
				while (left < right && nums[++left] == nums[left - 1]);
			}
		}
	}

	return result;
}

class letterCombination
{
	vector<string> letterMap =
	{
			" ",    //0
			"",     //1
			"abc",  //2
			"def",  //3
			"ghi",  //4
			"jkl",  //5
			"mno",  //6
			"pqrs", //7
			"tuv",  //8
			"wxyz"  //9
	};
public:

	vector<string>  result;
	void backtrack(int index, string digits, string str)
	{
		//ֹͣ����
		if (index == digits.size())
		{
			result.push_back(str);
			return;
		}
		int num = digits[index] - '0';

		for (int i = 0; i < letterMap[num].size(); i++)
		{
			string tempStr = str;
			tempStr = tempStr + letterMap[num][i];
			backtrack(index + 1, digits, tempStr);
		}
	}
};

//17. �绰�������ĸ���
vector<string> Solution::letterCombinations(string digits)
{
	//���ֱ�
	letterCombination l;
	if (digits.size() == 0)
	{
		vector<string> result;
		return l.result;
	}
	string str = "";
	l.backtrack(0, digits, str);

	cout << l.result.size() << endl;
	for (int i = 0; i < l.result.size(); i++)
	{
		cout << l.result[i] << "\t";
	}
	return l.result;
}

//18. ����֮��
vector<vector<int>> Solution::fourSum(vector<int>& nums, int target)
{
	//a,b,c,d����˫ָ�룬����Ҳ������ƶ���ָ�룬���Ҳ����������ƶ���ָ��
	//�ô����ڣ���������ڲ�����ֵ����Сֵ���Ǵ����㣨����С���㣩���Ϳ���ֱ���жϣ�Ȼ��������һ��ѭ����
	//���ָ��
	int left1 = 0, right1 = nums.size() - 1;
	//�ڲ�ָ��
	int left = 1, right = nums.size() - 2;
	int sum = 0;
	vector<vector<int>> result;
	if (nums.size() < 4)
	{
		return result;
	}
	sort(nums.begin(), nums.end());
	//����a
	// -1, 0, 0, 1,2
	while (left1 < nums.size() - 3)
	{
		//�����ָ���ʼ��
		right1 = nums.size() - 1;
		//�����ָ��仯

		//�ų��������
		int maxVal = target - (nums[left1] + nums[right1 - 2] + nums[right1 - 1] + nums[right1]);
		int minVal = target - (nums[left1] + nums[left1 + 1] + nums[left1 + 2] + nums[right1]);
		//���ֵ����Сֵ��������,˵�������Ǵ�����ģ�ֱ�������ָ��+1
		if (maxVal > 0 && minVal > 0)
		{
			while (nums[++left1] == nums[left1 - 1] && left1 < nums.size() - 3);
		}
		//���ֵ����Сֵ��С����,˵��������С����ģ�ֱ�������ָ��-1
		if (maxVal < 0 && minVal < 0)
		{
			while (nums[--right1] == nums[right1 + 1] && right1 - left1 >= 3);
		}

		while (right1 - left1 >= 3)
		{
			left = left1 + 1;
			right = right1 - 1;

			//�ڲ�˫ָ��ѭ��
			while (left < right)
			{
				sum = nums[left1] + nums[left] + nums[right] + nums[right1];

				if (sum == target)
				{
					result.push_back({ nums[left1] , nums[left] , nums[right],nums[right1] });
					while (nums[--right] == nums[right + 1] && left < right);
					while (nums[++left] == nums[left - 1] && left < right);
				}
				else if (sum > target)
				{
					while (nums[--right] == nums[right + 1] && left < right);
				}
				else
				{
					while (nums[++left] == nums[left - 1] && left < right);
				}
			}
			//���ָ�������ƶ�
			while (nums[--right1] == nums[right1 + 1] && right1 - left1 >= 3);
		}
		while (nums[++left1] == nums[left1 - 1] && left1 < nums.size() - 3);
	}
	return result;
}

ListNode* Solution::removeNthFromEnd(ListNode* head, int n)
{
	ListNode* left = head;
	ListNode* right = head;
	//�ȼ����n��
	for (int i = 0; i < n; i++)
	{
		right = right->next;
	}
	//ͬʱ�������
	while (right->next != NULL)
	{
		left = left->next;
		right = right->next;
	}
	//leftָ��ĵ���n+1���ڵ�
	left->next = left->next->next;

	return head;
}
//20. ��Ч������
bool Solution::isValid(string s)
{
	stack<char> sta;
	//����һ�����ұ�
	//keyΪ�Ҷ�Ԫ�أ�valΪ���Ԫ��
	unordered_map<char, char> pairs = {
	{')', '('},
	{']', '['},
	{'}', '{'}
	};
	for (char c : s)
	{
		//��ջ���������������֮ƥ��������ţ���ôһ����ƥ��ջ����Ԫ�أ�����ƥ��ʧ��
		if (pairs.count(c)	//ƥ�䵽���Ҷ�Ԫ��
			&& sta.empty())	//��ƥ�������������
		{
			return false;
		}
		else if (pairs.count(c)	//ƥ�䵽���Ҷ�Ԫ��
			&& sta.top() != pairs[c])	//��ƥ�������������

		{
			return false;
		}
		else if (pairs.count(c)	//ƥ�䵽���Ҷ�Ԫ��
			&& sta.top() == pairs[c])	//ƥ�䵽�����������
		{
			sta.pop();
			continue;
		}
		else //������
		{
			sta.push(c);
		}
	}
	if (sta.empty())
	{
		return true;
	}
	else
	{
		return false;
	}
}

//21. �ϲ�������������
//l1:[1,3,4]
//l2:[0,2,4]
ListNode* Solution::mergeTwoLists(ListNode* l1, ListNode* l2)
{
	ListNode* preHead = new ListNode(0);
	ListNode* curNode = preHead;

	while (l1 != NULL && l2 != NULL)
	{
		//�Ƚ�,����С���ó���������ǰ�ڵ���棬���߱��뱣�汻�����Ϣ
		if (l2->val > l1->val)
		{
			curNode->next = l1;
			l1 = l1->next;
			curNode = curNode->next;
		}
		else
		{
			curNode->next = l2;
			l2 = l2->next;
			curNode = curNode->next;
		}
	}
	if (l1 == NULL)
	{
		curNode->next = l2;
	}
	if (l2 == NULL)
	{
		curNode->next = l1;
	}

	ListNode* ret = preHead->next;
	delete preHead;
	return ret;
}
//22. ��������

/*
* ��������
* 	1.���������
*	2.��������ţ������������������Ŀ����������
*	����������n��
*
*/
//������ȱ���
class dfs22
{
public:
	vector<string> result;

	void dfs(string curStr, int left, int right)
	{
		if (left == 0 && right == 0)
		{
			result.push_back(curStr);
			return;
		}
		//���������
		if (left > 0)
		{
			dfs(curStr + "(", left - 1, right);
		}
		//���������
		if (left >= right)
		{
			return;
		}
		else
		{
			dfs(curStr + ")", left, right - 1);
		}
	}
};
vector<string> Solution::generateParenthesis(int n)
{
	dfs22 dfs;
	dfs.dfs("", n, n);
	for (int i = 0; i < dfs.result.size(); i++)
	{
		cout << dfs.result[i] << endl;
	}
	return dfs.result;
}

//23.�ϲ�K����������

/*
* ���κϲ���k���ϲ��� k/2,Ȼ���ٺϲ���k/4
*
* �������������ϲ�������
* ����ֵ���ϲ�һ�κ�õ�����ͷ
* �����������ϲ�
*
*/

vector<ListNode*> mergeK(vector<ListNode*>& lists)
{
	if (lists.size() == 1)
	{
		return lists;
	}
	vector<ListNode*> ret;
	int left = 0, right = lists.size() - 1;
	while (left < right)
	{
		ret.push_back(Solution::mergeTwoLists(lists[left++], lists[right--]));
	}
	if (left == right)
	{
		ret.push_back(lists[left]);
	}

	return mergeK(ret);
}

//23. �ϲ�K����������
/*
* �ݹ�д�ϲ�������������
*
* ������
*	�Ƚ���������ͷ���Ĵ�С���õ�ǰָ��ָ���С�Ľڵ�
*
*	����������ͷ���
*	����ֵ����С��ͷ���
*	ֹͣ�������ڵ�Ϊ��
*
*/

ListNode* merge2(ListNode* l1, ListNode* l2)
{
	if (l1 == NULL)
	{
		return l2;
	}
	if (l2 == NULL)
	{
		return l1;
	}
	if (l1->val < l2->val)
	{
		l1->next = merge2(l1->next, l2);
		return l1;
	}
	else
	{
		l2->next = merge2(l1, l2->next);
		return l2;
	}
}

ListNode* Solution::mergeKLists(vector<ListNode*>& lists)
{
	if (lists.empty())
	{
		return NULL;
	}
	ListNode* p = lists[0];
	for (int i = 1; i < lists.size(); i++)
	{
		p = merge2(p, lists[i]);
	}
	return p;
}
//24. �������������еĽڵ�
//3������Ҫ�� ǰ��ģ��м�ģ������
ListNode* swap2(ListNode* head, ListNode* preHead)
{
	//��������
	if (head == NULL)
	{
		return NULL;
		cout << "ż����" << endl;
	}
	if (head->next == NULL)
	{
		return head;
		cout << "������" << endl;
	}
	ListNode* left = head;
	ListNode* right = left->next;
	//�޸������ߵ����ӹ�ϵ
	left->next = right->next;	//���������ڵ����Ϣ
	right->next = left;
	//ǰ��ڵ�ָ�����
	preHead->next = right;

	swap2(left->next, left);
	return right;
}

ListNode* Solution::swapPairs(ListNode* head)
{
	ListNode* left = head;
	ListNode* right = left->next;
	//�޸������ߵ����ӹ�ϵ
	left->next = swapPairs(right->next);	//���������ڵ����Ϣ
	right->next = left;
	//ǰ��ڵ�ָ�����
	return right;
}
//��תһ������
pair<ListNode*, ListNode*> reverseK(ListNode* head, ListNode* tail)
{
	if (tail == NULL)
	{
		return { head,tail };
	}

	ListNode* cur = head;		//��ǰ�ڵ�
	ListNode* pre = tail->next;//pre����ָ��ǰӦ��ָ��Ľڵ�
	while (pre != tail)
	{
		ListNode* temp = cur->next;
		cur->next = pre;
		pre = cur;
		cur = temp;
	}
	return{ tail, head }; //�����µ�ͷ��β
}

//25. K ��һ�鷭ת����
//	���������η�תk������һ�η�ת��β��ָ����һ�ε�ͷ��
ListNode* Solution::reverseKGroup(ListNode* head, int k)
{
	ListNode pre(0, head);
	ListNode* right = &pre;
	ListNode* left = &pre;

	bool flag = true;
	while (flag == true)
	{
		right = left;
		//������k��
		for (int i = 0; i < k; i++)
		{
			if (right == NULL)
			{
				flag = false;
				break;
			}
			right = right->next;
		}
		//�ҵõ�
		if (flag == true)
		{
			pair<ListNode*, ListNode*> result = reverseK(left->next, right);
			left->next = result.first;
			left = result.second;
		}
	}
	return pre.next;
}
//26. ɾ�����������е��ظ���
int removeDuplicates(vector<int>& nums) {
	if (nums.empty())
	{
		return 0;
	}

	//i:��ָ�룬j��ָ��
	int i = 1, j = 1;

	while (j < nums.size())
	{
		//�ҵ���ͬ��j�����һλ
		if (nums[j] != nums[j - 1])
		{
			nums[i++] = nums[j++];
		}
		else
		{
			j++;
		}
	}
	return (i + 1);
}

//27. �Ƴ�Ԫ��
//i jΪ����ָ�� ��i����Ӧ�÷Ž��ķ� val ��λ�ã�jΪ��ָ��
//j����val,��ô����Ž�iλ��
//j��val,��ô��������
int Solution::removeElement(vector<int>& nums, int val)
{
	int i = 0, j = 0;
	int ret = nums.size();
	while (j < nums.size())
	{
		if (nums[j] != val)
		{
			nums[i++] = nums[j++];
		}
		else
		{
			j++;
		}
	}
	return i;
}

//28. ʵ�� strStr()

//��ѧǰ�᣺������
//һ����������ǰ��׺�Ӵ�Ҫô�����ͬǰ��׺��Ҫô����
// jǰ׺ĩβ
// i��׺ĩβ��
//  [] [] [] [] [] [] [] []
//	   j     i
//next��ʾ����1.ǰ���Ӵ������ͬǰ��׺ 	2.���������ƥ�䣬��ôj��Ҫ���˵���λ��
//��ǰ�Ѿ�����������ǣ��Ѿ��з��������� ǰ���������������һ���ַ���ƥ��Ļ���Ӧ����Сƥ�䷶Χ�������ܳɹ��������һ������λ�ã�
//���λ��Ҫ���ϵ��������ڣ�ǰ����ַ��Ϳ�ͷ��ͬ���ַ������ܶ�
//�������ǡ�þ�������������ͬǰ��׺
//����������Ҫ��j����˵�next[j]ָ��ļ���

//��ȷnextָ�������ӦΪ��ǰָ���ǰ��׺��ǰ���ַ����� ���ͬǰ��׺

vector<int> KMPNext(string pat)
{
	vector<int> next(pat.size() + 1);	//i++�����Ժ�ᵽ��pat.size()
	int j = -1;//ǰ׺ĩβ
	int i = 0;//��׺ĩβ
	next[0] = -1;
	while (i < pat.size())
	{
		if (j == -1		//���������������Ļ������ܲ���ȵľͲ���������飬�������Ա�֤ÿ�����鶼�����������
			|| pat[j] == pat[i])
		{
			i++;
			j++;
			next[i] = j;
		}
		else
		{
			j = next[j];//�������Ӵ������ͬǰ��׺��λ��
		}
	}

	for (int i = 0; i < pat.size(); i++)
	{
		cout << next[i] << "\t";
	}
	cout << endl;
	return next;
}

int Solution::strStr(string haystack, string needle)
{
	if (needle.empty())
	{
		return 0;
	}
	vector<int> next = KMPNext(needle);//�����next�Ӵ�
	int i = 0, j = 0;//ָ��ģʽ����ָ��
	while (i < haystack.size())
	{
		if (j == -1		//ָ��ָ����ǵ�һ���Ͳ�ƥ��
			|| haystack[i] == needle[j])
		{
			j++;
			i++;
		}
		else
		{
			j = next[j];
		}
		if (j == needle.size())
		{
			return i - j;
		}
	}
	return -1;
}

//����ͬǰ��׺
vector<int> samPreSuf(string s)
{
	int sublen = 1;

	vector<int> result;
	while (sublen < s.size())
	{
		bool flag = true;
		int  sufstart = s.size() - sublen;//������ʼλ��
		//���֮���ÿ���ַ�
		for (int i = 0; i < sublen; i++)
		{
			if (s[i] != s[sufstart + i])
			{
				flag = false;
				break;
			}
		}
		if (flag == true)
		{
			result.push_back(sublen);
		}

		sublen++;
	}
	return result;
}

//29. �������
//dividend��������divisor����
//

int divR(long long dividend, long long  divisor)
{
	if (dividend < divisor)
	{
		return 0;
	}

	int numCur = divisor;
	int result = 1;

	while (dividend > numCur + numCur)
	{
		result += result;//result=2*result		�洢�ҵ��˶��ٸ�divisor
		numCur += numCur;
		if (numCur > INT_MAX / 2)
		{
			break;
		}
	}
	result += divR(dividend - numCur, divisor);

	return result;
}

int Solution::divide(int dividend, int divisor)
{
	if ((dividend == INT_MIN) && divisor == (-1))
	{
		return INT_MAX;
	}
	if (divisor == 1)
	{
		return dividend;
	}
	long long a = dividend;
	long long b = divisor;

	bool sign = (a > 0 && b > 0) || (a < 0 && b < 0);
	a = a > 0 ? a : -a;
	b = b > 0 ? b : -b;
	if (sign == true)
	{
		return divR(a, b);
	}
	else
	{
		return -divR(a, b);
	}
}

//30. �������е��ʵ��Ӵ�

//����������ƥ��ĵ��ʶ���¼����ϣ���У���ϣ���valueֵΪ�����ظ��Ĵ���
//ά��һ������Ϊ ���е��ʵĺ� �Ļ������ڣ�ÿ�ν�������һ�� ���� ����->����Ӧ����һ���ַ�������ƥ��Ļ���һ��ǡ�����ַ�������������λ��
//��ͷ��ʼ���μ�鴰����ÿ�������Ƿ����Ҫ�������������ô�ͽ����������ƶ�һ���ַ�
//���ķ������£�

//�����鵽β������ÿ�����ʶ����ڣ���ô��������
//����������ֹͣ����������
//	1.û�ҵ���ǰ�ĵ���
//	2.�ҵ���������ʣ�������������

//"barfoothefoobarman"
//["foo", "bar"]

vector<int> Solution::findSubstring(string s, vector<string>& words)
{
	vector<int> result;
	if (words.empty() || s.empty())
	{
		return result;
	}
	unordered_map<string, int> wordcnt;
	//��ʼ��������
	for (auto& w : words)
		wordcnt[w]++;
	int singleLen = words[0].size();//ÿ�����ʵĳ���
	int left = 0, right = left;

	//left ������߽�
	for (int left = 0; left + singleLen * words.size() <= s.size(); left++)
	{
		unordered_map<string, int> curWindow;//��¼��ǰ���ڵĵ��ʺ���
		//��鵱ǰ���ڣ���ͷ��β����
		int count = 0;
		right = left;
		string temp = "";
		for (count = 0; count < words.size(); count++)
		{
			temp = s.substr(right, singleLen);//��¼���Ƚϵĵ���
			if (wordcnt.find(temp) == wordcnt.end() || curWindow.count(temp) > wordcnt.count(temp))	//û���� �� �Ҷ���
			{
				break;
			}
			//�ҵ���
			curWindow[temp]++;
			right += singleLen;
		}
		if (curWindow == wordcnt)
		{
			result.push_back(left);
		}
		if (curWindow.count(temp) > wordcnt.count(temp))//�������Ϊ��ƥ��ĵ�����ƥ��ʧ��
		{
			curWindow.erase(s.substr(left, singleLen));//�ǿ��Խ���ߵ�ȥ���ٿ���
		}
		else
			curWindow.clear();
	}
	return result;
}

//31. ��һ������
/*
1. �Ӻ���ǰ�ҵ���һ������յ�[i,j]��
2. Ȼ���j��ʼ���������ҳ�һ����i����ҿ�������ֽ���(������ǰ���Ժ������С��)�һ���
3. Ȼ�󽫴�j��ʼ�����ֽ�������
*/

//[1,3,2]
void Solution::nextPermutation(vector<int>& nums)
{
	if (nums.empty())
	{
		return;
	}

	int right = nums.size() - 1;
	int left = nums.size() - 2;
	while (left >= 0)
	{
		if (nums[left] < nums[right])
		{
			break;
		}
		left--;
		right--;
	}
	//�Ѿ�����������
	if (left < 0)
	{
		sort(nums.begin(), nums.end());
		return;
	}
	while ((right + 1 < nums.size()) && nums[left] < nums[right + 1])
	{
		right++;
	}
	swap(nums[left], nums[right]);
	sort(nums.begin() + left + 1, nums.end());
}
//32. ���Ч����
/*˼·����̬�滮
*
��ʼ˼·��
��ͷ������������м�¼ÿ�������Ե�ǰ�ַ�Ϊ��β�����׺�Ӵ��ĳ��� ��¼��dp[]��
������µ����ż���
		1.����������ţ��ǿ϶�����Ϊ��β�����׺�Ӵ�������0
		2.����������ţ����ǿ�����ǰ�������ŵ�(�����˹����һ��)������Ҳֻ����һ����

			��������������ŵĻ�(( )���ǲ���˵����ǰ�ĺ�׺�Ӵ�ֵ�϶�����2��ǰ��Ĳ��ÿ��ˣ���������ֻ��Ҫ��ǰ�������ַ���ֵ�ǲ��������ţ���������������ţ���ô��ǰ��ֵ����2
			����� )() ���� dp��ôӦ���ǣ���һ�������ŵ�dp[]ֵ+2
			����� ())
			����� )))		dp��ôӦ���ǣ���һ�������ŵ�dp[]ֵ+�Լ�ƥ�������ŵĽ��
									if s(i-dp[i-1])is ')'=		dp[i-1]+2
									else dp[i]=0

����֤���Ǵ���ģ�����()(())
*/

//�ڶ���˼·�������ַ�������¼��Ч��������������������
//���ַ�Ϊ������ʱ��left++��
//��Ϊ�����ŵ�ʱ��,��������Ŵ��ڵ��������ţ���ô����right++�����Ҽ������Ҳ�������ӡ�����������������������ţ���ô����������Ƕϵ��ģ���˵�ǰ�ļ������㡣
//Ȼ���ڴ�β��ͷ����һ��
int Solution::longestValidParentheses(string s)
{
	//"()(())"  s[i - dp[i - 1] - 1]
	int maxans = 0;
	vector<int> dp(s.size());
	if (s.size() < 2)
	{
		return 0;
	}
	if (s[0] == '(' && s[1] == ')')
	{
		dp[1] = 2;
		maxans = 2;
	}
	for (int i = 2; i < s.size(); i++)
	{
		if (s[i] == ')')
		{
			if (s[i - 1] == '(')
				dp[i] = dp[i - 2] + 2;
			else if (s[i - 1] == ')')
			{
				//Խ��
				if (i - dp[i - 1] - 1 < 0)
				{
					dp[i] = 0;
				}
				else
				{
					if (s[i - dp[i - 1] - 1] == '(')
						if (i - dp[i - 1] - 2 > 0)	dp[i] = dp[i - 1] + 2 + dp[i - dp[i - 1] - 2];
						else dp[i] = dp[i - 1] + 2;
					else dp[i] = 0;
				}
			}
		}
		maxans = max(maxans, dp[i]);
	}
	return maxans;
}
//33. ������ת��������

/*
	ԭ��1.��ת������������Ժ�϶���һ���������
		  2.�����������У��ȽϿ�ͷ�ͽ�β�Ϳ���֪��Ŀ��ֵ�Ƿ���������
		  3.�ж��ڲ������������У����ɴﵽ��֦��Ŀ�ģ���ɿ�������

	���������ֲ��ң��Ƚ�nums[left]��nums[mid]��ֵ���ж�����ǲ����������飬
				�ǵĻ�����߽��бȽϣ���target�ǲ��������������У�������ڣ���ô�����ұ�����

*/
int Solution::search(vector<int>& nums, int target)
{
	int left = 0, right = nums.size() - 1;
	int mid = (left + right) / 2;
	while (left <= right)
	{
		mid = (left + right) / 2;
		if (nums[mid] == target) return mid;
		if (nums[left] <= nums[mid])
		{
			//�����������
			if (target >= nums[left] && target < nums[mid])
			{
				//�������������
				right = mid - 1;
			}
			else
			{
				left = mid + 1;//���ұ���������
			}
		}
		else
		{
			//�ұ���������
			if (target > nums[mid] && target <= nums[right])
			{
				left = mid + 1;//���ұ���������
			}
			else
			{
				//�������������
				right = mid - 1;
			}
		}
	}
	return -1;
}

//34. �����������в���Ԫ�صĵ�һ�������һ��λ��
/*
˼·��
	���ֲ��ң��ҵ����Ԫ���Ժ��������ң����ط�Χ

*/

//��mid�������ҷ�Χ
vector<int> MysearchRange(vector<int>& nums, int mid, int target)
{
	int left = mid;
	int right = mid;
	while (left - 1 >= 0 && nums[left - 1] == target)
	{
		left--;
	}

	while (right + 1 < nums.size() && nums[right + 1] == target)
	{
		right++;
	}
	return { left,right };
}
//5,7,7,8,8,10
vector<int> Solution::searchRange(vector<int>& nums, int target)
{
	if (nums.empty())
	{
		return { -1,-1 };
	}
	int left = 0;
	int right = nums.size() - 1;
	int mid = 0;
	while (left <= right)
	{
		mid = (left + right) / 2;
		if (nums[mid] == target) return MysearchRange(nums, mid, target);
		if (target < nums[mid])
		{
			right = mid - 1;
		}
		else
		{
			left = mid + 1;
		}
	}
	return { -1,-1 };
}
//35.��������λ��
/*
˼·�����ֲ���
	�ҵ��˷���
	�Ҳ�����ô��ʱleft=right,���������ӽ��ģ������������Ŀ��ֵ����С��Ȼ��������뵽��߻����ұ�

*/
//1,3,5,6
int Solution::searchInsert(vector<int>& nums, int target)
{
	if (nums.empty())
	{
		return 0;
	}
	int left = 0;
	int right = nums.size() - 1;
	int mid = 0;
	while (left < right)
	{
		mid = (left + right) / 2;

		if (nums[mid] == target) return mid;
		if (target < nums[mid])
		{
			//�������
			right = mid - 1;
		}
		else
		{
			left = mid + 1;
		}
	}
	//left==right
	if (nums[left] == target)
	{
		return left;
	}
	else if (target < nums[left])
	{
		return left;
	}
	else
	{
		return left + 1;
	}
}

//36. ��Ч������
/*
˼·������������ֱ�洢��ǰ���У��У�box�У�������ϣ��ÿ�������µ������Ȳ鿴һ�µ�ǰ��������Ƿ��ǳ��ֹ���������ֹ�����ôֱ�ӷ���false����ʾ����
				1						2						3
1		(0,0)-(2,2)->0			(3,0)-(5,2)->1			(6,0)-(8,2)->2		row/3
2		(0,3)-(2,5)->4			(3,3)-(5,5)->5			(6,3)-(8,5)->6
3		(0,6)-(2,8)->7			(3,6)-(5,8)->8			(6,6)-(8,8)->9

		col/3*3+row/3
		����

*/
bool Solution::isValidSudoku(vector<vector<char>>& board)
{
	int row[9][10] = { 0 };//��¼��
	int col[9][10] = { 0 };//��¼��
	int box[9][10] = { 0 };//��¼ÿ�����
	//������������
	for (int i = 0; i < 9; i++)
	{
		for (int j = 0; j < 9; j++)
		{
			if (board[i][j] == '.')
				continue;
			int num = board[i][j] - '0';

			//�и�ֵ
			if (row[i][num] == 0)
			{
				row[i][num] = 1;
			}
			else
			{
				return false;
			}
			//�и�ֵ
			if (col[j][num] == 0)
			{
				col[j][num] = 1;
			}
			else
			{
				return false;
			}
			//box��ֵ
			if (box[j / 3 * 3 + i / 3][num] == 0)
			{
				box[j / 3 * 3 + i / 3][num] = 1;
			}
			else
			{
				return false;
			}
		}
	}
	return true;
}

//37. ������
/*
˼·������=���+��֦
���������鱣���У��кͿ�Ķ�ӦԪ�صĸ���
Ȼ��һ������������ţ�����鵽��Ԫ�س�ͻ��ʱ��ֱ��return,��ͷ�Żض�û���⣬�Ǿͽ�����Ž�result,Ȼ�󷵻�
*/
//���ܣ���x,y���λ�÷���һ�� 0-9 ��ֵ
/*
vector<vector<int>> result(9, vector<int>(10, 0));
void placeNum(int x,int y,vector<vector<int>> row, vector<vector<int>> col, vector<vector<int>> box, vector<vector<int>> curBoard)
{
	//��֦����
	if (row[x][num] == 0){
		row[x][num] = 1;//�и�ֵ
	}
	else{
		return;
	}
	if (col[y][num] == 0){
		col[y][num] = 1;//�и�ֵ
	}
	else{
		return;
	}
	if (box[x / 3 * 3 + y / 3][num] == 0){
		box[x / 3 * 3 + y / 3][num] = 1;//box��ֵ
	}
	else{
		return;
	}
	//��ǰλ�ÿ��Է��������,�¸�λ�÷�����
	curBoard[x][y] = num;
}

void Solution::solveSudoku()
{
	vector<vector<int>> row(9, vector<int>(10, 0));//��¼��
	vector<vector<int>> col(9, vector<int>(10, 0));//��¼��
	vector<vector<int>> box(9, vector<int>(10, 0));//��¼ÿ�����
	vector<vector<int>> curBoard(9, vector<int>(10, 0));
	for (int i = 0; i < 10; i++)
	{
		placeNum(0, 0, i, row, col, box, curBoard);
	}

	for (int i = 0; i < 9; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			cout << result[i][j] << "\t";
		}
		cout << endl;
	}
}
*/
//38. �������
/*
˼·��
Ҫ ���� һ�������ַ���������Ҫ���ַ����ָ�Ϊ ��С �������飬ÿ���鶼����������� ��ͬ�ַ� ��ɡ�
Ȼ�����ÿ���飬�������ַ���������Ȼ�������ַ����γ�һ�������顣
Ҫ������ת��Ϊ�����ַ������Ƚ�ÿ���е��ַ������������滻���ٽ���������������������

*/
string Solution::countAndSay(int n) {
	string result = "";
	if (n == 1) {
		return "1";
	}
	else {
		string s = countAndSay(n - 1);
		int left = 0, right = 0;
		while (right < s.size())
		{
			while (right < s.size() && s[left] == s[right]) { right++; }
			result += right - left + '0';
			result += s[left];
			if (right == s.size()) break;
			left = right;
		}
		return result;
	}
}

//39. ����ܺ�
/*
˼·��
�����ݹ����
ÿ�μ�ȥָ������

ѭ����
1.�鿴�Լ��ǲ��������С����	��������ǰ��ֵ
2.���������Ļ�����ǰ�����������¼�candidates�е�����,����¼��ǰ��ȥ����

ֹͣ���������������߼���0
*/
class dfs39
{
public:
	vector<vector<int>> result;
	void dfs(int num, int begin, vector<int> candidates, vector<int> path)
	{
		if (num == 0)
		{
			result.push_back(path);
			return;
		}
		else if (num < 0)
		{
			return;
		}
		else
		{
			for (int i = begin; i < candidates.size(); i++)
			{
				path.push_back(candidates[i]);//��i��������ȥ����Ч��
				dfs(num - candidates[i], i, candidates, path);
				path.pop_back();//�������ó�����,������һ��
			}
		}
	}
};

vector<vector<int>> Solution::combinationSum(vector<int>& candidates, int target) {
	dfs39 d;
	d.dfs(target, 0, candidates, {});
	return d.result;
}

//40. ����ܺ� II
/*
˼·������
�����٣�
����һ�����ֵ����ó��������ǲ��ǵ���target���������õڶ����������Ǿ��õ�����,ָ��ֻ����ָ

��֦�� �����ּ�С���㣬������ͷ�������˸���������

ȥ��:����ֻ����ͬ���ֵĵ�һ�����б���������������
*/

class dfs40
{
public:
	vector<vector<int>> result;
	void dfs(int num, int begin, vector<int> candidates, vector<int> path)
	{
		if (num == 0)
		{
			result.push_back(path);
			return;
		}
		else if (num < 0)
		{
			return;
		}
		else
		{
			if (begin >= candidates.size())
				return;

			for (int i = begin; i < candidates.size(); i++)
			{
				if (i > begin && candidates[i] == candidates[i - 1])
					continue;
				path.push_back(candidates[i]);//��i��������ȥ����Ч��
				dfs(num - candidates[i], i + 1, candidates, path);
				path.pop_back();//�������ó�����,������һ��
			}
		}
	}
};

vector<vector<int>> Solution::combinationSum2(vector<int>& candidates, int target) {
	sort(candidates.begin(), candidates.end());
	dfs40 d;
	d.dfs(target, 0, candidates, {});
	return d.result;
}
//41. ȱʧ�ĵ�һ������
/*
˼·��ԭ�ع�ϣ��
ԭ��ȱʧ�ĵ�һ�������϶����ᳬ�����鳤��
���
����һ�����鳤��n+1��С�����飬������ϣ��һ�α����ҵ�ȱʧ��Ԫ��,���Ƕ�Ӧ����С���������������⻨����n�Ŀռ�

���ǿ������������鹹��ɶ�Ӧ�����Ӽ���
���췽����
��ͷ��β�������飬��鵱ǰ�����ǲ����ڶ�Ӧ��λ����
1 2 3 4 5 6 7	value
0 1 2 3 4 5 6	index

������ڵĻ�����Ҫ���������������Ӧ��λ����ȥ

�����ͻ��(������һ��)����ô������һ������ͻ��λ��

����ٴ�ͷ����һ���ҵ�ȱʧ��Ԫ�ؼ���
			//���ڴ�������������Ҫ����
			//�����Χ�� 0-N�ģ����佻������Ӧ��λ��
			//����N�ģ���else �͸���һ�� ��������������ν

*/
int Solution::firstMissingPositive(vector<int>& nums) {
	int i = 0;
	while (i < nums.size())
	{
		if (nums[i] > 0 && nums[i] <= nums.size()) {
			if (nums[i] - 1 == i) {
				//�ڶ�Ӧ��λ���ϵ�
				i++;
			}
			else {
				//���ڶ�Ӧ��λ����,��������Ӧ��λ����ȥ
				//���������λ���Ѿ����˶�Ӧ��Ԫ�����ˣ���ô��ǰ�����ʵ�Ѿ�ûʲô�����ˣ����Ե����ͷ���һ������
				if (nums[i] == nums[nums[i] - 1]) {
					i++;
					continue;
				}
				swap(nums[i], nums[nums[i] - 1]);
			}
		}
		else {
			//С����Ļ����Ǵ���N���޹ؽ�Ҫ����,�Ǿ���һ����
			i++;
		}
	}
	for (i = 0; i < nums.size(); i++)
	{
		if (nums[i] - 1 != i)
		{
			break;
		}
	}
	return i + 1;
}

//42. ����ˮ
/*
˼·��1.������ӵ�����һ�������ߵģ���ô������ľ͵�ͷ�ˣ�ǰ��ľͺͺ���û��ϵ�ˣ�����Ͱ�������ס�ˣ���¼��Ŀǰ����ʢ��ˮ

/////����۵㣬�����������м�һ����/////	2.�������һ���������ģ���ô�������󿴵�ͷ������¼��Ŀǰ���������(ֻҪ�������߾���)��Ŀǰ��ʢˮ��

���е�һ��˼·��

ԭ������ǰ���1.
�����������ҵ��ܵ�ס����ߵ��壬Ȼ���������м����

*/
// 0,1,0,2,1,0,1,3,2,1,2,1
int Solution::trap(vector<int>& height) {
	int left = 0, right = 0;
	int maxh = 0;
	int maxIndex = 0;
	int water = 0;
	for (int i = 0; i < height.size(); i++)
	{
		if (height[i] > maxh)
		{
			maxh = height[i];
			maxIndex = i;
		}
	}
	while (left < maxIndex)
	{
		if (height[left] > height[right])
		{
			water += height[left] - height[right];
		}
		else
		{
			left = right;
		}
		right++;
	}
	right = height.size() - 1;
	left = right;

	while (left > maxIndex)
	{
		if (height[right] > height[left])
		{
			water += height[right] - height[left];
		}
		else
		{
			right = left;
		}
		left--;
	}

	return water;
}
//43. �ַ������
/*
˼·��ģ������ʽ�ķ�ʽ
����һ�����ĳ˱�������ÿһλ��Ȼ�����ӷ�����¼��λ�ͽ��λ����������˷�����

*/
/*����:��װ�˷���
* num1,num2:���������ַ�
* ����ֵ��<��λ����λ>
*/
pair<char, char> multiplySingle(char num1, char num2, char carry)
{
	int n1 = num1 - '0';
	int n2 = num2 - '0';
	int n3 = carry - '0';
	int result = n1 * n2 + n3;
	return pair<char, char>{result / 10 + '0', result % 10 + '0'};
}

/*����:��װ�ӷ���
* num1,num2:���������ַ�,��һ���������û��ƫ�Ƶ������ڶ�����ƫ�ƹ�������n��ƫ����
* ����ֵ��<��λ����λ>
*/
string addNumString(string num1, string num2, int n)
{
	if (num1.empty())
		return num2;
	if (num2.empty())
		return num1;

	string result = "";
	for (int i = 0; i < n; i++)
	{
		if (i < num1.size())
			result += num1[num1.size() - 1 - i];
		else
			result += '0';
	}

	int temp = 0, carry = 0;
	for (int i = 0; i < num2.size() || i + n < num1.size(); i++)
	{
		//����������
		if (i + n < num1.size() && i < num2.size())
		{
			temp = num1[num1.size() - 1 - n - i] - '0' + num2[num2.size() - 1 - i] - '0';
			result += (temp + carry) % 10 + +'0';
			carry = (temp + carry) / 10;
		}
		else if (i < num2.size())
		{
			//ֻ�еڶ�������
			temp = num2[num2.size() - 1 - i] - '0';
			result += (temp + carry) % 10 + '0';
			carry = (temp + carry) / 10;
		}
		else
		{
			//ֻ�е�һ������
			temp = num1[num1.size() - 1 - n - i] - '0';
			result += (temp + carry) % 10 + '0';
			carry = (temp + carry) / 10;
		}
	}
	if (carry == 1)
		result += carry + '0';

	reverse(result.begin(), result.end());
	return result;
}
string multiplySring(string num1, char num2)
{
	char carry = '0';
	string curRes = "";
	for (int j = num1.size() - 1; j >= 0; j--)
	{
		pair<char, char> temp = multiplySingle(num1[j], num2, carry);
		carry = temp.first;
		curRes += temp.second;
	}
	//�������Ľ�λ�����ý���õ���ȷ��˳��,�ǳ�һλ�õ��Ľ��
	if (carry != '0')
		curRes += carry;
	reverse(curRes.begin(), curRes.end());
	return curRes;
}

string Solution::multiply(string num1, string num2) {
	if (num1 == "0" || num2 == "0")
	{
		return "0";
	}

	string result = "";
	for (int i = num2.size() - 1; i >= 0; i--)
	{
		string curRes = multiplySring(num1, num2[i]);
		//��ǰ��ļ�����
		result = addNumString(result, curRes, num2.size() - 1 - i);
	}
	return result;
}
//44. ͨ���ƥ��
/*
˼·�������

*/

bool Solution::isMatch(string s, string p)
{
	if (s.empty()) {
		for (int i = 0; i < p.size(); i++) {
			if (p[i] != '*')
				return false;
		}
		return true;
	}
	int row = p.size() + 1;
	int col = s.size() + 1;
	vector<vector<int>> map(row, vector<int>(col));
	//��ʼ��
	switch (p[0])
	{
	case '*':
		map[0] = vector<int>(col, 1);
		break;
	default:
		map[0][0] = 1;
		break;
	}

	for (int i = 1; i < row; i++) {
		for (int j = 1; j < col; j++) {
			switch (p[i - 1])
			{
			case '*'://�����ǰ��*�Ļ�����ô�������ƥ���������������������Ķ����ԣ�����Ӧ���Ǵ������ 1 ��ʼ
				//���*����Ŀ�����Ϊ�մ��Ļ�������Ҫ��֤����ǰ��Ŀ������ӵ��ϣ��������*���ڿ�ͷ�Ļ��ͻ����Ӳ��ϣ�����������Ҫ�������ӵ�����
				if (map[i - 1][0] == 1) {
					j = 0;//����һ�ж��������
					while (j < col) map[i][j++] = 1;
				}
				else if (map[i - 1][j] == 1)
					while (j < col) map[i][j++] = 1;
				break;
			case '?':
				if (map[i - 1][j - 1] == 1)
					map[i][j] = 1;
				break;
			default:
				if (map[i - 1][j - 1] == 1 && p[i - 1] == s[j - 1])
					map[i][j] = 1;
				break;
			}
		}
	}
	printVector2(map);
	return map[row - 1][col - 1];
}

//45. ��Ծ��Ϸ II
/*
˼·��̰���㷨
ԭ����֤��ÿ�ζ��������ҿ������յ��ķ�Χ�����λ�ã��������ԱȽϵ��������Ϣ
ÿ�ζ�����һ����Χ�ܴ��λ�ã�ֱ����������ȥΪֹ

*/
int Solution::jump(vector<int>& nums) {
	if (nums.size() == 1) {
		return 0;
	}
	int times = 0;
	int end = nums[0];//��ǰ����Ȧ��β
	int maxpos;//��һ������Ȧ�Ľ�β
	for (int i = 0; i < nums.size() - 1; i++) {
		//����ÿ������
		maxpos = max(maxpos, i + nums[i]);//����һ��������Χ����,����Ӧ��λ�þ���Ҫ������λ��
		if (i == end) {
			end = maxpos;
			times++;
		}
	}
	return times + 1;
}

//46. ȫ����
//����ÿ���ڵ�Ҫ�ɵ���
class dfs46 {
private:

public:
	vector<vector<int>> result;
	vector<int> nums;
	dfs46(vector<int>& nums) {
		this->nums = nums;
	}

	void dfs(int depth, vector<bool>& isUsed, vector<int>& path) {
		if (depth == nums.size()) {
			result.push_back(path);
			return;
		}
		for (int i = 0; i < nums.size(); i++) {
			if (isUsed[i] == false) {
				isUsed[i] = true;
				path.push_back(nums[i]);
				dfs(depth + 1, isUsed, path);
				path.pop_back();
				isUsed[i] = false;
			}
		}
	}
};

vector<vector<int>> Solution::permute(vector<int>& nums) {
	dfs46 df(nums);
	vector<bool> isUsed(nums.size(), false);
	vector<int> path;
	df.dfs(0, isUsed, path);
	return df.result;
}

//47. ȫ���� II
class dfs47 {
private:

public:
	vector<vector<int>> result;
	vector<int> nums;
	dfs47(vector<int>& nums) {
		this->nums = nums;
	}

	void dfs(int depth, vector<bool>& isUsed, vector<int>& path) {
		if (depth == nums.size()) {
			result.push_back(path);
			return;
		}
		for (int i = 0; i < nums.size(); i++) {
			if (i >= 1 && nums[i] == nums[i - 1]) {
				continue;
			}
			if (isUsed[i] == false) {
				isUsed[i] = true;
				path.push_back(nums[i]);
				dfs(depth + 1, isUsed, path);
				path.pop_back();
				isUsed[i] = false;
			}
		}
	}
};
vector<vector<int>> Solution::permuteUnique(vector<int>& nums) {
	sort(nums.begin(), nums.end());
	dfs47 df(nums);
	vector<bool> isUsed(nums.size(), false);
	vector<int> path;
	df.dfs(0, isUsed, path);
	return df.result;
}

//48. ��תͼ��
/*
˼·����ת�ĸ��Ǽ���
*/
void Solution::rotate(vector<vector<int>>& matrix) {
	int left = 0;
	int right = matrix.size() - 1;
	int len = right - left;

	int temp;
	while (left < right)
	{
		len = right - left;
		for (int i = 0; i < len; i++) {
			temp = matrix[left][left + i];
			//��ʼ��ת
			matrix[left][left + i] = matrix[right - i][left];
			matrix[right - i][left] = matrix[right][right - i];
			matrix[right][right - i] = matrix[left + i][right];
			matrix[left + i][right] = temp;
		}
		left++;
		right--;
	}
}
//49. ��ĸ��λ�ʷ���
/*
�õ���������+��ϣ��ķ������������ַ��϶���һ���ģ��������������ַ���Ϊkeyֵ��valueΪһ��string���vector��ÿ������������ַ�����

����һ�ַ����Ǽ����ķ�����ԭ������ÿ�����������Ӧ����ĸ�ĸ����϶���һ����
*/
vector<vector<string>> groupAnagrams(vector<string>& strs) {
	unordered_map<string, vector<string>> map;
	for (string& str : strs) {
		string temp = str;
		sort(temp.begin(), temp.end());
		map[temp].push_back(str);
	}
	vector<vector<string>> result;
	for (unordered_map<string, vector<string>>::iterator it = map.begin(); it != map.end(); it++) {
		result.push_back(it->second);
	}
	return result;
}
//50. Pow(x, n)
/*
˼·��x��n�η�����Ӧ���Ƕ��ٸ�n��ˣ���x4�ǿ�����x2ֱ��ƽ������
x11��x8��x2��x
��Ӧ���Ƕ����Ƶ�11  1011����λ�ľ���x���η��ĸ���

��������n�����ƻ���ÿ�ζ��˵�ǰ�������õ�x,x2,x4,x8�������Ƶ�ÿһλ��Ӧ��Ҫ��Ҫ�˵�ǰ�õ�����

*/

double Solution::myPow(double x, int n) {
	double result = 1;
	int sign = n > 0 ? 0 : 1;
	while (n != 0) {
		if (abs(n % 2) == 1) {
			//���λ��1,��Ҫ����ȥ
			result *= x;
		}
		x *= x;
		n /= 2;
	}
	if (sign == 0)	return result;
	else return 1 / result;
}
//51. N �ʺ�
/*
˼·:
	���ݣ����+��֦
	ÿ���ڵ�Ҫ�����£��鿴��ǰ�ڵ��Ƿ����Ҫ�����������Ҫ�󷵻�
						�������Ҫ������һ�п�
*/
class dfs51 {
public:
	vector<vector<string>> result;
	int N;
	dfs51(int n) {
		N = n;
	}
	void dfs(int row, vector<string>& path, vector<bool>& isused, unordered_map<int, int>& coordinate) {
		if (row == N - 1) {
			result.push_back(path);
			return;
		}
		//����Ҫ�󣬿�������һ�е�
		for (int i = 0; i < N; i++) {
			//�����ܲ����������
			//������ row+1 ,i ��һλ�÷ţ����Էŵ���������if�����ˣ�ͬ����Ҳ���Ǽ�֦����
			if (isused[i] == false && isXie(row + 1, i, coordinate)) {	//��һ��û����,	//�ų�һ��б�Խǵ�
				isused[i] = true;		//û����������
				path[row + 1][i] = 'Q';
				coordinate[row + 1] = i;
				dfs(row + 1, path, isused, coordinate);	//��ȥ����һ���˰�
				path[row + 1][i] = '.';		//�����˻���
				isused[i] = false;
			}
		}
	}
	bool isXie(int x, int y, unordered_map<int, int>& coordinate) {
		for (int i = 0; i < x; i++) {
			if (abs(y - coordinate[i]) == abs(x - i))
				return false;
		}
		return true;
	}
};

vector<vector<string>> Solution::solveNQueens(int n) {
	vector<bool> isused(n, false);
	dfs51 d(n);
	vector<string> path(n, string(n, '.'));
	unordered_map<int, int> coordinate;
	d.dfs(-1, path, isused, coordinate);
	for (vector<vector<string>>::iterator it = d.result.begin(); it != d.result.end(); it++) {
		cout << "{" << endl;
		for (vector<string>::iterator iit = it->begin(); iit != it->end(); iit++) {
			cout << *iit << endl;
		}
		cout << endl;
		cout << "}" << endl;
	}

	return d.result;
}
//53. ��������
/*
˼·һ����̬�滮���б���Ϊʱ��̫��ʧ����
*/
int Solution::maxSubArray(vector<int>& nums) {
	vector<vector<int>> map(nums.size(), vector<int>(nums.size()));
	//��ʼ��
	int result = nums[0];
	for (int i = 0; i < nums.size() - 1; i++) {
		for (int j = i; j < nums.size(); j++) {
			if (i == j) {
				//��ʼ��
				map[i][j] = nums[i];
			}
			else
				map[i][j] = map[i][j - 1] + nums[j];

			if (map[i][j] > result)
				result = map[i][j];
		}
	}
	return result;
}
/*
	����˼·2��ԭ�����ڼ�¼�����ֵ���п��ܲ��������µ�һ��Ԫ�صģ���������˼·�Ǹ��ٵĶ�̬�滮

	˼·������̬�滮
	˼����̬�滮����1��ʼ˼��
	�����һ��������ô���ֵ��������maxVal=nums[0]
	���������������ô���͵������������� maxVal=max(1.��һ��(maxVal)��2.���ߵڶ���(nums[1])��3.���������ĺ�(maxVal+nums[1]))
	�����������Ӧ���ǣ�max(maxVal+nums[2],nums[2],maxVal)

	[-2,1,-3,4,-1,2,1,-5,4]

*/
int Solution::maxSubArray2(vector<int>& nums) {
	int maxVal = nums[0];
	int temp = 0;
	for (int i = 1; i < nums.size(); i++) {
		temp = max(maxVal + nums[i], nums[i]);
		maxVal = max(temp, maxVal);
	}
	return maxVal;
}
/*
˼·3����̬�滮
	��ʵ��ɶ����ɶ�ͺã����赱ǰ���Ե� i ������β�ġ���������������͡� ��fi
	����ǰ��Ĺ�ϵ��ɶ�أ�
	Ҫô����ǰ���f(i-1)+��ǰ������Ҫô���ǵ��ڵ�ǰ����

	���Ժ��ĵı��ʽ
	f(i)=max(f(i-1)+nums[i],nums[i])

	���������е�fi��ȡ���ֵ�Ϳ���

*/

int Solution::maxSubArray3(vector<int>& nums) {
	int maxVal = nums[0];
	int fi = 0;
	for (int& num : nums) {
		fi = max(num + fi, num);
		maxVal = max(fi, maxVal);
	}
	return maxVal;
}

//54. ��������
/*
	˼·��ģ�����Ѱ�ҵĹ��̼���
*/

vector<int> Solution::spiralOrder(vector<vector<int>>& matrix) {
	int row = matrix.size();
	int col = matrix[0].size();
	int left = 0, right = matrix[0].size() - 1, up = 0, down = matrix.size() - 1;//�߽�
	enum { RIGHT = 0, DOWN, LEFT, UP };//����״̬
	int state = RIGHT;
	vector<int> result(row * col);
	int index = 0;
	int x = 0, y = 0;
	while (left <= right && up <= down) {
		switch (state % 4)
		{
		case RIGHT:
			for (x = up, y = left; y <= right; y++) {
				result[index++] = matrix[x][y];
			}
			up++;
			break;
		case DOWN:
			for (x = up, y = right; x <= down; x++) {
				result[index++] = matrix[x][y];
			}
			right--;

			break;
		case LEFT:
			for (x = down, y = right; y >= left; y--) {
				result[index++] = matrix[x][y];
			}
			down--;
			break;
		case UP:
			for (x = down, y = left; x >= up; x--) {
				result[index++] = matrix[x][y];
			}
			left++;
			break;

		default:
			break;
		}
		state++;
	}

	return result;
}
//55. ��Ծ��Ϸ
/*
˼·��̰���㷨
	��ͷ������һֱ���Լ������������λ��

*/

bool Solution::canJump(vector<int>& nums) {
	int start = 0, end = 0 + nums[0];
	int maxpos = 0;
	for (int i = 0; i < nums.size(); i++) {
		maxpos = max(maxpos, i + nums[i]);
		if (i == end) {
			end = maxpos;
		}
		else if (i > end) {
			return false;
		}
	}
	return true;
}

//56. �ϲ�����
/*
˼·:����
������տ�ͷ��˳�򣬼�¼�µ�ǰ�Ŀ�ʼ�ͽ�������������µĿ�ʼ�Ƚ��������Ǿ��Ƕϵ��ˣ���Ҫ��ǰ����������ȥ���������������

*/
vector<vector<int>> Solution::merge(vector<vector<int>>& intervals) {
	sort(intervals.begin(), intervals.end());
	int begin = intervals[0][0], end = intervals[0][1];
	vector<vector<int>> result;
	for (int i = 0; i < intervals.size(); i++) {
		//�ϵ���
		if (intervals[i][0] > end) {
			result.push_back({ begin,end });
			begin = intervals[i][0];
		}
		//����end
		end = max(end, intervals[i][1]);
	}
	result.push_back({ begin,end });
	return result;
}

//57. ��������
/*
˼·�����²�����������[a,b]���ĸ�λ��

��� a �������ڲ�����ô�µ�������Сֵ��������������Сֵ
��� a �������ⲿ���ҵ����Ӧ��λ��
*/
/*
{{1,2},{3,5},{6,7},{8,10},{12,16}}
*/
vector<vector<int>> Solution::insert(vector<vector<int>>& intervals, vector<int>& newInterval) {
	if (intervals.empty()) {
		return { newInterval };
	}
	int low = intervals[0][0];
	int high = intervals[0][1];
	int a = newInterval[0];
	int b = newInterval[1];
	int highIndex = 0;
	vector<vector<int>> ans;
	for (int i = 0; i < intervals.size(); i++) {
		//Ѱ��a�ڵ�λ��
		//���ڲ�
		if (a >= intervals[i][0] && a <= intervals[i][1]) {
			low = intervals[i][0];
			break;
		}
		//���ⲿ
		else if ((i - 1 >= 0 && a >= intervals[i - 1][1] && a <= intervals[i][0]) ||
			a< intervals[0][0] ||
			a>intervals[intervals.size() - 1][1])
		{
			low = a;
			break;
		}
	}
	//Ѱ��b�ڵ�λ��
	for (int i = 0; i < intervals.size(); i++) {
		//���ڲ�
		if (b >= intervals[i][0] && b <= intervals[i][1]) {
			high = intervals[i][1];
			highIndex = i + 1;
			break;
		}
		//���ⲿ
		else if (b < intervals[0][0]) {
			high = b;
			highIndex = 0;
			break;
		}
		else if (b > intervals[intervals.size() - 1][1]) {
			high = b;
			highIndex = intervals.size();
			break;
		}
		else if ((i - 1 >= 0 && b > intervals[i - 1][1] && b < intervals[i][0]))
		{
			high = b;
			highIndex = i;
			break;
		}
	}
	//�����������
	int i = 0;
	while (i < intervals.size() && intervals[i][0] < low) {
		ans.push_back(intervals[i]);
		i++;
	}
	ans.push_back({ low,high });
	i = highIndex;
	while (i < intervals.size()) {
		ans.push_back(intervals[i]);
		i++;
	}
	return ans;
}
//58. ���һ�����ʵĳ���
int Solution::lengthOfLastWord(string s) {
	if (s.empty()) {
		return 0;
	}
	int ans = 0;
	int end = s.size() - 1;
	while (end >= 0 && s[end] == ' ') end--;
	for (int i = 0; i <= end; i++) {
		if (s[i] == ' ') {
			ans = 0;
			continue;
		}
		ans++;
	}
	return ans;
}
//59. �������� II
/*
˼·��ģ��
*/
vector<vector<int>> Solution::generateMatrix(int n) {
	vector<vector<int>> result(n, vector<int>(n));
	int left = 0, right = n - 1, up = 0, down = n - 1;
	enum { RIGHT = 0, DOWN, LEFT, UP };
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
			for (int i = down, j = left; i >= up; i--) {
				result[i][j] = index++;
			}
			left++;
			break;
		}
		state++;
	}
	return result;
}

//60. ��������
/*
˼·��
	1��ͷ����(n-1)!�����У�	2��ͷ����(n-1)!������......

	X=a[n]*(n-1)!+a[n-1]*(n-2)!+...+a[i]*(i-1)!+...+a[1]*0!
	����ֵ
	000			->		1
	001			->		2

	��kȡ���Ժ�ʣ�� order  ���Ͱ���m+1����������Ž�ȥ(m��1��ʼ)
	���ڵ�k�����У�����λ���ĸ�λ��
*/
string Solution::getPermutation(int n, int k) {
	vector<int> factorial(n);
	factorial[0] = 1;
	for (int i = 1; i < n; i++) {
		factorial[i] = factorial[i - 1] * i;
	}
	string result;
	k--;//���ֵĵ�k���������ȡ�����õ�������һ��
	//��n������ȥ
	vector<bool> isused(n + 1);
	for (int i = 1; i <= n; i++) {
		int order = k / factorial[n - i] + 1;
		//�ҵ� orderС���� j
		for (int j = 1; j <= n; j++) {
			if (isused[j] == false) {
				order--;
				if (order == 0) {
					isused[j] = true;
					result += j + '0';
					break;
				}
			}
		}
		k = k % factorial[n - i];
	}
	cout << result << endl;
	return result;
}
//61. ��ת����

ListNode* rotateRight(ListNode* head, int k) {
	ListNode preHead(0, head);
	ListNode* left = &preHead, * right = &preHead;
	for (int i = 0; i < k; i++) {
		right = right->next;
		if (right->next == NULL) {
			k = k % i;
			break;
		}
	}
	while (right->next != NULL) {
		right = right->next;
		left = left->next;
	}
	//rightָ�������һ���ڵ�
	right->next = preHead.next;
	preHead.next = left->next;
	left->next = NULL;

	return preHead.next;
}
/*
˼·2:���Ҫ��ת�Ļ�����Ϊ��ôҲҪ�ҵ�β�������Կ���ֱ���ҵ�β��������Ȧ�ȣ�Ȼ���ڼ��������Ͱ���Ҫ�ĵط��𿪼���

*/

ListNode* Solution::rotateRight(ListNode* head, int k) {
	if (head == NULL) {
		return NULL;
	}
	ListNode* right = head;
	int len = 1;
	while (right->next != NULL) {
		right = right->next;
		len++;
	}
	right->next = head;
	k %= len;
	for (int i = 0; i < k + 1; i++) {
		right = right->next;
	}
	head = right->next;
	right->next = NULL;
	return head;
}

//62. ��ͬ·��
/*
˼·���ݹ���
*/
class dfs62 {
public:
	int wayNums = 0;
	int m;
	int n;
	dfs62(int mm, int nn) :m(mm), n(nn) {};
	void findway(int x, int y) {
		if (x >= m || y >= n) {
			return;
		}
		else if (x == m - 1 && y == n - 1) {
			wayNums++;
			return;
		}
		//����
		findway(x + 1, y);
		//����
		findway(x, y + 1);
	}
};
int Solution::uniquePaths1(int m, int n) {
	dfs62 d(m, n);
	d.findway(0, 0);
	return d.wayNums;
}
/*
˼·2:��̬�滮

	�������ֻ��һ����Ļ�����ô·������1�����������Ļ���Ҳ��1���ĸ��㣬·����2���ֱ��Ǻ��Ź����ĵĺ����Ź�����
	������� 2*3�Ļ��������½ǵ�Ӧ����	��ߵ�2��·�����+��������õ�1�����

	ֱ�����ߵ���m-1��n-1����·������Ϊ x,
	��ô�ܵ����յ�������֣�һ�ִ��������ģ�һ�ִ��������
	������������������������ͬһ�����⣬ͬ����������Ҳ�ǣ������������Ϊ�յ��·��������
	f(x,y)=f(x-1,y)+f(x,y-1)

*/
int Solution::uniquePaths(int m, int n) {
	vector<vector<int>> map(m, vector<int>(n));
	for (int i = 0; i < m; i++) {
		map[i][0] = 1;
	}
	for (int j = 0; j < n; j++) {
		map[0][j] = 1;
	}
	for (int i = 1; i < m; i++) {
		for (int j = 1; j < n; j++) {
			map[i][j] = map[i - 1][j] + map[i][j - 1];
		}
	}
	return map[m - 1][n - 1];
}

//63. ��ͬ·�� II
int Solution::uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
	if (obstacleGrid[0][0] == 1) {
		return 0;
	}
	int m = obstacleGrid.size();
	int n = obstacleGrid[0].size();
	vector<vector<int>> map(m, vector<int>(n));
	map[0][0] = 1;
	for (int i = 1; i < m; i++) {
		if (obstacleGrid[i][0] == 1) {
			map[i][0] = 0;
		}
		else {
			map[i][0] = map[i - 1][0];
		}
	}
	for (int j = 1; j < n; j++) {
		if (obstacleGrid[0][j] == 1) {
			map[0][j] = 0;
		}
		else {
			map[0][j] = map[0][j - 1];
		}
	}

	for (int i = 1; i < m; i++) {
		for (int j = 1; j < n; j++) {
			if (obstacleGrid[i][j] == 1) {
				map[i][j] = 0;
			}
			else {
				map[i][j] = map[i - 1][j] + map[i][j - 1];
			}
		}
	}
	return map[m - 1][n - 1];
}
/*
�Ľ��������ù��������˼���һ�����
��Ϊ���Ǳ�����ʱ��ÿ�ζ���һ���еı��������Ƕ�������⵱ǰ��״̬ ���õ���Ϣֻ����������У�
������ǿ���ֻ��һ�оͿ���������

*/
int Solution::uniquePathsWithObstacles2(vector<vector<int>>& obstacleGrid) {
	if (obstacleGrid.size() == 0 || obstacleGrid[0].size() == 0) {
		return 0;
	}
	int m = obstacleGrid.size();
	int n = obstacleGrid[0].size();
	//�������
	vector<int> map(n);
	//��ʼ��
	if (obstacleGrid[0][0] == 1) {
		return 0;
	}
	map[0] = 1;
	for (int j = 1; j < n; j++) {
		if (obstacleGrid[0][j] == 0) {
			map[j] = 1;
		}
		else {
			break;
		}
	}
	//�����ǰλ����1�Ļ������ϰ���mapֱ��Ϊ0
	//���û���ϰ�����ôӦ�õ������� �������Ĵ���(֮ǰ����)+�������
	for (int i = 1; i < m; i++) {
		if (obstacleGrid[i][0] == 1) {
			map[0] = 0;
		}
		for (int j = 1; j < n; j++) {
			if (obstacleGrid[i][j] != 1) {
				map[j] += map[j - 1];
			}
			else {
				map[j] = 0;
			}
		}
	}
	return map[n - 1];
}
/*
�Ľ�3�������ù��������˼���һ�����
��Ϊ���Ǳ�����ʱ��ÿ�ζ���һ���еı��������Ƕ�������⵱ǰ��״̬ ���õ���Ϣֻ����������У�
������ǿ���ֻ��һ�оͿ���������

*/
int Solution::uniquePathsWithObstacles3(vector<vector<int>>& obstacleGrid) {
	if (obstacleGrid.size() == 0 || obstacleGrid[0].size() == 0) {
		return 0;
	}
	int m = obstacleGrid.size();
	int n = obstacleGrid[0].size();
	//�������
	vector<int> map(n);
	//��ʼ��
	if (obstacleGrid[0][0] == 1) {
		return 0;
	}
	map[0] = 1;
	for (int i = 0; i < m; i++) {
		for (int j = 0; i < n; j++) {
			//��������Ľ��߽�Ҳ��������
			if (obstacleGrid[i][j] == 1) {
				map[j] = 0;
				continue;
			}
			else if (j - 1 >= 0) {
				map[j] += map[j - 1];
			}
		}
	}
	return map[n - 1];
}
void printVector(vector<int> v) {
	for (vector<int>::iterator it = v.begin(); it < v.end(); it++)
	{
		cout << (*it) << "\t";
	}
}

void printVector2(vector<vector<int>> dp)
{
	for (auto it = dp.begin(); it < dp.end(); it++)
	{
		for (auto vit = it->begin(); vit < it->end(); vit++)
		{
			cout << (*vit) << "\t";
		}
		cout << endl;
	}
	cout << "------------------------" << endl;
}

//64. ��С·����
/*
˼·����̬�滮

�����һ�����������ܺ���С����ô����һ�����һ��Ҳ����С

�赽ĳ�������С�ľ����� fi =min(fi��,fi��)

Ȼ���ͷ���ǣ�(��ʼ��)
һ�����ӵĻ��������ľ�����Сû���������ǵ�һ�����ӵ�ֵ,��һ�к͵�һ�еĻ�Ҳû����������һ��ѡ��
�м��ÿ���㶼�����ߵ� �����ֵ+min(������)
�����Ż��ռ䣬�������鼴��
*/

int Solution::minPathSum(vector<vector<int>>& grid) {
	if (grid.size() == 0 || grid[0].size() == 0) {
		return -1;
	}
	int m = grid.size();
	int n = grid[0].size();
	vector<int> f(n);
	//��ʼ��
	f[0] = grid[0][0];
	for (int j = 1; j < n; j++) {
		f[j] = f[j - 1] + grid[0][j];
	}

	//��̬�滮
	for (int i = 1; i < m; i++) {
		f[0] += grid[i][0];
		for (int j = 1; j < n; j++) {
			f[j] = grid[i][j] + min(f[j - 1], f[j]);
		}
	}
	return f[n - 1];
}
//66. ��һ
vector<int> Solution::plusOne(vector<int>& digits) {
	for (int i = digits.size() - 1; i >= 0; i--) {
		if (digits[i] != 9) {
			digits[i]++;
			return digits;
		}
		digits[i] = 0;
	}
	digits.insert(digits.begin(), 1);
	return digits;
}
//67. ���������
/*
˼·��˫ָ��
*/
//��һ���Ƿ���ֵ
//�ڶ����ǽ�λ��־
pair<char, bool> addSingleBinary(char a, char b, bool carry) {
	return { (a - '0' + b - '0' + carry) % 2 + '0',(a - '0' + b - '0' + carry) / 2 };
}
string Solution::addBinary1(string a, string b) {
	int l1 = a.size() - 1;
	int l2 = b.size() - 1;
	bool carry = 0;
	string result;
	while (l1 >= 0 && l2 >= 0) {
		char temp = addSingleBinary(a[l1], b[l2], carry).first;
		carry = addSingleBinary(a[l1], b[l2], carry).second;
		l1--;
		l2--;
		result += temp;
	}
	while (l1 >= 0) {
		char temp = addSingleBinary(a[l1], '0', carry).first;
		carry = addSingleBinary(a[l1], '0', carry).second;
		l1--;
		result += temp;
	}
	while (l2 >= 0) {
		char temp = addSingleBinary(b[l2], '0', carry).first;
		carry = addSingleBinary(b[l2], '0', carry).second;
		l2--;
		result += temp;
	}
	if (carry == 1) {
		result += '1';
	}

	reverse(result.begin(), result.end());
	return result;
}

string Solution::addBinary2(string a, string b) {
	int l1 = a.size() - 1;
	int l2 = b.size() - 1;
	int carry = 0;
	string result;
	while (l1 >= 0 || l2 >= 0) {
		if (l1 >= 0) carry += a[l1--] - '0';
		if (l2 >= 0) carry += b[l2--] - '0';
		result += carry % 2 + '0';
		carry /= 2;
	}
	if (carry == 1) {
		result += '1';
	}
	reverse(result.begin(), result.end());

	return result;
}
//68. �ı����Ҷ���
/*
˼·��ģ��

ÿ��������������¼��ǰ����ȥ�� ������ �� ����
���������������һ�����ʵ�ʱ�򣬳��ȳ�����һ���ˣ�ֹͣ���ˡ�
����һ��ÿ�����ʵĿո���=(���еĳ���-����ȥ�ĵ��ʳ���)/������

ѭ��������̣�ֱ��������ǰ������ǽ�β�ˣ�ֱ�Ӱ�ǰ���һ��������ȥ

*/
// "This", "is", "an", "example", "of", "text", "justification."
vector<string> Solution::fullJustify(vector<string>& words, int maxWidth) {
	//�����������ʱ�һ�л���

	//�������
	//ÿ�����ʿ�
	vector<string> result;
	//words����Ϊ��
	if (words.empty()) {
		return result;
	}
	int begin = 0;
	int len = 0;//��¼������ȥ�ĵ����ܳ���
	deque<string> q;//��¼����ȥ�ĵ���
	for (int i = 0; i < words.size(); i++) {
		if (len + words[i].size() <= maxWidth) {
			//������������
			q.push_back(words[i]);
			len += words[i].size() + 1;//��������� word �Ϳո�ϲ�������һ������
		}
		else {
			i--;
			//������ʲ������ˣ���ǰ��Ķ��Ž�ȥ
			len--;//�����һ���Ŀո�ȥ��
			int num = q.size();//�������ĵ����ܸ���

			//����ո����
			string temp = "";
			if (num == 1) {
				//ֻ��һ������
				int num = q.size();//�������ĵ����ܸ���
				//����ȥ��󲻴��ո��
				temp += q.front() + string(maxWidth - len, ' ');
				q.pop_front();
			}
			else {
				//������ʷ���ո�
				int aveBlank = (maxWidth - len) / (num - 1);//ƽ��ÿ�����ʷֵ��Ŀո���
				int remain = (maxWidth - len) % (num - 1);//ʣ�µĿո���
				for (int j = 0; j < num - 1; j++) {
					if (remain > 0) {
						temp += q.front() + string(aveBlank + 2, ' ');
						remain--;
					}
					else {
						temp += q.front() + string(aveBlank + 1, ' ');
					}
					q.pop_front();//ɾ����һ��
				}
				temp += q.front();
				q.pop_front();
			}
			result.push_back(temp);
			len = 0;
		}
	}
	//������ʣ�µĶ��������һ�м���
	if (!q.empty()) {
		len--;
		int num = q.size();//�������ĵ����ܸ���
		string temp = "";
		//����ȥǰ����ո��
		for (int j = 0; j < num - 1; j++) {
			temp += q.front() + " ";
			q.pop_front();
		}
		//����ȥ��󲻴��ո��
		temp += q.front() + string(maxWidth - len, ' ');
		q.pop_front();
		result.push_back(temp);
	}
	for (string s : result) {
		cout << s << endl;
	}

	return result;
}
/*
�Ľ���
ǰ�湦�ܻ����������ظ�Ҳ�ܴ�
�������ֿ���
	���������ñ߼�����ֱ����һ����������λ��
	��������ָ�������귶Χ��������ȥ��������ָ��Ҫ�����ո񣬲������õ������أ�
	��������һ�����ǾͰ�����һ�׹���ȥ��
*/
//���ܣ�����
/*
* words:
* start:
* end:
* maxWidth:
* len:ÿ�����ʼ��˿ո��Ժ�Ž����ĳ���(���һ�����ӿո�)
* isEnd:��β��
*/

string pushnum(vector<string>& words, int start, int end, int maxWidth, int len, bool isEnd) {
	string ans = "";
	//���ֻ��һ���ַ�Ҫ����
	if (start == end) {
		return words[start] + string(maxWidth - len, ' ');
	}
	//��������һ��
	if (isEnd) {
		for (int i = start; i < end; i++) {
			ans += words[i] + " ";
		}
		ans += words[end];
		return ans + string(maxWidth - len, ' ');
	}
	//���
	int spaceNum = maxWidth - len;//Ҫ����Ŀո���
	int aveSpace = spaceNum / (end - start);//ƽ��ÿ���ַ�������ŵĿո�
	int reserve = spaceNum % (end - start);//���µ���Ҫ����Ŀո�
	for (int i = start; i < end; i++) {
		if (reserve > 0) {
			ans += words[i] + string(aveSpace + 2, ' ');
			reserve--;
		}
		else {
			ans += words[i] + string(aveSpace + 1, ' ');
		}
	}
	ans += words[end];
	return ans;
}

vector<string> Solution::fullJustify2(vector<string>& words, int maxWidth) {
	int sysmbleCount = 0;//���ż���
	int start = 0;
	vector<string> ans;
	for (int i = 0; i < words.size() - 1; i++) {
		sysmbleCount += words[i].size() + 1;//�����������ȥ
		//�������һ������ȥ���Ͼͳ��ˣ���ô����һ������һ���������
		if (sysmbleCount + words[i + 1].size() > maxWidth) {
			ans.push_back(pushnum(words, start, i, maxWidth, sysmbleCount - 1, 0));
			start = i + 1;
			sysmbleCount = 0;
		}
	}
	sysmbleCount += words.back().size() + 1;//������������ȥ
	ans.push_back(pushnum(words, start, words.size() - 1, maxWidth, sysmbleCount - 1, 1));

	for (string s : ans) {
		cout << s << endl;
	}
	return ans;
}
//69. x ��ƽ����
/*
	��⣺f(x)=x2-c�����
	�������ʽ�� x=x- (x2-c)/2x =(x+C)/2x
�����Ǵ�������
	���ԣ�����û��Ҫ����Ϊ�ж���������ȷ������������ظ��Ļ�����������
��ȷ�뷨�����Եģ�x^2������������˵���һ�£��ұ�һ�£����Ҳ�����������ǽ������ж��������Բ���f(x)=x*x-c<0�������б�
���Ҹ�����ȡ�������ʣ�һ���ǿ���ת�Ƶ��������

�ö��ַ���ȡ��
	����м��ƽ��С��x�����а�����
	����м��ƽ������x�����������
*/

int Solution::mySqrt(int c) {
	int x = c;

	while (x * x > c) {
		x = (x * x + c) / (x * 2);
	}

	return x;
}
//70. ��¥��
/*
˼·��
1.�ݹ飬ÿ��ѡ���ȥ1���߼�ȥ2�����ռ�������0�ͷ���
2.��̬�滮��������ҵ�n��·�����ٶ����һ�ף�ͬ����+1�������ף�+2�Դˣ���ô�����й��ɣ�,
	���Ե���n�׵����������fn=fn-1+fn-2
*/
int Solution::climbStairs(int n) {
	vector<int> path(3);
	path[0] = 0;
	path[1] = 1;
	path[2] = 2;
	for (int i = 3; i <= n; i++) {
		path[i%3] = path[(i - 1)%3] + path[(i - 2)%3];
	}
	return path[n % 3];

}
//71. ��·��
/*
��Ҫ��������⣺
	1.β����/��ȥ��
	2.��..�ģ���ǰ���ɾ����
	3.��Ŀ¼�Ĵ� ..���Ǹ�Ŀ¼
	4.˫б�ߵ�ȥ��һ��
	5. . ��ʾ��ǰ��·����ûɶ�ã�ȥ��

˼·���������,��ͷ��β�ı����������
	����'/'�ͽ�����ǰ��¼�Ĵ�
	��鵱ǰ�Ĵ�������
	�������'.' ���� '..'ֱ������ջ
	�����'.' ����ᣬ���ſ�������
	�����'..',����һ��

	while(s[i]!='/')
		s+=s[i];
	if(not '.' or '..') push(/+s)
	else if(s=='.')
	else if(s=="..")  if(not empty)  pop(s)		
��������
	/...  ,
	/a..	,	/..a
*/


string Solution::simplifyPath(string path) {
	if (path.back() != '/') {
		path += '/';
	}
	deque<string> q;//ֻ�����ε�·��
	string s = "";
	for (int i = 0; i < path.size();i++) {
		if (path[i] != '/') {
			s += path[i];
		}
		else {
			if (s == "..") {
				if (!q.empty())
					q.pop_back();
			}
			else if (s == ".") {
				s = "";
				continue;
			}
			else {
				if (!s.empty())
					q.push_back(s);
			}
			s = "";
		}
	}
	string ans;
	if (q.empty()) {
		return "/";
	}
	else {
		for (string s : q) {
			ans += "/" + s;
		}
		return ans;
	}


}
//�Ľ�

class Solution71 {
public:
	deque<string> q;//ֻ�����ε�·��
	void dealSinglePath(string name) {
		if (name.empty()||name == ".") {
			return;
		}
		else if (name == "..") {
			if (!q.empty())
				q.pop_back();
		}
		else {
			q.push_back(name);
		}
	}
	string generateResult() {
		if (q.size()==0) {
			return "/";
		}
		string ans = "";
		for (int i = 0; i < q.size(); i++) {
			ans += "/" + q[i];
		}
		return ans;
	}
	string simplifyPath(string path) {
		string s = "";
		for (char c : path) {
			if (c == '/') {
				dealSinglePath(s);
				s = "";
			}
			else {
				s += c;
			}
		}
		dealSinglePath(s);
		return generateResult();
	}

};

string Solution::simplifyPath2(string path) {

	Solution71 s;

	return s.simplifyPath(path);
}
//72. �༭����
int Solution::minDistance(string word1, string word2) {
	int m = word1.size();
	int n = word2.size();
	if (n * m == 0) return n + m;
	vector<vector<int>> dp(m + 1, vector<int>(n + 1));
	for (int i = 0; i < m + 1; i++) {
		dp[i][0] = i;
	}
	for (int j = 0; j < n+1; j++) {
		dp[0][j] = j;
	}
	for (int i = 1; i < m + 1; i++) {
		for (int j = 1; j < n + 1; j++) {
			if (word1[i - 1] == word2[j - 1]) {
				dp[i][j] = min(dp[i - 1][j - 1], min(dp[i - 1][j] + 1, dp[i][j - 1] + 1));
			}
			else {
				dp[i][j] = min(dp[i - 1][j - 1]+1, min(dp[i - 1][j] + 1, dp[i][j - 1] + 1));
			}
		}
	}
	printVector2(dp);
	return dp[m][n];
}
int Solution::minDistance2(string word1, string word2) {
	int n = word1.length();
	int m = word2.length();

	// ��һ���ַ���Ϊ�մ�
	if (n * m == 0) return n + m;

	// DP ����
	vector<vector<int>> D(n + 1, vector<int>(m + 1));

	// �߽�״̬��ʼ��
	for (int i = 0; i < n + 1; i++) {
		D[i][0] = i;
	}
	for (int j = 0; j < m + 1; j++) {
		D[0][j] = j;
	}

	// �������� DP ֵ
	for (int i = 1; i < n + 1; i++) {
		for (int j = 1; j < m + 1; j++) {
			int left = D[i - 1][j] + 1;
			int down = D[i][j - 1] + 1;
			int left_down = D[i - 1][j - 1];
			if (word1[i - 1] != word2[j - 1]) left_down += 1;
			D[i][j] = min(left, min(down, left_down));

		}
	}
	printVector2(D);
	return D[n][m];
}


//73. ��������
/*
˼·����ͷ��β�������ҵ����㣬����һ�к���һ�п�ͷ�����0����һ�к͵�һ�����㣩


*/
void makeRowZero(vector<vector<int>>& matrix,int row) {

	for (int j = 0; j < matrix[0].size(); j++) {
		matrix[row][j] = 0;
	}

}
void makeColZero(vector<vector<int>>& matrix, int col) {
	for (int i = 0; i < matrix.size(); i++) {
		matrix[i][col] = 0;
	}
}
void Solution::setZeroes(vector<vector<int>>& matrix) {
	bool colZero = 0;
	bool rowZero = 0;
	int row = matrix.size();
	int col = matrix[0].size();
	//��һ��
	for (int i = 0; i < row; i++) {
		if (matrix[i][0] == 0) {
			colZero = 1;
			break;
		}
	}
	//��һ��
	for (int i = 0; i < col; i++) {
		if (matrix[0][i] == 0) {
			rowZero = 1;
			break;
		}
	}
	//����
	for (int i = 1; i < row; i++) {
		for (int j = 1; j < col; j++) {
			if (matrix[i][j] == 0) {
				matrix[i][0] = 0;
				matrix[0][j] = 0;
			}
		}		
	}
	for (int i = 1; i < row; i++) {
		if (matrix[i][0] == 0) {
			makeRowZero(matrix, i);
		}
	}
	for (int j = 1; j < col; j++) {
		if (matrix[0][j] == 0) {
			makeColZero(matrix, j);
		}
	}
	//����Ū
	if (rowZero == 1) {
		makeRowZero(matrix, 0);
	}
	if (colZero == 1) {
		makeColZero(matrix, 0);
	}
}
//74. ������ά����
/*
˼·�����ֲ���	
*/
bool Solution::searchMatrix(vector<vector<int>>& matrix, int target) {
	int m = matrix.size();
	int n = matrix[0].size();
	if (m * n == 0) {
		return false;
	}

	int left = 0;
	int right = (m - 1) * n + n - 1;
	int mid = 0;
	while (left <= right) {
		mid = (left + right) / 2;
		if (matrix[mid / n][mid % n] == target) {
			return true;
		}
		else if (target < matrix[mid / n][mid % n]) {
			right = mid - 1;
		}
		else {
			left = mid + 1;
		}
	}
	return false;

}
//75. ��ɫ����
/*
	˼·��
		�򵥵�˼·������������һ��0  2�ĸ�����Ȼ��1=ʣ�µģ������¸�ֵһ�鼴��
		�������Ҫ��һ������Ļ������ǿ�����˫ָ���滻����Ӧ�Ŀռ����� 
*/

void Solution::sortColors(vector<int>& nums) {
	int m = nums.size();
	int p0 = 0;
	int p2 = nums.size() - 1;
	for (int i = 0; i < p2; i++) {
		if (nums[i] == 2) {
			//�п���һֱ����������2,Ҳ�п��ܶ�������0�����������0���Ǻϲ���һ��ȥ��һ����֤
			while (p2 > i && nums[p2] == 2) {
				p2--;
			}
			swap(nums[i], nums[p2]);
		}
		if (nums[i] == 0) {
			swap(nums[i], nums[p0++]);
		}
	}
}
//76. ��С�����Ӵ�
/*
����˼·��
	��ͷ��β��һ�������������Ŀ��Ĵ��������Ļ�ֱ�ӷ��ز����ˣ����Ļ����ǾͰ���ߵ�ɾһ�������Ծ�ɾ����ֱ��������Ϊֹ��Ȼ���ɾ�ұߵ�...
	��̫�Ծ����п�����ɾ����ʱ�򣬰����Ž�������ɾ����
	�ǣ�
	���ɾһ�����ұ�ɾһ����	��ʼ�ݹ������ŵģ�
	���ǵݹ�Ļ���
	��������ص������
	������ ������   ��������  ������Ľ����һ���İ�
	�ǣ�
	��̬�滮��
	��Ȼ�������ɾ���ұ�ɾ���ַ������ǾͶ�ά�����ʾ�������е�������ϡ�

	�ö�ά�����ʾ�Ļ���ÿһλ��ʾ���ɾ������һλ�ܲ��ܰ�����Ҫ�����Щ�ַ�
	���Է��֣������ߵĺ��ϱߵĲ�������Ļ�����ô����Ŀ϶�����������
	ת�Ʒ��� dp[i][j]=dp[i-1][j]&&dp[i][j-1]&&(del cur is ok)
	������ô����ɾ��������ַ��ܲ��ܽ���ƥ���أ�
	1.����ƥ�䣬ֱ�Ӵ�ͷ��β������
	2.������������У����񣿺�֮ǰ�ļ���ظ��˶��Ϊ�Ͷ�������һ���ַ�������������ô�ͷ���һ��

	�Ǵ������ҹ����У���Ū��map����¼�أ�
	զ���˼һ������ڿ�ӽ��ˣ����������������������ϣ���ͷ���ܵ�β��Ȼ����ͷ����

	�����ܵ����ռ�ȫ���ٻ�������������������ָ��᲻��Ч�ʸߵ㣿

˼·����������
	��ͷ��β����s
	����ÿ���ַ������������Ҫƥ��Ĵ�����������ַ�����ô��ʼ��¼��

	��¼�ķ����������������ǰ�ַ������ƥ�䴮���棬��ô��Ӧ�Ĺ�ϣ������ּ�һ
	�����Ӧ������ȫ�ˣ�	cnt��ori��Ӧ�Ĵ�Сһ����
	��ô��ʼ��Сָ�뷶Χ��	void short
	Ѱ����Сֵ,				
	������Сֵ�Ժ�,������¼
	�ټ���������
*/
//���������Ժ�ĳ���
void Solution76::shrinkIndex(string& s, int& pleft, const int& pright) {
	int i = pleft;
	for (i = pleft; i <= pright; i++) {
		if (cnt.find(s[i])!=cnt.end()) {
			cnt[s[i]]--;
	
//-------------------������ĵ�---------------------------------------	


			if (cnt[s[i]] <ori[s[i]]) {
				cnt.erase(s[i]);
				pleft = i + 1;
				break;
			}
		}
	}
	if (pright - i + 1 < len) {
		len = pright - i + 1;
		sleft = pleft - 1;
		sright = pright;
	}
}

string Solution76::minWindow(string s, string t) {
	if (s.empty()||t.size()>s.size()) {
		return "";
	}
	if (t.empty()) {
		return s;
	}

	int m = s.size();
	int pleft = 0, pright = 0;
	for (char c : t) {
		ori[c]++;
	}
	//���ҵ���ͷ
	while (pleft < m && ori.find(s[pleft]) == ori.end()) {
		pleft++;
	}
	pright = pleft;
	while (pright < m && pleft < m) {
		if (pright < m && ori.find(s[pright]) != ori.end()) {
			//��ʼ�ң����˵�ҵ��˶�Ӧ���ַ�
			cnt[s[pright]]++;
			if (cnt.size() == ori.size()) {
				shrinkIndex(s, pleft, pright);
			}
		}
		pright++;
	}
	return s.substr(sleft,len);	
}
