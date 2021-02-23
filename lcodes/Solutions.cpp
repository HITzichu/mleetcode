#include "Solutions.h"

//4.寻找两个正序数组的中位数
int getKthElement(const vector<int>& nums1, const vector<int>& nums2, int k)
{
	//k为需要比较的个数
	int m = nums1.size();
	int n = nums2.size();
	int index1 = 0, index2 = 0;
	while (true)
	{
		if (index1 == m)
		{
			return nums2[index2 + k - 1];		//寻找第k小的
		}
		if (index2 == n)
		{
			return nums1[index1 + k - 1];
		}
		if (k == 1)
		{
			return min(nums1[index1], nums2[index2]);
		}

		int newindex1 = min(index1 - 1 + k / 2, m - 1);	//index-1:前面的坐标，k/2 要比较的数个数
		int newindex2 = min(index2 - 1 + k / 2, n - 1);
		//哪个小修改哪个
		if (nums1[newindex1] <= nums2[newindex2])
		{
			//减去排除掉的数字个数，k为剩下的需要比较的个数
			k = k - (newindex1 - index1 + 1);
			//修改坐标
			index1 = newindex1 + 1;
		}
		else
		{
			//减去排除掉的数字个数，k为剩下的需要比较的个数
			k = k - (newindex2 - index2 + 1);
			//修改坐标
			index2 = newindex2 + 1;
		}
	}
}
double Solution::findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
	int m = nums1.size();
	int n = nums2.size();
	//偶数
	if ((m + n) % 2 == 0)
		return (getKthElement(nums1, nums2, (m + n) / 2) + getKthElement(nums1, nums2, (m + n) / 2 + 1)) / 2.0;
	else
		return getKthElement(nums1, nums2, (m + n + 1) / 2);
}
//5.最长回文子串

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

	//初始化
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
				dp[i][j] = (dp[i + 1][j - 1] && (s[i] == s[j]));//核心一步
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

//6.Z字型变换
string Solution::convertZ(string s, int numRows)
{
	if (s.size() < numRows || numRows == 1)
	{
		return s;
	}

	bool flag = false;
	int curRow = -1;//记录当前的行
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
		int ge = x % 10;//求个位数

		//越界
		if ((rev > INT_MAX / 10)
			|| (rev == INT_MAX / 10 && ge > INT_MAX % 10)
			|| (rev < INT_MIN / 10)
			|| (rev == INT_MIN / 10 && ge < INT_MIN % 10))
		{
			return 0;
		}

		rev = rev * 10 + ge;//添加到最后
		x = x / 10;
	}

	return rev;
}

/*
	table行表示当前的状态
	列表示遇见的字符
	对应的元素为将要转化的状态
*/
class Automaton
{
	//实现talble的定义
	//table的
	string state = "start";
	unordered_map<string, vector<string>> table =
	{
		{"start",{"start", "signed", "in_number", "end"}},
		{"signed",{"end", "end", "in_number", "end"}},
		{"in_number", {"end", "end", "in_number", "end"}},
		{"end", {"end", "end", "end", "end"}}
	};

	//定义遇到对应的符号的时候需要去查找到的对应的vector 的列元素
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
		//记录符号

		for (char c : str)
		{
			//更新当前状态
			state = table[state][getcol(c)];
			//进入了符号状态，查看当前符号
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
				//越界判断
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

//9.回文数

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
		//计算当前的面积
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

//13.罗马数字转整数

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
		//对于每个元素（*it）遍历每个字符
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

//15.三数之和
vector<vector<int>> Solution::threeSum(vector<int>& nums)
{
	//先排序
	sort(nums.begin(), nums.end());

	vector<vector<int>> result;
	//中间的依次看和两边的数加起来值是不是零
	//小于零的话左边的右移，大于零的话右边的左移 等于零塞进去	->变大   0.0   <-变小

	if (nums.size() < 3)
	{
		return result;
	}

	//-1,-1,0,1
	for (int i = 0; i < nums.size() - 2; i++)	//遍历中间的数
	{
		if (i > 0 && nums[i] == nums[i - 1])
			continue;

		int left = i + 1, right = nums.size() - 1;	//双指针
		while (left < right)				//越界条件
		{
			//小于零的话左边的右移，大于零的话右边的左移 等于零塞进去	->变大   0.0   <-变小
			int sum = nums[i] + nums[left] + nums[right];
			if (sum == 0)
			{
				result.push_back({ nums[i] , nums[left] , nums[right] });
				//跳过重复
				while (left < right && nums[++left] == nums[left - 1]);
				while (left < right && nums[--right] == nums[right + 1]);
			}
			else if (sum > 0)
			{
				//去除重复的,	如果这次的left值和right值都还是上一次的值，说明重复了
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

//因为我们优化的目标也就是使得目标函数最接近，这个用绝对值来刻画是比较合理的,我们期望它是趋于0的
//思路：给定了一个数t，寻找两个数m,n，使得目标函数f=|t-(m+n+k)|最小
//遍历方法：双指针，每次遍历只能去寻找一个比原本的数绝对值小两个值，如果变大了，说明前一个即为最小值
//双指针我们使用的时候，向量是排好序的。那么一般情况下我们认为，left+1即m+n变大，right-1即m+n变小,我们控制每次的结果，如果小于0变大，大于零变小，那么结果就趋向于0.这个和目标函数的最小值是一致的
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
			//变化双指针
			if ((target - sum) == 0)
			{
				return  sum;
			}
			else if ((target - sum) < 0)	//结果小于0，sum要变小结果才能变大
			{
				//去除重复的,	如果这次的left值和right值都还是上一次的值，说明重复了	--right
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
		//停止条件
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

//17. 电话号码的字母组合
vector<string> Solution::letterCombinations(string digits)
{
	//数字表
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

//18. 四数之和
vector<vector<int>> Solution::fourSum(vector<int>& nums, int target)
{
	//a,b,c,d两个双指针，如果找不到就移动右指针，都找不到就重新移动左指针
	//好处在于，如果我们内层的最大值和最小值都是大于零（或者小于零），就可以直接判断，然后跳到下一层循环了
	//外层指针
	int left1 = 0, right1 = nums.size() - 1;
	//内层指针
	int left = 1, right = nums.size() - 2;
	int sum = 0;
	vector<vector<int>> result;
	if (nums.size() < 4)
	{
		return result;
	}
	sort(nums.begin(), nums.end());
	//遍历a
	// -1, 0, 0, 1,2
	while (left1 < nums.size() - 3)
	{
		//外层右指针初始化
		right1 = nums.size() - 1;
		//外层右指针变化

		//排除特殊情况
		int maxVal = target - (nums[left1] + nums[right1 - 2] + nums[right1 - 1] + nums[right1]);
		int minVal = target - (nums[left1] + nums[left1 + 1] + nums[left1 + 2] + nums[right1]);
		//最大值和最小值都大于零,说明整体是大于零的，直接外层左指针+1
		if (maxVal > 0 && minVal > 0)
		{
			while (nums[++left1] == nums[left1 - 1] && left1 < nums.size() - 3);
		}
		//最大值和最小值都小于零,说明整体是小于零的，直接外层右指针-1
		if (maxVal < 0 && minVal < 0)
		{
			while (nums[--right1] == nums[right1 + 1] && right1 - left1 >= 3);
		}

		while (right1 - left1 >= 3)
		{
			left = left1 + 1;
			right = right1 - 1;

			//内层双指针循环
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
			//外层指针往内移动
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
	//先间隔了n个
	for (int i = 0; i < n; i++)
	{
		right = right->next;
	}
	//同时往后遍历
	while (right->next != NULL)
	{
		left = left->next;
		right = right->next;
	}
	//left指向的倒数n+1个节点
	left->next = left->next->next;

	return head;
}
//20. 有效的括号
bool Solution::isValid(string s)
{
	stack<char> sta;
	//构造一个查找表
	//key为右端元素，val为左端元素
	unordered_map<char, char> pairs = {
	{')', '('},
	{']', '['},
	{'}', '{'}
	};
	for (char c : s)
	{
		//用栈操作，如果遇到与之匹配的右括号，那么一定是匹配栈顶的元素，否则匹配失败
		if (pairs.count(c)	//匹配到了右端元素
			&& sta.empty())	//不匹配最近的左括号
		{
			return false;
		}
		else if (pairs.count(c)	//匹配到了右端元素
			&& sta.top() != pairs[c])	//不匹配最近的左括号

		{
			return false;
		}
		else if (pairs.count(c)	//匹配到了右端元素
			&& sta.top() == pairs[c])	//匹配到最近的左括号
		{
			sta.pop();
			continue;
		}
		else //左括号
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

//21. 合并两个有序链表
//l1:[1,3,4]
//l2:[0,2,4]
ListNode* Solution::mergeTwoLists(ListNode* l1, ListNode* l2)
{
	ListNode* preHead = new ListNode(0);
	ListNode* curNode = preHead;

	while (l1 != NULL && l2 != NULL)
	{
		//比较,将较小的拿出来给到当前节点后面，拆线必须保存被拆的信息
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
//22. 括号生成

/*
* 树遍历：
* 	1.添加左括号
*	2.添加右括号，添加条件：左括号数目大于右括号
*	结束条件：n层
*
*/
//深度优先遍历
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
		//添加左括号
		if (left > 0)
		{
			dfs(curStr + "(", left - 1, right);
		}
		//添加右括号
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

//23.合并K个升序链表

/*
* 分治合并：k个合并成 k/2,然后再合并成k/4
*
* 参数：两个待合并的链表
* 返回值：合并一次后得到链表头
* 做法，两两合并
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

//23. 合并K个升序链表
/*
* 递归写合并两个有序链表
*
* 做法：
*	比较两个链表头部的大小，让当前指针指向较小的节点
*
*	参数：两个头结点
*	返回值：较小的头结点
*	停止条件：节点为空
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
//24. 两两交换链表中的节点
//3条线需要改 前面的，中间的，后面的
ListNode* swap2(ListNode* head, ListNode* preHead)
{
	//结束条件
	if (head == NULL)
	{
		return NULL;
		cout << "偶数个" << endl;
	}
	if (head->next == NULL)
	{
		return head;
		cout << "奇数个" << endl;
	}
	ListNode* left = head;
	ListNode* right = left->next;
	//修改三条线的连接关系
	left->next = right->next;	//交换两个节点的信息
	right->next = left;
	//前面节点指向后面
	preHead->next = right;

	swap2(left->next, left);
	return right;
}

ListNode* Solution::swapPairs(ListNode* head)
{
	ListNode* left = head;
	ListNode* right = left->next;
	//修改三条线的连接关系
	left->next = swapPairs(right->next);	//交换两个节点的信息
	right->next = left;
	//前面节点指向后面
	return right;
}
//反转一组链表
pair<ListNode*, ListNode*> reverseK(ListNode* head, ListNode* tail)
{
	if (tail == NULL)
	{
		return { head,tail };
	}

	ListNode* cur = head;		//当前节点
	ListNode* pre = tail->next;//pre用于指向当前应该指向的节点
	while (pre != tail)
	{
		ListNode* temp = cur->next;
		cur->next = pre;
		pre = cur;
		cur = temp;
	}
	return{ tail, head }; //返回新的头和尾
}

//25. K 个一组翻转链表
//	做法：依次翻转k个，上一次翻转的尾部指向新一次的头部
ListNode* Solution::reverseKGroup(ListNode* head, int k)
{
	ListNode pre(0, head);
	ListNode* right = &pre;
	ListNode* left = &pre;

	bool flag = true;
	while (flag == true)
	{
		right = left;
		//往后找k个
		for (int i = 0; i < k; i++)
		{
			if (right == NULL)
			{
				flag = false;
				break;
			}
			right = right->next;
		}
		//找得到
		if (flag == true)
		{
			pair<ListNode*, ListNode*> result = reverseK(left->next, right);
			left->next = result.first;
			left = result.second;
		}
	}
	return pre.next;
}
//26. 删除排序数组中的重复项
int removeDuplicates(vector<int>& nums) {
	if (nums.empty())
	{
		return 0;
	}

	//i:慢指针，j快指针
	int i = 1, j = 1;

	while (j < nums.size())
	{
		//找到相同的j的最后一位
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

//27. 移除元素
//i j为两个指针 ，i代表应该放进的非 val 的位置，j为快指针
//j不是val,那么将其放进i位置
//j是val,那么将其跳过
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

//28. 实现 strStr()

//数学前提：集合论
//一个串的两个前后缀子串要么是最长相同前后缀，要么不是
// j前缀末尾
// i后缀末尾，
//  [] [] [] [] [] [] [] []
//	   j     i
//next表示的是1.前面子串的最长相同前后缀 	2.如果发生不匹配，那么j需要回退到的位置
//当前已经满足的条件是：已经有符合条件的 前后两个串，如果下一个字符不匹配的话，应当缩小匹配范围，到可能成功的最近的一个串的位置，
//这个位置要符合的条件在于，前面的字符和开头相同的字符尽可能多
//这个意义恰好就是这个串的最长相同前后缀
//所以我们需要将j其回退到next[j]指向的即可

//明确next指向的数据应为当前指向的前后缀的前面字符串的 最长相同前后缀

vector<int> KMPNext(string pat)
{
	vector<int> next(pat.size() + 1);	//i++完了以后会到了pat.size()
	int j = -1;//前缀末尾
	int i = 0;//后缀末尾
	next[0] = -1;
	while (i < pat.size())
	{
		if (j == -1		//如果不加这个条件的话，可能不相等的就不会填充数组，这样可以保证每个数组都能有数字填充
			|| pat[j] == pat[i])
		{
			i++;
			j++;
			next[i] = j;
		}
		else
		{
			j = next[j];//跳到它子串的最长相同前后缀的位置
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
	vector<int> next = KMPNext(needle);//计算出next子串
	int i = 0, j = 0;//指向模式串的指针
	while (i < haystack.size())
	{
		if (j == -1		//指针指向的是第一个就不匹配
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

//求相同前后缀
vector<int> samPreSuf(string s)
{
	int sublen = 1;

	vector<int> result;
	while (sublen < s.size())
	{
		bool flag = true;
		int  sufstart = s.size() - sublen;//两个开始位置
		//检查之后的每个字符
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

//29. 两数相除
//dividend被除数，divisor除数
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
		result += result;//result=2*result		存储找到了多少个divisor
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

//30. 串联所有单词的子串

//做法：将待匹配的单词都记录到哈希表中，哈希表的value值为单词重复的次数
//维护一个长度为 所有单词的和 的滑动窗口，每次进出都是一个 单词 的量->错误，应该是一个字符的量，匹配的话不一定恰好在字符长度整数倍的位置
//从头开始依次检查窗口中每个单词是否符合要求，如果不符合那么就将窗口往后移动一个字符
//检查的方法如下：

//如果检查到尾，发现每个单词都存在，那么符合条件
//不符合条件停止检查的条件有
//	1.没找到当前的单词
//	2.找到了这个单词，但是数量多了

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
	//初始化计数表
	for (auto& w : words)
		wordcnt[w]++;
	int singleLen = words[0].size();//每个单词的长度
	int left = 0, right = left;

	//left 窗口左边界
	for (int left = 0; left + singleLen * words.size() <= s.size(); left++)
	{
		unordered_map<string, int> curWindow;//记录当前窗口的单词和数
		//检查当前窗口，从头至尾遍历
		int count = 0;
		right = left;
		string temp = "";
		for (count = 0; count < words.size(); count++)
		{
			temp = s.substr(right, singleLen);//记录待比较的单词
			if (wordcnt.find(temp) == wordcnt.end() || curWindow.count(temp) > wordcnt.count(temp))	//没找着 或 找多了
			{
				break;
			}
			//找到了
			curWindow[temp]++;
			right += singleLen;
		}
		if (curWindow == wordcnt)
		{
			result.push_back(left);
		}
		if (curWindow.count(temp) > wordcnt.count(temp))//如果是因为多匹配的导致了匹配失败
		{
			curWindow.erase(s.substr(left, singleLen));//那可以将左边的去掉再看看
		}
		else
			curWindow.clear();
	}
	return result;
}

//31. 下一个排列
/*
1. 从后往前找到第一个升序拐点[i,j]，
2. 然后从j开始的数字中找出一个比i点大且靠后的数字进行(交换到前面以后就是最小的)兑换，
3. 然后将从j开始的数字进行逆序
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
	//已经是最大的排列
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
//32. 最长有效括号
/*思路：动态规划
*
初始思路：
从头往后遍历过程中记录每个串的以当前字符为结尾的最长后缀子串的长度 记录在dp[]中
如果有新的括号加入
		1.如果是左括号，那肯定以它为结尾的最长后缀子串长度是0
		2.如果是右括号，他是可以往前消左括号的(像俄罗斯方块一样)，但是也只能消一个，

			如果有两个左括号的话(( )，那不用说，当前的后缀子串值肯定就是2，前面的不用看了，所以我们只需要看前面两个字符的值是不是左括号，如果是两个左括号，那么当前的值就是2
			如果是 )() 这种 dp那么应该是，上一个右括号的dp[]值+2
			如果是 ())
			如果是 )))		dp那么应该是，上一个右括号的dp[]值+自己匹配左括号的结果
									if s(i-dp[i-1])is ')'=		dp[i-1]+2
									else dp[i]=0

但是证明是错误的，反例()(())
*/

//第二种思路：遍历字符串，记录有效的左括号数和右括号数
//当字符为左括号时，left++，
//当为右括号的时候,如果左括号大于等于右括号，那么正常right++，而且计数结果也跟着增加。如果左括号数量少于右括号，那么这个右括号是断档的，因此当前的计数清零。
//然后在从尾到头再来一遍
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
				//越界
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
//33. 搜索旋转排序数组

/*
	原理：1.旋转排序数组二分以后肯定有一半是有序的
		  2.在有序数组中，比较开头和结尾就可以知道目标值是否在数组中
		  3.判断在不在有序数组中，即可达到剪枝的目的，完成快速搜索

	做法：二分查找，比较nums[left]和nums[mid]的值，判断左边是不是有序数组，
				是的话在左边进行比较，看target是不是在有序数组中，如果不在，那么就在右边数组

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
			//左边数组有序
			if (target >= nums[left] && target < nums[mid])
			{
				//在左边数组里面
				right = mid - 1;
			}
			else
			{
				left = mid + 1;//在右边数组里面
			}
		}
		else
		{
			//右边数组有序
			if (target > nums[mid] && target <= nums[right])
			{
				left = mid + 1;//在右边数组里面
			}
			else
			{
				//在左边数组里面
				right = mid - 1;
			}
		}
	}
	return -1;
}

//34. 在排序数组中查找元素的第一个和最后一个位置
/*
思路：
	二分查找，找到这个元素以后往两边找，返回范围

*/

//从mid往两边找范围
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
//35.搜索插入位置
/*
思路：二分查找
	找到了返回
	找不到那么此时left=right,这个数是最接近的，看下这个数比目标值大还是小，然后决定插入到左边还是右边

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
			//在左边找
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

//36. 有效的数独
/*
思路：用三个数组分别存储当前的行，列，box中，用作哈希表，每次遇到新的数，先查看一下当前这个数字是否是出现过，如果出现过，那么直接返回false，表示错误
				1						2						3
1		(0,0)-(2,2)->0			(3,0)-(5,2)->1			(6,0)-(8,2)->2		row/3
2		(0,3)-(2,5)->4			(3,3)-(5,5)->5			(6,3)-(8,5)->6
3		(0,6)-(2,8)->7			(3,6)-(5,8)->8			(6,6)-(8,8)->9

		col/3*3+row/3
		坐标

*/
bool Solution::isValidSudoku(vector<vector<char>>& board)
{
	int row[9][10] = { 0 };//记录行
	int col[9][10] = { 0 };//记录列
	int box[9][10] = { 0 };//记录每个块的
	//遍历整个数独
	for (int i = 0; i < 9; i++)
	{
		for (int j = 0; j < 9; j++)
		{
			if (board[i][j] == '.')
				continue;
			int num = board[i][j] - '0';

			//行赋值
			if (row[i][num] == 0)
			{
				row[i][num] = 1;
			}
			else
			{
				return false;
			}
			//列赋值
			if (col[j][num] == 0)
			{
				col[j][num] = 1;
			}
			else
			{
				return false;
			}
			//box赋值
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

//37. 解数独
/*
思路：回溯=穷举+剪枝
用三个数组保存行，列和块的对应元素的个数
然后一个个的往里面放，当检查到有元素冲突的时候，直接return,到头放回都没问题，那就将结果放进result,然后返回
*/
//功能：在x,y这个位置放置一个 0-9 的值
/*
vector<vector<int>> result(9, vector<int>(10, 0));
void placeNum(int x,int y,vector<vector<int>> row, vector<vector<int>> col, vector<vector<int>> box, vector<vector<int>> curBoard)
{
	//剪枝操作
	if (row[x][num] == 0){
		row[x][num] = 1;//行赋值
	}
	else{
		return;
	}
	if (col[y][num] == 0){
		col[y][num] = 1;//列赋值
	}
	else{
		return;
	}
	if (box[x / 3 * 3 + y / 3][num] == 0){
		box[x / 3 * 3 + y / 3][num] = 1;//box赋值
	}
	else{
		return;
	}
	//当前位置可以放置这个数,下个位置放置数
	curBoard[x][y] = num;
}

void Solution::solveSudoku()
{
	vector<vector<int>> row(9, vector<int>(10, 0));//记录行
	vector<vector<int>> col(9, vector<int>(10, 0));//记录列
	vector<vector<int>> box(9, vector<int>(10, 0));//记录每个块的
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
//38. 外观数列
/*
思路：
要 描述 一个数字字符串，首先要将字符串分割为 最小 数量的组，每个组都由连续的最多 相同字符 组成。
然后对于每个组，先描述字符的数量，然后描述字符，形成一个描述组。
要将描述转换为数字字符串，先将每组中的字符数量用数字替换，再将所有描述组连接起来。

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

//39. 组合总和
/*
思路：
由树递归而成
每次减去指定的数

循环：
1.查看自己是不是零或者小于零	参数：当前的值
2.如果大于零的话将当前的数依次往下减candidates中的数字,并记录当前减去的数

停止条件：减到负或者减到0
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
				path.push_back(candidates[i]);//第i个数塞进去看看效果
				dfs(num - candidates[i], i, candidates, path);
				path.pop_back();//塞完了拿出来再,再塞下一个
			}
		}
	}
};

vector<vector<int>> Solution::combinationSum(vector<int>& candidates, int target) {
	dfs39 d;
	d.dfs(target, 0, candidates, {});
	return d.result;
}

//40. 组合总和 II
/*
思路：回溯
如何穷举？
将第一个数字单独拿出来，看是不是等于target，不是再拿第二个，还不是就拿第三个,指针只往后指

剪枝？ 当数字减小到零，减到了头，减到了负数，返回

去重:排序，只对相同数字的第一个进行遍历，其它的跳过
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
				path.push_back(candidates[i]);//第i个数塞进去看看效果
				dfs(num - candidates[i], i + 1, candidates, path);
				path.pop_back();//塞完了拿出来再,再塞下一个
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
//41. 缺失的第一个正数
/*
思路：原地哈希表
原理：缺失的第一个正数肯定不会超过数组长度
如果
建立一个数组长度n+1大小的数组，构建哈希表，一次遍历找到缺失的元素,就是对应的最小的正整数，但额外花销了n的空间

我们可以慢慢将数组构造成对应的样子即可
构造方法：
从头到尾遍历数组，检查当前的数是不是在对应的位置上
1 2 3 4 5 6 7	value
0 1 2 3 4 5 6	index

如果不在的话，需要将这个数交换到对应的位置上去

如果冲突了(两个数一样)，那么往后找一个不冲突的位置

最后再从头遍历一下找到缺失的元素即可
			//对于大于零的情况，需要分类
			//如果范围是 0-N的，将其交换到对应的位置
			//大于N的，当else 和负数一样 处理，放哪里无所谓

*/
int Solution::firstMissingPositive(vector<int>& nums) {
	int i = 0;
	while (i < nums.size())
	{
		if (nums[i] > 0 && nums[i] <= nums.size()) {
			if (nums[i] - 1 == i) {
				//在对应的位置上的
				i++;
			}
			else {
				//不在对应的位置上,交换到对应的位置上去
				//如果交换的位置已经有了对应的元素在了，那么当前这个其实已经没什么意义了，可以当做和废数一样处理
				if (nums[i] == nums[nums[i] - 1]) {
					i++;
					continue;
				}
				swap(nums[i], nums[nums[i] - 1]);
			}
		}
		else {
			//小于零的或者是大于N的无关紧要的数,那就下一轮了
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

//42. 接雨水
/*
思路：1.如果柱子的碰见一个比他高的，那么这个矮的就到头了，前面的就和后面没关系了，后面就把它给挡住了，记录到目前的能盛的水

/////错误观点，反例，阶梯中间一个坑/////	2.如果碰见一个比他矮的，那么接着往后看到头，并记录这目前见到的最矮的(只要不比他高就行)和目前的盛水量

可行的一个思路：

原理：依据前面的1.
做法：我们找到能挡住的最高挡板，然后两边往中间遍历

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
//43. 字符串相乘
/*
思路：模拟列竖式的方式
乘数一个个的乘被乘数的每一位，然后做加法，记录进位和结果位（惊现数电乘法器）

*/
/*功能:封装乘法器
* num1,num2:两个数字字符
* 返回值：<进位，个位>
*/
pair<char, char> multiplySingle(char num1, char num2, char carry)
{
	int n1 = num1 - '0';
	int n2 = num2 - '0';
	int n3 = carry - '0';
	int result = n1 * n2 + n3;
	return pair<char, char>{result / 10 + '0', result % 10 + '0'};
}

/*功能:封装加法器
* num1,num2:两个数字字符,第一个是上面的没有偏移的数，第二个是偏移过的数，n是偏移量
* 返回值：<进位，个位>
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
		//两个都有数
		if (i + n < num1.size() && i < num2.size())
		{
			temp = num1[num1.size() - 1 - n - i] - '0' + num2[num2.size() - 1 - i] - '0';
			result += (temp + carry) % 10 + +'0';
			carry = (temp + carry) / 10;
		}
		else if (i < num2.size())
		{
			//只有第二个有数
			temp = num2[num2.size() - 1 - i] - '0';
			result += (temp + carry) % 10 + '0';
			carry = (temp + carry) / 10;
		}
		else
		{
			//只有第一个有数
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
	//加上最后的进位，倒置结果得到正确的顺序,是乘一位得到的结果
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
		//将前面的加起来
		result = addNumString(result, curRes, num2.size() - 1 - i);
	}
	return result;
}
//44. 通配符匹配
/*
思路：先填表

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
	//初始化
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
			case '*'://如果当前是*的话，那么代表可以匹配的是上面的这个或者往后的都可以，所以应该是从上面的 1 开始
				//如果*代表的可以作为空串的话，必须要保证它和前面的可以连接的上，但是如果*是在开头的话就会连接不上，所以我们需要补充连接的条件
				if (map[i - 1][0] == 1) {
					j = 0;//对这一行都进行填充
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

//45. 跳跃游戏 II
/*
思路：贪心算法
原理：保证我每次都能跳到我可以掌握到的范围更大的位置，这样可以比较到更多的信息
每次都跳到一个范围很大的位置，直到可以跳出去为止

*/
int Solution::jump(vector<int>& nums) {
	if (nums.size() == 1) {
		return 0;
	}
	int times = 0;
	int end = nums[0];//当前势力圈结尾
	int maxpos;//下一个势力圈的结尾
	for (int i = 0; i < nums.size() - 1; i++) {
		//对于每个数字
		maxpos = max(maxpos, i + nums[i]);//找下一个势力范围最大的,它对应的位置就是要跳到的位置
		if (i == end) {
			end = maxpos;
			times++;
		}
	}
	return times + 1;
}

//46. 全排列
//对于每个节点要干的事
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

//47. 全排列 II
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

//48. 旋转图像
/*
思路：旋转四个角即可
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
			//开始旋转
			matrix[left][left + i] = matrix[right - i][left];
			matrix[right - i][left] = matrix[right][right - i];
			matrix[right][right - i] = matrix[left + i][right];
			matrix[left + i][right] = temp;
		}
		left++;
		right--;
	}
}
//49. 字母异位词分组
/*
用到的是排序+哈希表的方法，排序后的字符肯定是一样的，这样以排序后的字符作为key值，value为一个string类的vector，每次往里面添加字符即可

还有一种方法是计数的方法，原理在于每个数字里面对应的字母的个数肯定是一样的
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
思路：x的n次方，对应的是多少个n相乘，但x4是可以由x2直接平方得来
x11是x8×x2×x
对应的是二进制的11  1011，低位的就是x几次方的个数

做法：将n二进制化，每次都乘当前的数，得到x,x2,x4,x8，二进制的每一位对应了要不要乘当前得到的数

*/

double Solution::myPow(double x, int n) {
	double result = 1;
	int sign = n > 0 ? 0 : 1;
	while (n != 0) {
		if (abs(n % 2) == 1) {
			//最低位是1,需要乘上去
			result *= x;
		}
		x *= x;
		n /= 2;
	}
	if (sign == 0)	return result;
	else return 1 / result;
}
//51. N 皇后
/*
思路:
	回溯，穷举+剪枝
	每个节点要做的事：查看当前节点是否符合要求，如果不符合要求返回
						如果符合要求，往下一行看
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
		//符合要求，看看下面一行的
		for (int i = 0; i < N; i++) {
			//看看能不能往这里放
			//我想往 row+1 ,i 这一位置放，可以放的条件摆在if里面了，同样的也就是剪枝操作
			if (isused[i] == false && isXie(row + 1, i, coordinate)) {	//这一列没人用,	//排除一下斜对角的
				isused[i] = true;		//没人用我用了
				path[row + 1][i] = 'Q';
				coordinate[row + 1] = i;
				dfs(row + 1, path, isused, coordinate);	//再去找下一行了啊
				path[row + 1][i] = '.';		//用完了还你
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
//53. 最大子序和
/*
思路一：动态规划，列表，因为时间太长失败了
*/
int Solution::maxSubArray(vector<int>& nums) {
	vector<vector<int>> map(nums.size(), vector<int>(nums.size()));
	//初始化
	int result = nums[0];
	for (int i = 0; i < nums.size() - 1; i++) {
		for (int j = i; j < nums.size(); j++) {
			if (i == j) {
				//初始化
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
	错误思路2，原因在于记录的最大值是有可能不挨着最新的一个元素的，因此下面的思路是个假的动态规划

	思路二：动态规划
	思考动态规划，从1开始思考
	如果给一个数，那么最大值就是它，maxVal=nums[0]
	如果给两个数，那么最大和的连续子数组是 maxVal=max(1.第一个(maxVal)，2.或者第二个(nums[1])，3.或者两个的和(maxVal+nums[1]))
	三个数，结果应该是，max(maxVal+nums[2],nums[2],maxVal)

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
思路3：动态规划
	其实问啥就设啥就好，就设当前的以第 i 个数结尾的「连续子数组的最大和」 是fi
	他和前面的关系是啥呢？
	要么就是前面的f(i-1)+当前的数，要么就是等于当前的数

	所以核心的表达式
	f(i)=max(f(i-1)+nums[i],nums[i])

	最终在所有的fi中取最大值就可以

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

//54. 螺旋矩阵
/*
	思路：模拟这个寻找的过程即可
*/

vector<int> Solution::spiralOrder(vector<vector<int>>& matrix) {
	int row = matrix.size();
	int col = matrix[0].size();
	int left = 0, right = matrix[0].size() - 1, up = 0, down = matrix.size() - 1;//边界
	enum { RIGHT = 0, DOWN, LEFT, UP };//方向状态
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
//55. 跳跃游戏
/*
思路：贪心算法
	从头遍历，一直找自己能跳到的最长的位置

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

//56. 合并区间
/*
思路:排序
排序后按照开头的顺序，记录下当前的开始和结束，如果发现新的开始比结束还大，那就是断档了，需要把前面的塞进结果去，后面的再重新排

*/
vector<vector<int>> Solution::merge(vector<vector<int>>& intervals) {
	sort(intervals.begin(), intervals.end());
	int begin = intervals[0][0], end = intervals[0][1];
	vector<vector<int>> result;
	for (int i = 0; i < intervals.size(); i++) {
		//断档的
		if (intervals[i][0] > end) {
			result.push_back({ begin,end });
			begin = intervals[i][0];
		}
		//更新end
		end = max(end, intervals[i][1]);
	}
	result.push_back({ begin,end });
	return result;
}

//57. 插入区间
/*
思路：看新插入的这个区间[a,b]在哪个位置

如果 a 在区间内部，那么新的区间最小值就是这个区间的最小值
如果 a 在区间外部，找到其对应的位置
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
		//寻找a在的位置
		//在内部
		if (a >= intervals[i][0] && a <= intervals[i][1]) {
			low = intervals[i][0];
			break;
		}
		//在外部
		else if ((i - 1 >= 0 && a >= intervals[i - 1][1] && a <= intervals[i][0]) ||
			a< intervals[0][0] ||
			a>intervals[intervals.size() - 1][1])
		{
			low = a;
			break;
		}
	}
	//寻找b在的位置
	for (int i = 0; i < intervals.size(); i++) {
		//在内部
		if (b >= intervals[i][0] && b <= intervals[i][1]) {
			high = intervals[i][1];
			highIndex = i + 1;
			break;
		}
		//在外部
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
	//往里塞结果了
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
//58. 最后一个单词的长度
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
//59. 螺旋矩阵 II
/*
思路：模拟
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

//60. 排列序列
/*
思路：
	1打头的有(n-1)!个排列，	2打头的有(n-1)!个排列......

	X=a[n]*(n-1)!+a[n-1]*(n-2)!+...+a[i]*(i-1)!+...+a[1]*0!
	数字值
	000			->		1
	001			->		2

	看k取除以后，剩下 order  ，就把排m+1个排序的数放进去(m从1开始)
	对于第k个排列，看其位于哪个位置
*/
string Solution::getPermutation(int n, int k) {
	vector<int> factorial(n);
	factorial[0] = 1;
	for (int i = 1; i < n; i++) {
		factorial[i] = factorial[i - 1] * i;
	}
	string result;
	k--;//数字的第k个和排序后取除法得到的数差一个
	//放n个数进去
	vector<bool> isused(n + 1);
	for (int i = 1; i <= n; i++) {
		int order = k / factorial[n - i] + 1;
		//找第 order小的数 j
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
//61. 旋转链表

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
	//right指向了最后一个节点
	right->next = preHead.next;
	preHead.next = left->next;
	left->next = NULL;

	return preHead.next;
}
/*
思路2:如果要旋转的话，因为怎么也要找到尾部，所以可以直接找到尾部，连成圈先，然后在继续遍历就把需要的地方拆开即可

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

//62. 不同路径
/*
思路：递归树
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
		//往右
		findway(x + 1, y);
		//往下
		findway(x, y + 1);
	}
};
int Solution::uniquePaths1(int m, int n) {
	dfs62 d(m, n);
	d.findway(0, 0);
	return d.wayNums;
}
/*
思路2:动态规划

	想如果是只有一个点的话，那么路径就是1，如果两个点的话，也是1，四个点，路径是2，分别是横着过来的的和竖着过来的
	如果考虑 2*3的话，最右下角的应该是	左边的2条路的情况+上面过来拿的1种情况

	直接设走到（m-1，n-1）的路径条数为 x,
	那么能到达终点的有两种，一种从上面来的，一种从左边来的
	从上面来的问题和这个问题是同一个问题，同理从左边来的也是，都是视这个点为终点的路径的条数
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

//63. 不同路径 II
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
改进：可以用滚动数组的思想进一步解决
因为我们遍历的时候，每次都是一行行的遍历，但是对我们求解当前的状态 有用的信息只有最近的两行，
因此我们可以只用一行就可以求解出来

*/
int Solution::uniquePathsWithObstacles2(vector<vector<int>>& obstacleGrid) {
	if (obstacleGrid.size() == 0 || obstacleGrid[0].size() == 0) {
		return 0;
	}
	int m = obstacleGrid.size();
	int n = obstacleGrid[0].size();
	//正常情况
	vector<int> map(n);
	//初始化
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
	//如果当前位置是1的话，有障碍，map直接为0
	//如果没有障碍，那么应该等于它的 上面来的次数(之前的它)+左边来的
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
改进3：可以用滚动数组的思想进一步解决
因为我们遍历的时候，每次都是一行行的遍历，但是对我们求解当前的状态 有用的信息只有最近的两行，
因此我们可以只用一行就可以求解出来

*/
int Solution::uniquePathsWithObstacles3(vector<vector<int>>& obstacleGrid) {
	if (obstacleGrid.size() == 0 || obstacleGrid[0].size() == 0) {
		return 0;
	}
	int m = obstacleGrid.size();
	int n = obstacleGrid[0].size();
	//正常情况
	vector<int> map(n);
	//初始化
	if (obstacleGrid[0][0] == 1) {
		return 0;
	}
	map[0] = 1;
	for (int i = 0; i < m; i++) {
		for (int j = 0; i < n; j++) {
			//可以巧妙的将边界也计算在内
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

//64. 最小路径和
/*
思路：动态规划

到最后一点距离的数字总和最小，那么到上一个点的一定也是最小

设到某个点的最小的距离是 fi =min(fi←,fi↑)

然后从头考虑，(初始化)
一个格子的话，到他的距离最小没得商量就是第一个格子的值,第一行和第一列的话也没得商量，就一种选择
中间的每个点都是两边的 本身的值+min(←，↑)
可以优化空间，滚动数组即可
*/

int Solution::minPathSum(vector<vector<int>>& grid) {
	if (grid.size() == 0 || grid[0].size() == 0) {
		return -1;
	}
	int m = grid.size();
	int n = grid[0].size();
	vector<int> f(n);
	//初始化
	f[0] = grid[0][0];
	for (int j = 1; j < n; j++) {
		f[j] = f[j - 1] + grid[0][j];
	}

	//动态规划
	for (int i = 1; i < m; i++) {
		f[0] += grid[i][0];
		for (int j = 1; j < n; j++) {
			f[j] = grid[i][j] + min(f[j - 1], f[j]);
		}
	}
	return f[n - 1];
}
//66. 加一
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
//67. 二进制求和
/*
思路：双指针
*/
//第一个是返回值
//第二个是进位标志
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
//68. 文本左右对齐
/*
思路：模拟

每次往里面塞，记录当前塞进去的 单词数 和 长度
如果发现往下塞下一个单词的时候，长度超过这一行了，停止塞了。
计算一下每个单词的空格数=(本行的长度-塞进去的单词长度)/单词数

循环上面过程，直到塞进当前的这个是结尾了，直接把前面的一股脑塞进去

*/
// "This", "is", "an", "example", "of", "text", "justification."
vector<string> Solution::fullJustify(vector<string>& words, int maxWidth) {
	//可能整个单词比一行还长

	//正常情况
	//每个单词看
	vector<string> result;
	//words可能为空
	if (words.empty()) {
		return result;
	}
	int begin = 0;
	int len = 0;//记录已塞进去的单词总长度
	deque<string> q;//记录塞进去的单词
	for (int i = 0; i < words.size(); i++) {
		if (len + words[i].size() <= maxWidth) {
			//可以往里面塞
			q.push_back(words[i]);
			len += words[i].size() + 1;//将这个单词 word 和空格合并，看成一个长度
		}
		else {
			i--;
			//这个单词不能塞了，把前面的都放进去
			len--;//把最后一个的空格去掉
			int num = q.size();//可以塞的单词总个数

			//计算空格分配
			string temp = "";
			if (num == 1) {
				//只有一个单词
				int num = q.size();//可以塞的单词总个数
				//塞进去最后不带空格的
				temp += q.front() + string(maxWidth - len, ' ');
				q.pop_front();
			}
			else {
				//多个单词分配空格
				int aveBlank = (maxWidth - len) / (num - 1);//平均每个单词分到的空格数
				int remain = (maxWidth - len) % (num - 1);//剩下的空格数
				for (int j = 0; j < num - 1; j++) {
					if (remain > 0) {
						temp += q.front() + string(aveBlank + 2, ' ');
						remain--;
					}
					else {
						temp += q.front() + string(aveBlank + 1, ' ');
					}
					q.pop_front();//删掉第一个
				}
				temp += q.front();
				q.pop_front();
			}
			result.push_back(temp);
			len = 0;
		}
	}
	//把里面剩下的都塞成最后一行即可
	if (!q.empty()) {
		len--;
		int num = q.size();//可以塞的单词总个数
		string temp = "";
		//塞进去前面带空格的
		for (int j = 0; j < num - 1; j++) {
			temp += q.front() + " ";
			q.pop_front();
		}
		//塞进去最后不带空格的
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
改进：
前面功能混淆，代码重复也很大
功能区分开：
	拿数：边拿边计数，直到下一个不能塞了位置
	塞数：将指定的坐标范围的数塞进去，并按照指定要求分配空格，并将塞好的数返回，
	如果是最后一个，那就按另外一套规则去塞
*/
//功能：塞数
/*
* words:
* start:
* end:
* maxWidth:
* len:每个单词加了空格以后放进来的长度(最后一个不加空格)
* isEnd:结尾的
*/

string pushnum(vector<string>& words, int start, int end, int maxWidth, int len, bool isEnd) {
	string ans = "";
	//如果只有一个字符要塞的
	if (start == end) {
		return words[start] + string(maxWidth - len, ' ');
	}
	//如果是最后一行
	if (isEnd) {
		for (int i = start; i < end; i++) {
			ans += words[i] + " ";
		}
		ans += words[end];
		return ans + string(maxWidth - len, ' ');
	}
	//多个
	int spaceNum = maxWidth - len;//要分配的空格数
	int aveSpace = spaceNum / (end - start);//平均每个字符后面跟着的空格
	int reserve = spaceNum % (end - start);//余下的需要分配的空格
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
	int sysmbleCount = 0;//符号计数
	int start = 0;
	vector<string> ans;
	for (int i = 0; i < words.size() - 1; i++) {
		sysmbleCount += words[i].size() + 1;//把这个数塞进去
		//如果把下一数塞进去算上就超了，那么把这一行整理一下塞进结果
		if (sysmbleCount + words[i + 1].size() > maxWidth) {
			ans.push_back(pushnum(words, start, i, maxWidth, sysmbleCount - 1, 0));
			start = i + 1;
			sysmbleCount = 0;
		}
	}
	sysmbleCount += words.back().size() + 1;//把最后的数塞进去
	ans.push_back(pushnum(words, start, words.size() - 1, maxWidth, sysmbleCount - 1, 1));

	for (string s : ans) {
		cout << s << endl;
	}
	return ans;
}
//69. x 的平方根
/*
	求解：f(x)=x2-c的零点
	构造迭代式： x=x- (x2-c)/2x =(x+C)/2x
下面是错误推理：
	可以，但是没必要，因为判断条件不好确定，如果碰到重根的话，收敛很慢
正确想法：可以的，x^2它的收敛不会说左边一下，右边一下，左右波动，因此我们结束的判断条件可以采用f(x)=x*x-c<0来进行判别。
而且根据其取整的性质，一定是可以转移到根的左边

用二分法：取中
	如果中间的平方小于x，在有半区间
	如果中间的平方大于x，在左半区间
*/

int Solution::mySqrt(int c) {
	int x = c;

	while (x * x > c) {
		x = (x * x + c) / (x * 2);
	}

	return x;
}
//70. 爬楼梯
/*
思路：
1.递归，每次选择减去1或者减去2，最终减出来是0就返回
2.动态规划，如果我找到n条路，你再多出来一阶，同样得+1，多两阶，+2以此（怎么好像有规律）,
	所以到第n阶的情况的条数fn=fn-1+fn-2
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
//71. 简化路径
/*
需要解决的问题：
	1.尾部有/的去掉
	2.带..的，将前面的删除掉
	3.根目录的带 ..还是根目录
	4.双斜线的去掉一个
	5. . 表示当前的路径，没啥用，去掉

思路：搞个队列,从头到尾的遍历这个串，
	遇到'/'就结束当前记录的串
	检查当前的串的内容
	如果不是'.' 或者 '..'直接塞进栈
	如果是'.' 不理会，接着看后面了
	如果是'..',弹出一个

	while(s[i]!='/')
		s+=s[i];
	if(not '.' or '..') push(/+s)
	else if(s=='.')
	else if(s=="..")  if(not empty)  pop(s)		
错误案例：
	/...  ,
	/a..	,	/..a
*/


string Solution::simplifyPath(string path) {
	if (path.back() != '/') {
		path += '/';
	}
	deque<string> q;//只存依次的路径
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
//改进

class Solution71 {
public:
	deque<string> q;//只存依次的路径
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
//72. 编辑距离
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

	// 有一个字符串为空串
	if (n * m == 0) return n + m;

	// DP 数组
	vector<vector<int>> D(n + 1, vector<int>(m + 1));

	// 边界状态初始化
	for (int i = 0; i < n + 1; i++) {
		D[i][0] = i;
	}
	for (int j = 0; j < m + 1; j++) {
		D[0][j] = j;
	}

	// 计算所有 DP 值
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


//73. 矩阵置零
/*
思路：从头到尾遍历，找到了零，把这一列和这一行开头的设成0（第一列和第一行另算）


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
	//第一列
	for (int i = 0; i < row; i++) {
		if (matrix[i][0] == 0) {
			colZero = 1;
			break;
		}
	}
	//第一行
	for (int i = 0; i < col; i++) {
		if (matrix[0][i] == 0) {
			rowZero = 1;
			break;
		}
	}
	//遍历
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
	//重新弄
	if (rowZero == 1) {
		makeRowZero(matrix, 0);
	}
	if (colZero == 1) {
		makeColZero(matrix, 0);
	}
}
//74. 搜索二维矩阵
/*
思路：二分查找	
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
//75. 颜色分类
/*
	思路：
		简单的思路：计数排序，数一数0  2的个数，然后1=剩下的，再重新赋值一遍即可
		但是如果要求一遍遍历的话，我们可以用双指针替换掉相应的空间消耗 
*/

void Solution::sortColors(vector<int>& nums) {
	int m = nums.size();
	int p0 = 0;
	int p2 = nums.size() - 1;
	for (int i = 0; i < p2; i++) {
		if (nums[i] == 2) {
			//有可能一直给丢回来个2,也有可能丢回来个0，但是如果是0我们合并下一步去进一步验证
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
//76. 最小覆盖子串
/*
暴力思路：
	从头到尾看一遍嘛，看看够不够目标的串，不够的话直接返回不行了，够的话，那就把左边的删一个，可以就删俩，直到不行了为止，然后就删右边的...
	不太对劲，有可能你删除的时候，把最优解从左边先删除了
	那？
	左边删一个？右边删一个？	开始递归找最优的？
	但是递归的话？
	好像会有重叠的情况
	比如我 左左右   和右左左  左右左的结果是一样的哎
	那？
	动态规划？
	显然可以左边删和右边删两种方案，那就二维数组表示他们所有的排列组合。

	用二维数组表示的话，每一位表示如果删除掉这一位能不能包含了要求的这些字符
	可以发现，如果左边的和上边的不能满足的话，那么后面的肯定更不能满足
	转移方程 dp[i][j]=dp[i-1][j]&&dp[i][j-1]&&(del cur is ok)
	但是怎么计算删除掉这个字符能不能进行匹配呢？
	1.暴力匹配，直接从头到尾检查个遍
	2.发现你检查过程中，好像？和之前的检查重复了额，因为就多增加了一个字符的量，结果还得从头检查一遍

	那从做到右过程中，多弄个map做记录呢？
	咋跟人家滑动窗口快接近了，滑动窗口如果在这个基础上，从头先跑到尾，然后两头缩？

	那先跑到能收集全了召唤神龙的情况，再缩左边指针会不会效率高点？

思路：滑动窗口
	从头到尾遍历s
	对于每个字符，如果发现需要匹配的串里面有这个字符，那么开始记录了

	记录的方法就是如果看到当前字符在这个匹配串里面，那么对应的哈希表的数字加一
	如果对应的数字全了，	cnt和ori相应的大小一样了
	那么开始缩小指针范围，	void short
	寻找最小值,				
	缩到最小值以后,做个记录
	再继续往后找
*/
//返回缩掉以后的长度
void Solution76::shrinkIndex(string& s, int& pleft, const int& pright) {
	int i = pleft;
	for (i = pleft; i <= pright; i++) {
		if (cnt.find(s[i])!=cnt.end()) {
			cnt[s[i]]--;
	
//-------------------出问题的点---------------------------------------	


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
	//先找到开头
	while (pleft < m && ori.find(s[pleft]) == ori.end()) {
		pleft++;
	}
	pright = pleft;
	while (pright < m && pleft < m) {
		if (pright < m && ori.find(s[pright]) != ori.end()) {
			//开始找，如果说找到了对应的字符
			cnt[s[pright]]++;
			if (cnt.size() == ori.size()) {
				shrinkIndex(s, pleft, pright);
			}
		}
		pright++;
	}
	return s.substr(sleft,len);	
}
