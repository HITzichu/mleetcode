#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <stack> 
#include <map>
#include <queue>
using namespace std;
//19. ɾ������ĵ�����N���ڵ�
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




class Solution
{
public:
    //4.Ѱ�����������������λ��  
    static double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2);
    //5.������Ӵ�  
    static string longestPalindrome(string s);
    //6.Z���ͱ任
    static string convertZ(string s, int numRows);
    //7.������ת
    static int Myreverse(int x);
    //8.�ַ���ת������ (atoi)
    static int myAtoi(string str);
    //9.������
    static bool isPalindrome(int x);
    //11.ʢ���ˮ������
    static int maxArea(vector<int>& height);
    //12.����ת��������
    static string intToRoman(int num);
    //13.����ת����
    static int romanToInt(string s);
    //14.�����ǰ׺
    static string longestCommonPrefix(vector<string>& strs);
    //15.����֮��
    static vector<vector<int>> threeSum(vector<int>& nums);
    //16.��ӽ�������֮��  
    static int threeSumClosest(vector<int>& nums, int target);
    //17. �绰�������ĸ���
    static vector<string> letterCombinations(string digits);
    //18.����֮��
    static vector<vector<int>> fourSum(vector<int>& nums, int target);
    //19. ɾ������ĵ�����N���ڵ�
    static ListNode* removeNthFromEnd(ListNode* head, int n);
    //20. ��Ч������
    static bool isValid(string s);
    //21.�ϲ�������������  
    static ListNode* mergeTwoLists(ListNode* l1, ListNode* l2);
    //22.��������
    static vector<string> generateParenthesis(int n);
    //23.�ϲ�K����������
    static ListNode* mergeKLists(vector<ListNode*>& lists);
    //24. �������������еĽڵ�
    static ListNode* swapPairs(ListNode* head);
    //25. K ��һ�鷭ת����
    static ListNode* reverseKGroup(ListNode* head, int k);
    //26. ɾ�����������е��ظ���
    static int removeDuplicates(vector<int>& nums);
    //27. �Ƴ�Ԫ��
    static int removeElement(vector<int>& nums, int val);
    //28. ʵ�� strStr()
    static int strStr(string haystack, string needle);
    //29. �������
    static int divide(int dividend, int divisor);
    //30. �������е��ʵ��Ӵ�
    static vector<int> findSubstring(string s, vector<string>& words);
    //31. ��һ������
    static void nextPermutation(vector<int>& nums);
    //32. ���Ч����
    static int longestValidParentheses(string s);
    //33. ������ת��������
    static int search(vector<int>& nums, int target);
    //34. �����������в���Ԫ�صĵ�һ�������һ��λ��
    static vector<int> searchRange(vector<int>& nums, int target);
    //35. ��������λ��
    static int searchInsert(vector<int>& nums, int target);
    //36. ��Ч������
    static bool isValidSudoku(vector<vector<char>>& board);
    //37. ������
    static void solveSudoku();
    //38.�������
    static string countAndSay(int n);
    //39. ����ܺ�
    static vector<vector<int>> combinationSum(vector<int>& candidates, int target);
    //40. ����ܺ� II
    static vector<vector<int>> combinationSum2(vector<int>& candidates, int target);
    //41. ȱʧ�ĵ�һ������
    static  int firstMissingPositive(vector<int>& nums);
    //42. ����ˮ
    static int trap(vector<int>& height);
    //43. �ַ������
    static string multiply(string num1, string num2);
    //44.ͨ���ƥ��
    static bool isMatch(string s, string p);
    //45. ��Ծ��Ϸ II
    static int jump(vector<int>& nums);
    //46. ȫ����
    static vector<vector<int>> permute(vector<int>& nums);
    //47. ȫ���� II
    static vector<vector<int>> permuteUnique(vector<int>& nums);
    //48. ��תͼ��
    static void rotate(vector<vector<int>>& matrix);
    //49. ��ĸ��λ�ʷ���
    static vector<vector<string>> groupAnagrams(vector<string>& strs);
    //50. Pow(x, n)
    static double myPow(double x, int n);
    //51. N �ʺ�
    static vector<vector<string>> solveNQueens(int n);
    //53. ��������
    static int maxSubArray(vector<int>& nums);
    static int maxSubArray2(vector<int>& nums);
    static int maxSubArray3(vector<int>& nums);
    //54. ��������
    static vector<int> spiralOrder(vector<vector<int>>& matrix);
    //55. ��Ծ��Ϸ
    static bool canJump(vector<int>& nums);
    //56. �ϲ�����
    static vector<vector<int>> merge(vector<vector<int>>& intervals);
    //57. ��������
    static vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval);
    //58. ���һ�����ʵĳ���
    static int lengthOfLastWord(string s);
    //59. �������� II
    static vector<vector<int>> generateMatrix(int n);
    //60. ��������
    static string getPermutation(int n, int k);
    //61. ��ת����
    static ListNode* rotateRight(ListNode* head, int k);
    //62. ��ͬ·��
    static int uniquePaths(int m, int n);
    static int uniquePaths1(int m, int n);
    //63. ��ͬ·�� II
    static int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid);
    static int uniquePathsWithObstacles2(vector<vector<int>>& obstacleGrid);
    static int uniquePathsWithObstacles3(vector<vector<int>>& obstacleGrid);
    //64. ��С·����
    static int minPathSum(vector<vector<int>>& grid);
    //65.����β֭

    //66. ��һ
    static vector<int> plusOne(vector<int>& digits);
    //67.���������
    static string addBinary1(string a, string b);
    static string addBinary2(string a, string b);
    //68. �ı����Ҷ���
    static vector<string> fullJustify(vector<string>& words, int maxWidth);
    static vector<string> fullJustify2(vector<string>& words, int maxWidth);
    //69. x ��ƽ����
    static int mySqrt(int x);
    //70. ��¥��
    static int climbStairs(int n);
    //71. ��·��
    static string simplifyPath(string path);
    static string simplifyPath2(string path);
    //72. �༭����
    static int minDistance(string word1, string word2);
    static int minDistance2(string word1, string word2);
    //73. ��������
    static void setZeroes(vector<vector<int>>& matrix);
    //74. ������ά����
    static bool searchMatrix(vector<vector<int>>& matrix, int target);
    //75. ��ɫ����
    static void sortColors(vector<int>& nums);
};
//76. ��С�����Ӵ�
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
    //�����ұ�������¼ÿ�����ӵ���߽�
    vector<int> findLeftBorder(vector<int>& heights) {
        stack<int> borderStack;
        int border = 0;
        vector<int> ret(heights.size(), 0);
        for (int i = 0; i < heights.size(); i++) {
            if (i == 0 || heights[i] > heights[i - 1]) {
                //������
                borderStack.push(i);
                border = i;
            }
            else {
                //�½��ˣ���������߽�
                while (!borderStack.empty() && borderStack.top() >= heights[i]) {
                    border = borderStack.top();
                    borderStack.pop();
                }
            }
            ret[i] = border;
        }
        return ret;
    }
    //���ҵ����������¼ÿ�����ӵ��ұ߽�
    //[ 3,2,4,3,4 ]
    vector<int> findRightBorder(vector<int>& heights) {
        stack<int> borderStack;
        int n = heights.size();
        int border = n - 1;
        vector<int> ret(n, n - 1);
        for (int i = heights.size() - 1; i >= 0; i--) {
            if (i == n - 1 || heights[i] > heights[i + 1]) {
                //������
                borderStack.push(i);
                border = i;
            }
            else {
                //�½��ˣ��������ұ߽�
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
˼·��
    ��������ⷨ������ĳһ�����ӣ����������������ң�ֱ���ҵ���Ϊֹ���㵽ͷ�ˣ�Ȼ�������������

    ��������Ҳ���֣�������Ǵ�ͷ���������Ҫ���½��ˣ��Ǻ���Ŀ϶���ǰ���Ǹ����ұ߽磬��Ӧ��������߽�����Լ���ǰȥ����һ��
    ������while()һֱ�ҵ�����С��heights,Ȼ�����������߽�
    ���������Ļ��������׳����ظ��ı���


*/
//�����ұ�������¼ÿ�����ӵ���߽�
/*
    ����ĳ������
    ����������� �ϸ��½��Ļ����������ұ߽������������ͷ��
    ��������� �����Ļ�����ô�����ұ߽��ǲ���ȷ���ģ���Ҫ�������ˣ�ÿһ���������������ұ߽磬
    ����ÿһ����Ҫ�͵�ǰûȷ���ұ߽�����ӱȽ�һ�£�����Լ���ûȷ���߽��С�������λ�õľ��ҵ��Լ����ڵ��ұ߽���
    ��������ȽϵĻ�������Ӧ�ú����ıȣ��������������С���������ұ߽��ȷ���ˣ�Ȼ���ٿ���С�ģ������ܲ���ȷ�������ұ߽翴���Ӳź���

    ���������Ҫһ���ܴ洢δȷ���ұ߽�����ݽṹ�����һ��ܰ���˳�����У����������õ�ʱ���ó��������ֵ���������뵽������ջ
    ��Ȼ����Ӧ�õĹ���Ҳ�϶�������ģ���Ϊ����ÿ��������ʱ���ȿ����Լ��ǲ��Ǳ�ûȷ���ұ߽������ֵ �ϸ�С��
    ���С�Ļ����ҾͰ����ó����ˣ���Ϊ�����ұ߽��Ѿ�ȷ����
    �����Ļ������Լ�������ȥ�ˣ���Ϊ��ȷ��������

    ����һ�룬����������Ż�
    ���ǰѵ�һ�κͺ���ıȽϵ�ֱ��Ҳ������ѭ����,�����ǰ��Լ�ѹ��ջô��

    ����һ����û�з��֣�������߽���ôȷ������һ��ʼ��Ϊ����ͨ������ķ�����ȷ��������߽�
    ���ǿ���һ�������ͻȻ����ջ�����Ȼ�ǵ������ģ���ô��Ӧ�����������ߵĲ�����������߽�ô��
    ��Ϊ�м�ı�����ȥ��ʵ�����Ǳ��Һ�ǰ���Ǹ�С����һ�����������ȥ��
*/
    int largestRectangleArea(vector<int>& heights) {
        stack<int> st;
        st.push(-1);
        int maxArea = 0;
        int n = heights.size();
        for (int i = 0; i < heights.size(); i++) {
            while (st.top() != -1 && heights[st.top()] > heights[i]) {
                //���Լг�ȥ��ǰ���ˣ�ջ�����������Ѿ�ȷ����
                int cur = st.top();//����ȷ���ĺ�
                st.pop();
                int left = st.top() + 1;
                int right = i;//����ֿ�
                maxArea = max(maxArea, (right - left) * heights[cur]);
            }
            st.push(i);
        }
        while (st.top() != -1) {
            int cur = st.top();//����ȷ���ĺ�
            st.pop();
            int left = st.top() + 1;
            int right = n;//һֱ�����û����������˵�����������������
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
                //���Լг�ȥ��ǰ���ˣ�ջ�����������Ѿ�ȷ����
                int cur = st.top();//����ȷ���ĺ�
                st.pop();
                int left = st.top() + 1;
                int right = i;//����ֿ�
                maxArea = max(maxArea, (right - left) * heights[cur]);
            }
            st.push(i);
        }
        while (st.top() != -1) {
            int cur = st.top();//����ȷ���ĺ�
            st.pop();
            int left = st.top() + 1;
            int right = n;//һֱ�����û����������˵�����������������
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
    //ÿһ���������ǿ����ܲ�������һ��Ŷ���
    void dfs(int depth, int start, vector<int> cur) {
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
        this->n = nums.size();
        sort(nums.begin(), nums.end());
        vector<int> cur;
        result.push_back(cur);
        dfs(1, 0, cur);
        return result;
    }
};
/*
˼·�����ݣ�һ�����Ĳ�һ��
��:
ÿһ��Ҫ���ģ����õ���ǰ���ַ�,
�����λ����
1.��������
2.��С����

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
            //û��,ֱ������ȥ���������һ����
            curs += c;
            partnum = c - '0';
            dfs(index + 1, part, partnum, curs, s);
        }
        else {
            //���������в���
            if (partnum == 0 || (partnum * 10 + c - '0') > 255) {
                //�������У��ŵ��в���
                if (part < 4) {
                    curs.push_back('.');
                    dfs(index + 1, part + 1, -1, curs, s);
                }
                else {
                    return;
                }
            }
            else {
                //������
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
/*******************����ר��*****************************/

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
class Solution90duplicate {
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
/*
 * @lc app=leetcode.cn id=498 lang=cpp
 *
 * [498] �Խ��߱���
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
        int dir = UP;//trueΪ���ϣ�falseΪ����
        int x = -1, y = 1;
        for (int i = 0; i < m * n; i++)
        {
            if (dir == UP) {
                x += 1;
                y -= 1;
                if (y == -1 && x == n) {
                    //���Ͻ�
                    x -= 1;
                    y += 2;
                    dir = DOWN;
                }
                else {
                    if (y == -1) {
                        //�ϱ߽�

                        y = 0;
                        dir = DOWN;
                    }
                    if (x == n) {
                        //�ұ߽�
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
                    //���½�
                    x += 2;
                    y = m - 1;
                    dir = UP;
                }
                else
                {
                    if (y == m) {
                        //�±߽�
                        x += 2;
                        y = m - 1;
                        dir = UP;
                    }
                    if (x == -1) {
                        //��߽�
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
		//��С�����Ÿ�
		if (p1[0] < p2[0]) {
			return true;
		}
		else if (p1[0] > p2[0]) {
			return false;
		}
		else { /*(p1[0] == p2[0])*/
			//ͬ�ŵĵڶ�λ�Ӵ�С�ţ�������ȥ��ʱ������߼�
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
			//����ͷ            
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
		//��ߴӵ͵��ߣ���λ�Ӹߵ���
		sort(people.begin(), people.end(), [](vector<int> p1, vector<int> p2)
			{
				return p1[0] < p2[0] || (p1[0] == p2[0] && p1[1] > p2[1]);
			}
		);
		vector<vector<int>> ret(people.size());
		for (auto person : people) {
			int n = person[1];
			//��n����λ�����������ߵ�
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
        ��������ĵ�һ��Ϊ���ڵ㣬�������ķ�ΧΪ����������ҵ��ĸ��ڵ��λ��
        ����ҿ�
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
        //�������Ѱ�Ҹ��ڵ��λ�ã��ҵ�ֵΪval�����
        for (i = istart; i <= iend; i++) {
            if (inorder[i] == head->val) {
                break;
            }
        }

        int leftNum = i - istart;
        int rightNum = iend - i;
        //�������ķ�Χ[pstart,i-1]
        head->left = getHead(pstart + 1, pstart + leftNum, istart, istart + leftNum - 1, preorder, inorder);
        //�������ķ�Χ[i+1,pend]
        head->right = getHead(1 + pstart + leftNum, pend, istart + leftNum + 1, iend, preorder, inorder);
        return head;
    }
};
/*
 * @lc app=leetcode.cn id=106 lang=cpp
 *
 * [106] �����������������й��������
 * [9,3,15,20,7]
 * [9,15,7,20,3]
 * [3,9,20,null,null,15,7]
 */

class Solution106 {
public:
    unordered_map<int, int> map;
    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
        if (inorder.size() != postorder.size()) {
            //��������
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
        // ���������istart+i��Ϊͷ���
        TreeNode* head = new TreeNode(inorder[iStart + i]);
        head->left = findHead(iStart, iStart + i - 1, pStart, pStart + i - 1, inorder, postorder);
        head->right = findHead(iStart + i + 1, iEnd, pStart + i, pEnd - 1, inorder, postorder);
        return head;
    }
};
/*
����: 5
���:
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
���룺triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
�����11
*/
class Solution120 {
public:
	int minimumTotal(vector<vector<int>>& triangle) {
		//ֻ��һ�л���0��
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
				//����
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
//���ʽ���
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
	// ǰ��Ŀ϶�Ҫ+K������Ŀ϶�Ҫ-K�������Կ��ǵĻ�����ǰ��Ĳ��䣬�����-2K
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
	//���ұߵĿ϶�С�ڵ�������ߵ�
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
	//�ҿ�
	int findMin(vector<int>& nums) {
		int left = 0;
		int right = nums.size() - 1;
		int mid = 0;
		while (left < right) {
			mid = (left + right) / 2;
			if (nums[mid] >= nums[left]) {
				//�����
				if (mid + 1 <= right && nums[mid] > nums[mid + 1]) {
					return nums[mid + 1];
				}
				else {
					left = mid + 1;
				}
			}
			else if (nums[mid] < nums[right])
			{
				//�ұ���
				if (mid - 1 >= left && nums[mid - 1] > nums[mid]) {
					return nums[mid];
				}
				else {
					right = mid - 1;
				}
			}
			else {
				//���ߵ�
				left++;
			}

		}
		return nums[0];
	}
};