#include "Solutions.h"


int main() 
{
	Solution498 solution;
	vector<vector<int>> matrix =
	{
		{1,2,3},
		{4,5,6},
		{7,8,9}
	};
	vector<int> result = solution.findDiagonalOrder(matrix);
	//cout << result << endl;
	printVector(result);

	return 0;
}