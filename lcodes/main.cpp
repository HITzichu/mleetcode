#include "Solutions.h"


int main() 
{
	Solution93 solution;
	string s = "19216811";
	vector<string> result = solution.restoreIpAddresses(s);
	//cout << result << endl;
	for (auto it = result.begin(); it != result.end(); it++) {
		for (auto iit = it->begin(); iit != it->end(); iit++) {
			cout << *iit;
		}
		cout << endl;
	}

	return 0;
}