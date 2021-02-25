int main(){
	int number(int n, int low);
	int n, i, data, ans;
	scanf("%d", &n);
	for (i=1; i<=n; i++)
	{
		scanf("%d", &data);
		ans = number(data, 2);
		printf("%d\n", ans);
	}
}
int number(int n, int low)
{
	int j, ans;
	ans = 0;
	for (j = low ; j * j<= n; j++)
	{
		if (n % j == 0) 
			ans = ans + number(n / j, j);
	}
	ans = ans + 1;
	return ans;
}