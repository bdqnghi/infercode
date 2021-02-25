
int calc(int p,int t);

int main()
{
	int  m,n,ans;


	scanf("%d",&m);

	while (m--)
	{
		scanf("%d",&n);
		ans=calc(2,n);
		printf("%d\n",ans);   
	}


	return 0;

}

int calc(int p,int t)
{
	int i,a;
	
	a=1;
	for (i=p;i<=sqrt(t);i++)
		if (t%i==0) 
			a=a+calc(i,t/i);
	return a;
	
}