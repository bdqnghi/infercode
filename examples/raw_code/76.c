int main()
{
	int fun(int m,int n);
	int n,i,sum;
	scanf("%d",&n);
	int *p;
	p=(int *)malloc(sizeof(int)*n);
	for(i=0;i<n;i++)
	scanf("%d",&p[i]);
	for(i=0;i<n;i++)
	{
		sum=0;
	sum=fun(p[i],1);
	printf("%d\n",sum);
	}
	free(p);
}
int fun(int m,int n)
{
	int i,sum=1;
	if(n==1)
	{
		for(i=2;i*i<=m;i++)
		{
			if(m%i==0)
			sum=sum+fun(m/i,i);
		}
	}
	else
	{
		for(i=n;i*i<=m;i++)
		{
			if(m%i==0)
			sum=sum+fun(m/i,i);
		}
	}
	return sum;
}