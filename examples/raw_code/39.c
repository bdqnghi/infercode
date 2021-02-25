int x[1000]={0};
void f(int m,int n,int l)
{
	int i;
	for(i=m;i<=n;i++)
	{
		if(n%i!=0) continue;
		else
		{
			if(n/i==1)
			{
				x[l]++;
				return;
			}
			else f(i,n/i,l);
		}
	}	
}

int main()
{
	int a[1000],s,j,sum;
	cin>>sum;
	for(j=1;j<=sum;j++)
	{
		cin>>a[j];
		s=2;
	    f(s,a[j],j);
	}
	for(j=1;j<=sum;j++)
	{
		cout<<x[j]<<endl;
	}
	return 0;
}