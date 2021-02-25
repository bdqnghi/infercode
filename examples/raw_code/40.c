
int main()
{
	int f(int x, int y);
	int n,m,i,z;

	scanf("%d",&n);
	for(i=0;i<n;i++)
	{
		scanf("%d",&m);
		z=f(1,m);
		printf("%d\n",z);
	}
	
	return 0;
}

int f(int x, int y)
{
	int z,t=1,l=0,p;
	p=(int) pow(y,0.5);
	if(x==1) x=2;
	for(z=x;z<=p;z++)
	{		
		if(y%z==0 && z!=y)
		{
			t+=f(z,y/z);
		}	
	}

	return t;
}