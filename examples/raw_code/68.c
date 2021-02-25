
int f(int a,int b)
{
	if(a < b)
		return 0;
	int i,r =1 ;
	for(i = b; i <= a; i++ )
	{
		if( !(a%i) && i <= a/i )
		{
			r += f(a/i,i);
		}
	}
	return r;
}
void main()
{
	int n,m;
	scanf("%d",&n);
	while(n--)
	{
		scanf("%d",&m);
		int result = f(m,2);
		printf("%d\n",result);
	}
}
