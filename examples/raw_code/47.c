struct node
{
	int s,n;
};
void cpy_clr(struct node a[],struct node b[])
{
	int i;
	for(i=0;i<=MAX&&(a[i].s!=0||b[i].s!=0);i++)
		if(b[i].s!=0)
		{a[i].s=b[i].s;a[i].n=b[i].n;b[i].s=0;b[i].n=0;}
		else
		{a[i].s=0;a[i].n=0;b[i].n=0;}
}
int chklst(struct node a[],int n)
{
	int i;
	int result=0;
	for(i=0;a[i].s!=0;i++)
	{	if(a[i].s!=n){result=0;break;}
		else result++;
	}
	return result;
}
void init(struct node a[])
{
	int i;
	for(i=0;i<MAX;i++) {a[i].s=0;a[i].n=0;}
}
void main()
{
	struct node a[MAX],b[MAX];	
	int i,j,k,l,_n,n,sum=0;
	scanf("%d",&n);
	while(n--)
	{
	scanf("%d",&_n);
	for(l=2;l<_n;l++) 
		if(_n%l==0)	
		{	init(a);init(b); 
			a[0].s=l;a[0].n=l;
			while(chklst(a,_n)==0&&a[0].s!=0)
			{
				i=0;k=-1;
				if(a[0].s==0) break;
				while(a[i].s!=0)
				{
					if(a[i].s==_n) sum++;
					else
					{
						for(j=a[i].n;j<=(_n/a[i].s);j++)
							if(_n%(a[i].s*j)==0)
							{b[++k].s=a[i].s*j;b[k].n=j;}
					}
					i++;
				}
				cpy_clr(a,b);
			}
			i=0;
			while(a[i].s!=0)
			{
				if(a[i].s==_n) sum++;
					i++;
			}
		} 
	printf("%d\n",++sum);
	sum=0;
	}
}
