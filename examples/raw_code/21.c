int t;
void calc(int a,int d)
{
 int i;
 if (a==1) t++;
   else{
    if (a>=d)
      for (i=d;i<=a;i++)
	if (a%i==0)
	  calc(a/i,i);}
}
void main()
{
 int n,i,a;
 scanf("%d",&n);
 for (i=1;i<=n;i++)
   {
   scanf("%d",&a);
   t=0;
   calc(a,2);
   printf("%d\n",t);
   }
}