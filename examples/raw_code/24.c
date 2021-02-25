int a[1000];
int l=0,x;
int fff(int k,int w,int t,int x);
int main()
{
    int i,j,q,p,n,m,y;
    scanf("%d",&n);
    for(p=0;p<n;p++)
       {q=1;l=0;
        scanf("%d",&x);
        for(i=2;i<=x;i++)
           if(x%i==0)
             {a[q]=i;
             q++;
             }
        fff(1,q-1,1,x);
        printf("%d\n",l,x);
       }
}
int fff(int k,int w,int t,int x)
{
    int i,j,y,n,m,p;
      for(i=t;i<=w;i++)
            {if(x%a[i]==0)
             {
             x=x/a[i]; 
             if(x==1){l++;}
             if(x>1)
                 fff(k+1,w,i,x);
             x=x*a[i];
             }
            }
}

            
