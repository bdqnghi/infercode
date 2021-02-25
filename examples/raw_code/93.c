/**
 * author:liucheng
 * date:2010-10-23 13:04:16
 * file:factors.cpp
 */

int factor(int num,int minA1)
{
        int count=0;
        int sqrtNum;
        int i;

        if(num <= minA1)
                return 0;

        sqrtNum=(int)sqrt((double)num);
        for(i=minA1;i<=sqrtNum;i++)
                if(num % i == 0)
                {
                        count++;//num=i*(num/i)
                        count+=factor(num/i,i);//num=i*(factor(num/i))
                }

        return count;
}


int main()
{
        int n;
        int num;

        cin>>n;

        while(n -- )
        {
                cin>>num;
                cout<<factor(num,2)+1<<endl;
        }

        return 0;
}