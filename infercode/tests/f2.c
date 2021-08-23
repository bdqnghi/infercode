int main()
{
    int clear_i;
    int i, left_i, cur_sz, cur_num;
    n_obj **cl;
    n_obj *temp;
    n_obj *cur_list;
    n_obj *cur_seq;
     
    cl = malloc(sizeof(n_obj*)*NUM_OBJECTS);
     
    for(i=0;i<NUM_OBJECTS;i++)
    {
        cl[i] = Q_OBJ;
        cl[i]->val = NULL;
         
        temp = Q_OBJ;
        temp->val = Q_SEQ(1);
         
        temp->val[0] = i + 1;
         
        temp->next = cl[i]; 
        cl[i] = temp;
         
        cur_list = cl[i];
        for(left_i=0; left_i<i; left_i++)
        {
            for(cur_seq=cl[left_i];cur_seq->val!=NULL;cur_seq=cur_seq->next)
            {               
                if(cur_seq->val[0]<=(i - left_i))
                {
                    temp = Q_OBJ;
                    temp->val = Q_SEQ(left_i + 3);
                     
                    temp->val[0] =  i - left_i;
                     
                    temp->next = cur_list->next;
                    cur_list->next = temp; 
                    cur_list = temp;
                     
                    for(cur_num=0;cur_num<=left_i;cur_num++)
                        cur_list->val[cur_num+1] = cur_seq->val[cur_num];
                }
            }
        }
    }
     
    for(i = 0; i < NUM_OBJECTS; i++)
    {
        cur_sz = 0;
        printf(\"---------------------------\n\");
        for(cur_list = cl[i]; cur_list->val != NULL; cur_list = cur_list->next)
        {
            for(clear_i = 0; cur_list->val[clear_i] != 0; clear_i++)
            {
                printf(\"%d \", cur_list->val[clear_i]);
            }
            printf(\"\n\");
            cur_sz++;
        }
        printf(\"Number of partitions = %d\n\", cur_sz);
     
    }
    return 0;
}
