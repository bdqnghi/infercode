bool isPalindrome( char array[] )
{
    bool isPalindrome = true;
    int size = 0, index = 0, startingPos = 0, count1 = 0;   
    // Step 1
    while ( array[size] != '\0' ) 
    {                              
            size++;
    }   
    char array1[size + 1];
    // Step 2
    while ( index < size ) 
    {                     
            while ( (array[index] >= 'A' and array[index] <= 'Z') or
                    (array[index] >= 'a' and array[index] <= 'z')    )
                    index++;
            for ( int count = startingPos; count < index; count++ )
            {
                    array1[count1] = array[count];
                    count1++;
            }
            index++;
            startingPos = index;
    }
    array1[count1] = '\0';  
    //Step 3
    index = 0;
    while ( index <= (count1 - 1)/2 and isPalindrome )
    {
            if ( array1[index] != array1[count1 - index - 1]      and
                 array1[index] != array1[count1 - index - 1] - 32 and
                 array1[index] != array1[count1 - index - 1] + 32 and
                 array1[index] - 32 != array1[count1 - index - 1] and
                 array1[index] + 32 != array1[count1 - index - 1]     )
                    isPalindrome = false;
            index++;
    }
    // Step 4
    return isPalindrome;
}