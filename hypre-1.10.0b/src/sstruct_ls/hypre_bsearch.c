
/*--------------------------------------------------------------------------
 * hypre_LowerBinarySearch
 * performs a binary search for a value over a list of ordered non-negative
 * integers such that
 *      list[m-1] < value <= list[m].
 * The routine returns location m or -1.
 *--------------------------------------------------------------------------*/
int hypre_LowerBinarySearch(int *list, int value, int list_length)
{
   int low, high, m;
   int not_found = 1;

   /* special case, list is size zero. */
   if (list_length < 1)
   {
      return -1;
   }

   /* special case, list[0] >= value */
   if (list[0] >= value)
   {
      return 0;
   }

   low = 0;
   high= list_length-1;
   while (not_found && low <= high)
   {
      m = (low + high) / 2;
      if (m < 1)
      {
         m= 1;
      }

      if (list[m-1] < value && list[m] < value)
      {
         low= m + 1;
      }
      else if (value <= list[m-1] && value <= list[m])
      {
         high= m - 1;
      }
      else
      {
        not_found = 0;
        return m;
      }
   }
   return -1;
}

/*--------------------------------------------------------------------------
 * hypre_UpperBinarySearch
 * performs a binary search for a value over a list of ordered non-negative
 * integers such that
 *      list[m] <= value < list[m+1].
 * The routine returns location m or -1.
 *--------------------------------------------------------------------------*/
int hypre_UpperBinarySearch(int *list, int value, int list_length)
{
   int low, high, m;
   int not_found = 1;

   /* special case, list is size zero. */
   if (list_length < 1)
   {
      return -1;
   }

   /* special case, list[list_length-1] >= value */
   if (list[list_length-1] <= value)
   {
      return (list_length-1);
   }

   low = 0;
   high= list_length-1;
   while (not_found && low <= high)
   {
      m = (low + high) / 2;
      if (list[m] <= value && list[m+1] <= value)
      {
         low= m + 1;
      }
      else if (value < list[m] && value < list[m+1])
      {
         high= m - 1;
      }
      else
      {
        not_found = 0;
        return m;
      }
   }

   return -1;
}
