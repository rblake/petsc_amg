/******************************************************************************
 *
 * Queue implementation using double link list struct.
 *
 * -David Alber, January 2005
 *
 *****************************************************************************/

#ifndef QUEUE_H
#define QUEUE_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "utilities.h"

struct queue {
  hypre_QueueElement * head;
  hypre_QueueElement * tail;
  int length;
};
typedef struct queue hypre_Queue;  

struct queue_element {
       void                 *data;
       struct queue_element *next_elt;
       struct queue_element *prev_elt;
       int                  head;
       int                  tail;

       int                  own_element; // if 1, the queue code destroys
                                         // the element when it is dequeued
                                         // or the queue is destroyed
};
typedef struct queue_element hypre_QueueElement;

hypre_Queue * newQueue();
void destroyQueue(hypre_Queue * queue);
void enqueue(void * data, hypre_Queue * queue);
void * dequeue(hypre_Queue * queue);
void removeElement(hypre_QueueElement * source, hypre_Queue * queue);
void moveAfter(hypre_QueueElement * source, hypre_QueueElement * dest, hypre_Queue * queue);
void moveToHead(hypre_QueueElement * source, hypre_Queue * queue);

#endif
